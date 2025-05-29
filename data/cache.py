import redis
import json
import pickle
import time
import hashlib
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import asyncio

from dataModels.data_models import Job, Resume

logger = logging.getLogger(__name__)

class ResumeGenerationStatus(Enum):
    """Resume generation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class RedisCache:
    """
    Unified Redis cache for jobs, resumes, search results, and resume generation status.
    Uses different key prefixes to organize different types of cached data.
    """

    def __init__(self, redis_url="redis://localhost:6379", default_ttl=86400):
        """
        Initialize unified Redis cache.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        self.default_ttl = default_ttl

        # Key prefixes for different data types
        self.prefixes = {
            # Job cache
            'job': 'job:',
            'job_url_index': 'job_url:',
            'job_signature_index': 'job_sig:',
            'job_user_index': 'job_user:',

            # Resume cache
            'resume': 'resume:',
            'resume_user_index': 'resume_user:',
            'resume_job_index': 'resume_job:',

            # Search cache
            'search': 'search:',
            'search_user_index': 'search_user:',

            # Resume generation status
            'resume_status': 'resume_status:',

            # Cache statistics
            'stats': 'cache_stats'
        }

        # Statistics tracking
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    # ============= JOB CACHE METHODS =============

    def add_job(self, job: Job, user_id: str) -> None:
        """Add job to Redis cache with indexing."""
        try:
            job_key = f"{self.prefixes['job']}{user_id}:{job.id}"

            # Remove old entry if exists
            if self.redis_client.exists(job_key):
                self._remove_job_entry(job.id, user_id)

            # Serialize job with metadata
            job_data = {
                'job': job,
                'cached_at': time.time(),
                'access_count': 0,
                'user_id': user_id
            }

            pipe = self.redis_client.pipeline()

            # Store main job entry
            pipe.setex(job_key, self.default_ttl, pickle.dumps(job_data))

            # Update URL index
            if job.job_url:
                url_key = f"{self.prefixes['job_url_index']}{user_id}:{hashlib.md5(job.job_url.encode()).hexdigest()}"
                pipe.setex(url_key, self.default_ttl, f"{user_id}:{job.id}")

            # Update signature index for duplicate detection
            signature = self._generate_job_signature(job, user_id)
            sig_key = f"{self.prefixes['job_signature_index']}{signature}"
            pipe.setex(sig_key, self.default_ttl, f"{user_id}:{job.id}")

            # Update user index (set of job IDs)
            user_jobs_key = f"{self.prefixes['job_user_index']}{user_id}"
            pipe.sadd(user_jobs_key, job.id)
            pipe.expire(user_jobs_key, self.default_ttl)

            pipe.execute()

            logger.debug(f"Added job {job.id} to Redis cache for user {user_id}")

        except Exception as e:
            logger.error(f"Error adding job to Redis cache: {e}")

    def get_job(self, job_id: str, user_id: str) -> Optional[Job]:
        """Get job from Redis cache."""
        try:
            job_key = f"{self.prefixes['job']}{user_id}:{job_id}"

            serialized_data = self.redis_client.get(job_key)
            if not serialized_data:
                self._stats['misses'] += 1
                return None

            job_data = pickle.loads(serialized_data)

            # Update access statistics
            job_data['access_count'] += 1
            job_data['last_accessed'] = time.time()
            self._stats['hits'] += 1

            # Update cache entry asynchronously
            try:
                self.redis_client.setex(job_key, self.default_ttl, pickle.dumps(job_data))
            except Exception as e:
                logger.warning(f"Error updating job access stats: {e}")

            return job_data['job']

        except Exception as e:
            logger.error(f"Error getting job from Redis cache: {e}")
            self._stats['misses'] += 1
            return None

    def get_job_by_url(self, url: str, user_id: str) -> Optional[Job]:
        """Get job by URL using Redis index."""
        try:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            url_key = f"{self.prefixes['job_url_index']}{user_id}:{url_hash}"

            job_cache_id = self.redis_client.get(url_key)
            if not job_cache_id:
                self._stats['misses'] += 1
                return None

            job_cache_id = job_cache_id.decode('utf-8')
            job_id = job_cache_id.split(':', 1)[1]
            return self.get_job(job_id, user_id)

        except Exception as e:
            logger.error(f"Error getting job by URL from Redis cache: {e}")
            self._stats['misses'] += 1
            return None

    def find_similar_job(self, job: Job, user_id: str) -> Optional[Job]:
        """Find similar job using signature matching."""
        try:
            signature = self._generate_job_signature(job, user_id)
            sig_key = f"{self.prefixes['job_signature_index']}{signature}"

            job_cache_id = self.redis_client.get(sig_key)
            if not job_cache_id:
                return None

            job_cache_id = job_cache_id.decode('utf-8')
            job_id = job_cache_id.split(':', 1)[1]
            return self.get_job(job_id, user_id)

        except Exception as e:
            logger.error(f"Error finding similar job in Redis cache: {e}")
            return None

    def get_user_jobs(self, user_id: str, limit: int = None) -> List[Job]:
        """Get all jobs for a user."""
        try:
            user_jobs_key = f"{self.prefixes['job_user_index']}{user_id}"
            job_ids = self.redis_client.smembers(user_jobs_key)

            jobs = []
            for job_id in job_ids:
                job_id = job_id.decode('utf-8')
                job = self.get_job(job_id, user_id)
                if job:
                    jobs.append(job)
                    if limit and len(jobs) >= limit:
                        break

            return jobs

        except Exception as e:
            logger.error(f"Error getting user jobs from Redis cache: {e}")
            return []

    def remove_job(self, job_id: str, user_id: str) -> None:
        """Remove job from Redis cache."""
        self._remove_job_entry(job_id, user_id)

    def _remove_job_entry(self, job_id: str, user_id: str) -> None:
        """Internal method to remove job and all its indexes."""
        try:
            job_key = f"{self.prefixes['job']}{user_id}:{job_id}"

            # Get job data to clean up indexes
            serialized_data = self.redis_client.get(job_key)
            if serialized_data:
                job_data = pickle.loads(serialized_data)
                job = job_data['job']

                pipe = self.redis_client.pipeline()

                # Remove main entry
                pipe.delete(job_key)

                # Remove from URL index
                if job.job_url:
                    url_hash = hashlib.md5(job.job_url.encode()).hexdigest()
                    url_key = f"{self.prefixes['job_url_index']}{user_id}:{url_hash}"
                    pipe.delete(url_key)

                # Remove from signature index
                signature = self._generate_job_signature(job, user_id)
                sig_key = f"{self.prefixes['job_signature_index']}{signature}"
                pipe.delete(sig_key)

                # Remove from user index
                user_jobs_key = f"{self.prefixes['job_user_index']}{user_id}"
                pipe.srem(user_jobs_key, job_id)

                pipe.execute()

                logger.debug(f"Removed job {job_id} from Redis cache for user {user_id}")

        except Exception as e:
            logger.error(f"Error removing job from Redis cache: {e}")

    # ============= RESUME CACHE METHODS =============

    def add_resume(self, resume: Resume, user_id: str) -> None:
        """Add resume to Redis cache."""
        try:
            resume_key = f"{self.prefixes['resume']}{user_id}:{resume.id}"

            resume_data = {
                'resume': resume,
                'cached_at': time.time(),
                'access_count': 0,
                'user_id': user_id
            }

            pipe = self.redis_client.pipeline()

            # Store main resume entry
            pipe.setex(resume_key, self.default_ttl, pickle.dumps(resume_data))

            # Update user index
            user_resumes_key = f"{self.prefixes['resume_user_index']}{user_id}"
            pipe.sadd(user_resumes_key, resume.id)
            pipe.expire(user_resumes_key, self.default_ttl)

            # Update job index if resume is associated with a job
            if resume.job_id:
                job_resumes_key = f"{self.prefixes['resume_job_index']}{user_id}:{resume.job_id}"
                pipe.sadd(job_resumes_key, resume.id)
                pipe.expire(job_resumes_key, self.default_ttl)

            pipe.execute()

            logger.debug(f"Added resume {resume.id} to Redis cache for user {user_id}")

        except Exception as e:
            logger.error(f"Error adding resume to Redis cache: {e}")

    def get_resume(self, resume_id: str, user_id: str) -> Optional[Resume]:
        """Get resume from Redis cache."""
        try:
            resume_key = f"{self.prefixes['resume']}{user_id}:{resume_id}"

            serialized_data = self.redis_client.get(resume_key)
            if not serialized_data:
                self._stats['misses'] += 1
                return None

            resume_data = pickle.loads(serialized_data)

            # Update access statistics
            resume_data['access_count'] += 1
            resume_data['last_accessed'] = time.time()
            self._stats['hits'] += 1

            # Update cache entry
            try:
                self.redis_client.setex(resume_key, self.default_ttl, pickle.dumps(resume_data))
            except Exception as e:
                logger.warning(f"Error updating resume access stats: {e}")

            return resume_data['resume']

        except Exception as e:
            logger.error(f"Error getting resume from Redis cache: {e}")
            self._stats['misses'] += 1
            return None

    def get_user_resumes(self, user_id: str, job_id: Optional[str] = None) -> List[Resume]:
        """Get all resumes for a user, optionally filtered by job."""
        try:
            if job_id:
                # Get resumes for specific job
                job_resumes_key = f"{self.prefixes['resume_job_index']}{user_id}:{job_id}"
                resume_ids = self.redis_client.smembers(job_resumes_key)
            else:
                # Get all user resumes
                user_resumes_key = f"{self.prefixes['resume_user_index']}{user_id}"
                resume_ids = self.redis_client.smembers(user_resumes_key)

            resumes = []
            for resume_id in resume_ids:
                resume_id = resume_id.decode('utf-8')
                resume = self.get_resume(resume_id, user_id)
                if resume:
                    resumes.append(resume)

            return resumes

        except Exception as e:
            logger.error(f"Error getting user resumes from Redis cache: {e}")
            return []

    def remove_resume(self, resume_id: str, user_id: str) -> None:
        """Remove resume from Redis cache."""
        try:
            resume_key = f"{self.prefixes['resume']}{user_id}:{resume_id}"

            # Get resume data to clean up indexes
            serialized_data = self.redis_client.get(resume_key)
            if serialized_data:
                resume_data = pickle.loads(serialized_data)
                resume = resume_data['resume']

                pipe = self.redis_client.pipeline()

                # Remove main entry
                pipe.delete(resume_key)

                # Remove from user index
                user_resumes_key = f"{self.prefixes['resume_user_index']}{user_id}"
                pipe.srem(user_resumes_key, resume_id)

                # Remove from job index if applicable
                if resume.job_id:
                    job_resumes_key = f"{self.prefixes['resume_job_index']}{user_id}:{resume.job_id}"
                    pipe.srem(job_resumes_key, resume_id)

                pipe.execute()

                logger.debug(f"Removed resume {resume_id} from Redis cache for user {user_id}")

        except Exception as e:
            logger.error(f"Error removing resume from Redis cache: {e}")

    # ============= SEARCH CACHE METHODS =============

    def add_search_results(self, keywords: str, location: str, filters: Dict[str, Any],
                           job_ids: List[str], user_id: str) -> None:
        """Add search results to Redis cache."""
        try:
            search_key = self._generate_search_key(keywords, location, filters, user_id)
            cache_key = f"{self.prefixes['search']}{search_key}"

            search_data = {
                'keywords': keywords,
                'location': location,
                'filters': filters,
                'job_ids': job_ids,
                'user_id': user_id,
                'cached_at': time.time(),
                'access_count': 0
            }

            pipe = self.redis_client.pipeline()

            # Store search results with shorter TTL (searches expire faster)
            pipe.setex(cache_key, self.default_ttl // 2, pickle.dumps(search_data))

            # Update user search index
            user_searches_key = f"{self.prefixes['search_user_index']}{user_id}"
            pipe.sadd(user_searches_key, search_key)
            pipe.expire(user_searches_key, self.default_ttl // 2)

            pipe.execute()

            logger.debug(f"Added search results to Redis cache for user {user_id}")

        except Exception as e:
            logger.error(f"Error adding search results to Redis cache: {e}")

    def get_search_results(self, keywords: str, location: str, filters: Dict[str, Any],
                           user_id: str) -> Optional[List[str]]:
        """Get search results from Redis cache."""
        try:
            search_key = self._generate_search_key(keywords, location, filters, user_id)
            cache_key = f"{self.prefixes['search']}{search_key}"

            serialized_data = self.redis_client.get(cache_key)
            if not serialized_data:
                self._stats['misses'] += 1
                return None

            search_data = pickle.loads(serialized_data)

            # Update access statistics
            search_data['access_count'] += 1
            search_data['last_accessed'] = time.time()
            self._stats['hits'] += 1

            # Update cache entry
            try:
                self.redis_client.setex(cache_key, self.default_ttl // 2, pickle.dumps(search_data))
            except Exception as e:
                logger.warning(f"Error updating search access stats: {e}")

            return search_data['job_ids']

        except Exception as e:
            logger.error(f"Error getting search results from Redis cache: {e}")
            self._stats['misses'] += 1
            return None

    # ============= RESUME STATUS CACHE METHODS =============

    async def set_resume_status(self, resume_id: str, user_id: str, status: ResumeGenerationStatus,
                                data: Optional[Dict] = None, error: Optional[str] = None) -> None:
        """Set resume generation status in Redis cache."""
        try:
            status_key = f"{self.prefixes['resume_status']}{user_id}:{resume_id}"

            status_data = {
                'status': status,
                'data': data,
                'error': error,
                'updated_at': time.time(),
                'user_id': user_id,
                'resume_id': resume_id
            }

            # Store with shorter TTL for status data
            self.redis_client.setex(status_key, self.default_ttl // 4, pickle.dumps(status_data))

            logger.debug(f"Set resume status {status} for resume {resume_id} user {user_id}")

        except Exception as e:
            logger.error(f"Error setting resume status in Redis cache: {e}")

    async def get_resume_status(self, resume_id: str, user_id: str) -> Optional[Dict]:
        """Get resume generation status from Redis cache."""
        try:
            status_key = f"{self.prefixes['resume_status']}{user_id}:{resume_id}"

            serialized_data = self.redis_client.get(status_key)
            if not serialized_data:
                return None

            status_data = pickle.loads(serialized_data)
            return status_data

        except Exception as e:
            logger.error(f"Error getting resume status from Redis cache: {e}")
            return None

    async def remove_resume_status(self, resume_id: str, user_id: str) -> None:
        """Remove resume generation status from Redis cache."""
        try:
            status_key = f"{self.prefixes['resume_status']}{user_id}:{resume_id}"
            self.redis_client.delete(status_key)

        except Exception as e:
            logger.error(f"Error removing resume status from Redis cache: {e}")

    # ============= CACHE MANAGEMENT METHODS =============

    def clear_user_cache(self, user_id: str) -> None:
        """Clear all cache data for a user."""
        try:
            pipe = self.redis_client.pipeline()

            # Clear user jobs
            user_jobs_key = f"{self.prefixes['job_user_index']}{user_id}"
            job_ids = self.redis_client.smembers(user_jobs_key)
            for job_id in job_ids:
                job_id = job_id.decode('utf-8')
                self._remove_job_entry(job_id, user_id)

            # Clear user resumes
            user_resumes_key = f"{self.prefixes['resume_user_index']}{user_id}"
            resume_ids = self.redis_client.smembers(user_resumes_key)
            for resume_id in resume_ids:
                resume_id = resume_id.decode('utf-8')
                self.remove_resume(resume_id, user_id)

            # Clear user searches
            user_searches_key = f"{self.prefixes['search_user_index']}{user_id}"
            search_keys = self.redis_client.smembers(user_searches_key)
            for search_key in search_keys:
                search_key = search_key.decode('utf-8')
                cache_key = f"{self.prefixes['search']}{search_key}"
                pipe.delete(cache_key)

            # Clear user indexes
            pipe.delete(user_jobs_key)
            pipe.delete(user_resumes_key)
            pipe.delete(user_searches_key)

            # Clear resume status entries
            pattern = f"{self.prefixes['resume_status']}{user_id}:*"
            for key in self.redis_client.scan_iter(match=pattern):
                pipe.delete(key)

            pipe.execute()

            logger.info(f"Cleared all cache data for user {user_id}")

        except Exception as e:
            logger.error(f"Error clearing user cache: {e}")

    async def cleanup_expired_cache(self) -> None:
        """Clean up expired cache entries (Redis handles TTL automatically)."""
        try:
            # Redis automatically handles TTL expiration
            # But we can do additional cleanup for consistency
            logger.info("Redis TTL handles automatic cleanup")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            info = self.redis_client.info()

            return {
                'redis_info': {
                    'used_memory': info.get('used_memory_human', 'N/A'),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                },
                'application_stats': {
                    'hits': self._stats['hits'],
                    'misses': self._stats['misses'],
                    'evictions': self._stats['evictions'],
                    'hit_rate': (self._stats['hits'] / max(self._stats['hits'] + self._stats['misses'], 1)) * 100
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis cache."""
        try:
            # Test Redis connection
            self.redis_client.ping()

            info = self.redis_client.info()

            return {
                'status': 'healthy',
                'redis_version': info.get('redis_version', 'unknown'),
                'used_memory': info.get('used_memory_human', 'N/A'),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0)
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    # ============= HELPER METHODS =============

    def _generate_job_signature(self, job: Job, user_id: str) -> str:
        """Generate job signature for similarity matching."""
        metadata = job.metadata or {}
        title = metadata.get('title', '').lower().strip()
        company = metadata.get('company', '').lower().strip()
        location = metadata.get('location', '').lower().strip()

        signature = f"{user_id}|{title}|{company}|{location}"
        return hashlib.md5(signature.encode('utf-8')).hexdigest()

    def _generate_search_key(self, keywords: str, location: str, filters: Dict[str, Any], user_id: str) -> str:
        """Generate search key for caching search results."""
        keywords_norm = ' '.join(keywords.lower().split())
        location_norm = ' '.join(location.lower().split())
        filters_str = json.dumps(filters, sort_keys=True)

        key_str = f"{user_id}|{keywords_norm}|{location_norm}|{filters_str}"
        return hashlib.md5(key_str.encode()).hexdigest()