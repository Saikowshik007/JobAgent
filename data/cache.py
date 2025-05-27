from typing import Dict, List, Optional, Any
import hashlib
import json
import logging
import threading

# Configure logging
logger = logging.getLogger(__name__)

class DBCacheManager:
    """
    Unified cache manager that handles all caching operations including jobs, search results, and resumes.
    This manager provides a single interface for all database and cache interactions with consistent caching patterns.
    """

    # Thread local storage to ensure thread safety
    _thread_local = threading.local()

    def __init__(self, database=None, job_cache=None, search_cache=None, resume_cache=None):
        """
        Initialize the unified cache manager with database and cache instances.

        Args:
            database: Database instance
            job_cache: JobCache instance
            search_cache: SearchCache instance (optional)
            resume_cache: ResumeCache instance (optional, will be created if not provided)
        """
        self.db = database
        self.job_cache = job_cache
        self.search_cache = search_cache

        # Initialize resume cache if not provided
        if resume_cache is None:
            from data.cache import ResumeCache
            self.resume_cache = ResumeCache()
        else:
            self.resume_cache = resume_cache

        # Add in-memory resume storage for consistency with job caching
        self._resume_memory_cache = {}
        self._resume_cache_lock = threading.RLock()

    # Job Management Methods
    async def get_cached_search_results(self, keywords: str, location: str, filters: Dict[str, Any], user_id: str) -> List[Dict[str, Any]]:
        """
        Get search results from cache or database.
        Implementation of the method expected by JobSearcher.

        This method follows this process:
        1. Try to get results from in-memory search cache
        2. If not found, try to get from database
        3. If found in database, update in-memory cache

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            user_id: ID of the user who performed the search

        Returns:
            List of job details dictionaries
        """
        # Skip if no caching is available
        if not self.search_cache and not self.db:
            return []

        # Generate a search key for consistent identification
        search_key = self._generate_search_key(keywords, location, filters, user_id)

        # Try in-memory cache first (fastest)
        if self.search_cache:
            cache_results = self.search_cache.get_search_results(keywords, location, filters, user_id)
            if cache_results:
                # Get full job details from job cache
                results = []
                for job_id in cache_results:
                    if self.job_cache:
                        job = self.job_cache.get_job(job_id, user_id)
                        if job:
                            results.append(job.to_dict())

                # If we have all the jobs in the cache, return them
                if len(results) == len(cache_results):
                    return results

        # If not in memory cache or incomplete, try database
        if self.db:
            try:
                db_results = await self.db.get_cached_search_results(keywords, location, filters, user_id)
                if db_results:
                    # Update in-memory caches with these results
                    if self.search_cache:
                        job_ids = [job["id"] for job in db_results if job.get("id")]
                        self.search_cache.add_search_results(keywords, location, filters, job_ids, user_id)

                    # Add individual jobs to job cache
                    if self.job_cache:
                        for job_dict in db_results:
                            # Convert dict to Job object
                            from dataModels.data_models import Job
                            job = Job.from_dict(job_dict)
                            self.job_cache.add_job(job, user_id)

                    return db_results
            except Exception as e:
                logger.error(f"Error retrieving cached search results for user {user_id}: {e}")

        # No results found in either cache
        return []

    async def save_search_results(self, keywords: str, location: str, filters: Dict[str, Any],
                                  job_listings: List[Dict[str, Any]], user_id: str) -> bool:
        """
        Save search results to both database and in-memory cache.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            job_listings: Job listing dictionaries
            user_id: ID of the user who performed the search

        Returns:
            bool: True if successful, False otherwise
        """
        # Skip if no caching is available
        if not self.search_cache and not self.db:
            return False

        # Extract job IDs from listings
        job_ids = [job.get("id") for job in job_listings if job.get("id")]

        # Save to database
        success = True
        if self.db:
            try:
                search_id = self._generate_search_key(keywords, location, filters, user_id)
                # Save individual jobs first to ensure they exist
                for job_dict in job_listings:
                    # Convert dict to Job object
                    from dataModels.data_models import Job
                    try:
                        job = Job.from_dict(job_dict)
                        await self.db.save_job(job, user_id)
                    except Exception as e:
                        logger.error(f"Error saving job to database for user {user_id}: {e}")
                        success = False

                # Then save the search history
                success = success and await self.db.save_search_history(keywords, location, filters, job_ids, user_id, search_id)
            except Exception as e:
                logger.error(f"Error saving search history to database for user {user_id}: {e}")
                success = False

        # Save to in-memory cache
        try:
            if self.search_cache:
                self.search_cache.add_search_results(keywords, location, filters, job_ids, user_id)

            # Add individual jobs to job cache
            if self.job_cache:
                for job_dict in job_listings:
                    # Convert dict to Job object
                    from dataModels.data_models import Job
                    try:
                        job = Job.from_dict(job_dict)
                        self.job_cache.add_job(job, user_id)
                    except Exception as e:
                        logger.error(f"Error adding job to cache for user {user_id}: {e}")
                        success = False
        except Exception as e:
            logger.error(f"Error updating in-memory caches for user {user_id}: {e}")
            success = False

        return success

    async def job_exists(self, url: str, user_id: str) -> Optional[str]:
        """
        Check if a job with the given URL exists in cache or database.

        Args:
            url: URL of the job posting
            user_id: ID of the user who owns the job

        Returns:
            Job ID if it exists, None otherwise
        """
        # Try in-memory cache first (fastest)
        if self.job_cache:
            job = self.job_cache.get_job_by_url(url, user_id)
            if job:
                return job.id

        # If not in memory cache, try database
        if self.db:
            try:
                return await self.db.job_exists(url, user_id)
            except Exception as e:
                logger.error(f"Error checking if job exists in database for user {user_id}: {e}")

        return None

    async def get_job(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job by ID from cache or database.

        Args:
            job_id: ID of the job to retrieve
            user_id: ID of the user who owns the job

        Returns:
            Job dictionary if found, None otherwise
        """
        # Try in-memory cache first (fastest)
        if self.job_cache:
            job = self.job_cache.get_job(job_id, user_id)
            if job:
                return job.to_dict()

        # If not in memory cache, try database
        if self.db:
            try:
                job = await self.db.get_job(job_id, user_id)
                if job:
                    # Update cache with this job
                    if self.job_cache:
                        self.job_cache.add_job(job, user_id)
                    return job.to_dict()
            except Exception as e:
                logger.error(f"Error getting job from database for user {user_id}: {e}")

        return None

    async def save_job(self, job, user_id: str) -> bool:
        """
        Save a job to both database and cache.

        Args:
            job: Job object to save
            user_id: ID of the user who owns the job

        Returns:
            bool: True if successful, False otherwise
        """
        success = True

        # Save to database
        if self.db:
            try:
                success = await self.db.save_job(job, user_id)
            except Exception as e:
                logger.error(f"Error saving job to database for user {user_id}: {e}")
                success = False

        # Save to cache
        try:
            if self.job_cache:
                self.job_cache.add_job(job, user_id)
        except Exception as e:
            logger.error(f"Error adding job to cache for user {user_id}: {e}")
            success = False

        return success

    async def update_job_status(self, job_id: str, user_id: str, status) -> bool:
        """
        Update job status in both cache and database.

        Args:
            job_id: ID of the job to update
            user_id: ID of the user who owns the job
            status: New job status

        Returns:
            bool: True if successful, False otherwise
        """
        success = True

        # Update in database
        if self.db:
            try:
                # Use the batch update method with single item
                success = await self.db.update_job_status_batch([(job_id, user_id, str(status))])
            except Exception as e:
                logger.error(f"Error updating job status in database for user {user_id}: {e}")
                success = False

        # Update in cache - get job, update status, and re-add to cache
        if self.job_cache and success:
            try:
                job = self.job_cache.get_job(job_id, user_id)
                if job:
                    job.status = status
                    self.job_cache.add_job(job, user_id)
            except Exception as e:
                logger.error(f"Error updating job status in cache for user {user_id}: {e}")
                # Don't mark as failure if cache update fails but DB succeeded

        return success

    async def get_all_jobs(self, user_id: str, status=None, limit: int = None, offset: int = 0) -> List:
        """
        Get all jobs for a user, optionally filtered by status.

        Args:
            user_id: ID of the user
            status: Optional status filter
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of Job objects
        """
        if self.db:
            try:
                return await self.db.get_all_jobs(user_id, status, limit, offset)
            except Exception as e:
                logger.error(f"Error getting all jobs for user {user_id}: {e}")
                return []
        return []

    async def get_job_stats(self, user_id: str) -> Dict[str, int]:
        """
        Get job statistics for a user.

        Args:
            user_id: ID of the user

        Returns:
            Dictionary with job statistics
        """
        if self.db:
            try:
                return await self.db.get_job_stats(user_id)
            except Exception as e:
                logger.error(f"Error getting job stats for user {user_id}: {e}")
                return {'total': 0}
        return {'total': 0}

    async def delete_job(self, job_id: str, user_id: str) -> bool:
        """Delete a job from both database and cache."""
        success = True

        # Delete from database first
        if self.db:
            try:
                success = await self.db.delete_job(job_id, user_id)
            except Exception as e:
                logger.error(f"Error deleting job from database for user {user_id}: {e}")
                success = False

        # Remove from cache
        if self.job_cache and success:
            try:
                self.job_cache.remove_job(job_id, user_id)
            except Exception as e:
                logger.error(f"Error removing job from cache for user {user_id}: {e}")
                # Don't mark as failure if cache removal fails but DB succeeded

        return success

    # Resume Management Methods - FIXED FOR CONSISTENCY
    async def save_resume(self, resume, user_id: str) -> bool:
        """
        Save a resume to both database and in-memory cache (consistent with job caching).
        This method handles both new resumes and updates with proper cache invalidation.

        Args:
            resume: Resume object to save
            user_id: ID of the user who owns the resume

        Returns:
            bool: True if successful, False otherwise
        """
        success = True

        # Save to database
        if self.db:
            try:
                success = await self.db.save_resume(resume, user_id)
            except Exception as e:
                logger.error(f"Error saving resume to database for user {user_id}: {e}")
                success = False

        # Update in-memory cache for consistency with job caching
        if success:
            try:
                # For updates, we need to invalidate first to ensure fresh data
                self._remove_resume_from_cache(resume.id, user_id)

                # Add/update resume in cache
                self._add_resume_to_cache(resume, user_id)

                logger.info(f"Updated resume {resume.id} in cache for user {user_id}")
            except Exception as e:
                logger.error(f"Error adding resume to cache for user {user_id}: {e}")
                # Don't mark as failure if cache update fails but DB succeeded

        return success

    async def get_resume(self, resume_id: str, user_id: str):
        """
        Get a resume by ID from cache or database (consistent with job retrieval).

        Args:
            resume_id: ID of the resume to retrieve
            user_id: ID of the user who owns the resume

        Returns:
            Resume object if found, None otherwise
        """
        # Try in-memory cache first (consistent with job caching)
        cached_resume = self._get_resume_from_cache(resume_id, user_id)
        if cached_resume:
            return cached_resume

        # If not in cache, try database
        if self.db:
            try:
                resume = await self.db.get_resume(resume_id, user_id)
                if resume:
                    # Update cache with this resume (consistent with job caching)
                    self._add_resume_to_cache(resume, user_id)
                    return resume
            except Exception as e:
                logger.error(f"Error getting resume from database for user {user_id}: {e}")
                return None
        return None

    async def update_resume(self, resume, user_id: str) -> bool:
        """
        Update a resume in both database and cache with proper cache invalidation.

        Args:
            resume: Updated Resume object
            user_id: ID of the user who owns the resume

        Returns:
            bool: True if successful, False otherwise
        """
        success = True

        # Update in database first
        if self.db:
            try:
                success = await self.db.save_resume(resume, user_id)  # Using save_resume as it handles upserts
            except Exception as e:
                logger.error(f"Error updating resume in database for user {user_id}: {e}")
                success = False

        # Invalidate and update cache
        if success:
            try:
                # Remove old cached version to ensure fresh data
                self._remove_resume_from_cache(resume.id, user_id)

                # Add updated resume to cache
                self._add_resume_to_cache(resume, user_id)

                logger.info(f"Updated resume {resume.id} in cache for user {user_id}")
            except Exception as e:
                logger.error(f"Error updating resume in cache for user {user_id}: {e}")
                # Don't mark as failure if cache update fails but DB succeeded

        return success

    async def delete_resume(self, resume_id: str, user_id: str) -> bool:
        """Delete a resume from both database and cache (consistent with job deletion)."""
        success = True

        # Delete from database first
        if self.db:
            try:
                success = await self.db.delete_resume(resume_id, user_id)
            except Exception as e:
                logger.error(f"Error deleting resume from database for user {user_id}: {e}")
                success = False

        # Remove from cache (consistent with job deletion)
        if success:
            try:
                self._remove_resume_from_cache(resume_id, user_id)
                logger.info(f"Removed resume {resume_id} from cache for user {user_id}")
            except Exception as e:
                logger.error(f"Error removing resume from cache for user {user_id}: {e}")
                # Don't mark as failure if cache removal fails but DB succeeded

        return success

    async def get_resumes_for_job(self, job_id: str, user_id: str):
        """Get all resumes associated with a specific job."""
        if self.db:
            try:
                resumes = await self.db.get_resumes_for_job(job_id, user_id)
                # Cache the retrieved resumes for future use
                for resume in resumes:
                    self._add_resume_to_cache(resume, user_id)
                return resumes
            except Exception as e:
                logger.error(f"Error getting resumes for job from database for user {user_id}: {e}")
                return []
        return []

    async def get_all_resumes(self, user_id: str, job_id: Optional[str] = None,
                              limit: int = None, offset: int = 0):
        """Get all resumes for a user with optional filtering."""
        if self.db:
            try:
                resumes = await self.db.get_all_resumes(user_id, job_id, limit, offset)
                # Cache the retrieved resumes for future use
                for resume in resumes:
                    self._add_resume_to_cache(resume, user_id)
                return resumes
            except Exception as e:
                logger.error(f"Error getting all resumes from database for user {user_id}: {e}")
                return []
        return []

    async def update_job_resume_id(self, job_id: str, user_id: str, resume_id: Optional[str]) -> bool:
        """Update job's resume_id field in both cache and database (can be None to clear)."""
        success = True

        # Update in database
        if self.db:
            try:
                success = await self.db.update_job_resume_id(job_id, user_id, resume_id)
            except Exception as e:
                logger.error(f"Error updating job resume_id in database for user {user_id}: {e}")
                success = False

        # Update in cache - get job, update resume_id, and re-add to cache
        if self.job_cache and success:
            try:
                job = self.job_cache.get_job(job_id, user_id)
                if job:
                    job.resume_id = resume_id
                    self.job_cache.add_job(job, user_id)
                    logger.info(f"Updated job {job_id} with resume_id {resume_id} in cache")
            except Exception as e:
                logger.error(f"Error updating job resume_id in cache for user {user_id}: {e}")
                # Don't mark as failure if cache update fails but DB succeeded

        return success

    # Resume Generation Status Management (now integrated)
    async def get_resume_status(self, resume_id: str, user_id: str) -> Optional[Dict]:
        """Get resume generation status from cache."""
        try:
            return await self.resume_cache.get_status(resume_id, user_id)
        except Exception as e:
            logger.error(f"Error getting resume status for user {user_id}: {e}")
            return None

    async def set_resume_status(self, resume_id: str, user_id: str, status, data=None, error=None):
        """Set resume generation status in cache."""
        try:
            await self.resume_cache.set_status(resume_id, user_id, status, data, error)
        except Exception as e:
            logger.error(f"Error setting resume status for user {user_id}: {e}")

    async def remove_resume_status(self, resume_id: str, user_id: str):
        """Remove resume generation status from cache."""
        try:
            await self.resume_cache.remove(resume_id, user_id)
        except Exception as e:
            logger.error(f"Error removing resume status for user {user_id}: {e}")

    # Cache Management Methods
    async def clear_user_cache(self, user_id: str):
        """Clear all cache data for a user."""
        try:
            # Clear job cache
            if self.job_cache:
                self.job_cache.clear_user(user_id)

            # Clear search cache
            if self.search_cache:
                self.search_cache.clear_user(user_id)

            # Clear resume cache
            await self.resume_cache.clear_user_cache(user_id)

            # Clear in-memory resume cache
            self._clear_user_resume_cache(user_id)

            logger.info(f"Cleared all cache data for user {user_id}")
        except Exception as e:
            logger.error(f"Error clearing cache for user {user_id}: {e}")

    async def cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        try:
            await self.resume_cache.cleanup_expired()
            self._cleanup_expired_resume_cache()
            logger.info("Cleaned up expired cache entries")
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "timestamp": "now",
            "resume_cache": self.resume_cache.get_stats() if self.resume_cache else {},
            "resume_memory_cache": self._get_resume_cache_stats(),
        }

        if self.job_cache:
            stats["job_cache"] = self.job_cache.get_stats()

        return stats

    # Database Health Check
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on database and caches."""
        health_info = {
            "database": "unavailable",
            "job_cache": "available" if self.job_cache else "unavailable",
            "resume_cache": "available" if self.resume_cache else "unavailable",
            "resume_memory_cache": "available"
        }

        if self.db:
            try:
                db_health = await self.db.health_check()
                health_info["database"] = db_health
            except Exception as e:
                health_info["database"] = {"status": "error", "error": str(e)}

        return health_info

    # Helper Methods
    def _generate_search_key(self, keywords: str, location: str, filters: Dict[str, Any], user_id: str) -> str:
        """
        Generate a unique identifier for a search based on its parameters.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            user_id: ID of the user who performed the search

        Returns:
            String hash of the search parameters
        """
        # Normalize the strings
        keywords_norm = ' '.join(keywords.lower().split())
        location_norm = ' '.join(location.lower().split())

        # Create a stable string representation of filters
        filters_str = json.dumps(filters, sort_keys=True)

        # Create the key string, including user_id
        key_str = f"{user_id}|{keywords_norm}|{location_norm}|{filters_str}"

        # Hash for consistent ID
        return hashlib.md5(key_str.encode()).hexdigest()

    # NEW: In-memory Resume Cache Methods for Consistency
    def _add_resume_to_cache(self, resume, user_id: str):
        """Add resume to in-memory cache with proper versioning."""
        import time
        with self._resume_cache_lock:
            cache_key = f"{user_id}:{resume.id}"
            self._resume_memory_cache[cache_key] = {
                "resume": resume,
                "cached_at": time.time(),
                "user_id": user_id,
                "version": getattr(resume, 'updated_at', time.time())  # Track version for consistency
            }

    def _get_resume_from_cache(self, resume_id: str, user_id: str):
        """Get resume from in-memory cache with staleness check."""
        import time
        with self._resume_cache_lock:
            cache_key = f"{user_id}:{resume_id}"
            entry = self._resume_memory_cache.get(cache_key)
            if entry:
                # Simple TTL check (1 hour)
                if time.time() - entry["cached_at"] < 3600:
                    return entry["resume"]
                else:
                    # Remove expired entry
                    del self._resume_memory_cache[cache_key]
                    logger.debug(f"Removed expired resume {resume_id} from cache for user {user_id}")
            return None

    def _remove_resume_from_cache(self, resume_id: str, user_id: str):
        """Remove resume from in-memory cache."""
        with self._resume_cache_lock:
            cache_key = f"{user_id}:{resume_id}"
            if cache_key in self._resume_memory_cache:
                del self._resume_memory_cache[cache_key]
                logger.debug(f"Invalidated cache for resume {resume_id} for user {user_id}")

    def _invalidate_related_caches(self, resume_id: str, user_id: str):
        """
        Invalidate caches that might be affected by resume changes.
        This includes job caches that reference this resume.
        """
        try:
            # If this resume is associated with jobs, we might need to invalidate job cache entries
            # that reference this resume_id
            if self.job_cache:
                # Get user jobs and check for resume references
                user_jobs = self.job_cache.get_user_jobs(user_id)
                for job in user_jobs:
                    if hasattr(job, 'resume_id') and job.resume_id == resume_id:
                        # Re-add job to cache to ensure consistency
                        self.job_cache.add_job(job, user_id)

            logger.debug(f"Invalidated related caches for resume {resume_id} for user {user_id}")
        except Exception as e:
            logger.warning(f"Error invalidating related caches for resume {resume_id}: {e}")

    def _clear_user_resume_cache(self, user_id: str):
        """Clear all resume cache entries for a user."""
        with self._resume_cache_lock:
            keys_to_remove = [k for k in self._resume_memory_cache.keys() if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self._resume_memory_cache[key]
            if keys_to_remove:
                logger.info(f"Cleared {len(keys_to_remove)} resume cache entries for user {user_id}")

    def _cleanup_expired_resume_cache(self):
        """Clean up expired resume cache entries."""
        import time
        with self._resume_cache_lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._resume_memory_cache.items()
                if now - v["cached_at"] > 3600  # 1 hour TTL
            ]
            for key in expired_keys:
                del self._resume_memory_cache[key]
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired resume cache entries")

    def _get_resume_cache_stats(self) -> Dict[str, Any]:
        """Get resume cache statistics."""
        import time
        with self._resume_cache_lock:
            now = time.time()
            active_entries = sum(1 for v in self._resume_memory_cache.values()
                                 if now - v["cached_at"] < 3600)

            return {
                "total_entries": len(self._resume_memory_cache),
                "active_entries": active_entries,
                "expired_entries": len(self._resume_memory_cache) - active_entries,
                "memory_estimate_kb": len(str(self._resume_memory_cache)) // 1024
            }