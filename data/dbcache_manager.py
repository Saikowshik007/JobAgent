from typing import Dict, List, Optional, Any
import hashlib
import json
import logging
import threading

# Configure logging
logger = logging.getLogger(__name__)

class DBCacheManager:
    """
    Unified cache manager that handles all caching operations using Redis.
    This manager provides a single interface for all database and cache interactions.
    """

    def __init__(self, database=None, redis_cache=None, redis_url="redis://0.0.0.0:6379"):
        """
        Initialize the unified cache manager with database and Redis cache.

        Args:
            database: Database instance
            redis_cache: RedisCache instance (will be created if not provided)
            redis_url: Redis connection URL
        """
        self.db = database

        # Initialize Redis cache if not provided
        if redis_cache is None:
            from data.cache import RedisCache
            self.cache = RedisCache(redis_url=redis_url)
        else:
            self.cache = redis_cache

    # Job Management Methods
    async def get_cached_search_results(self, keywords: str, location: str, filters: Dict[str, Any], user_id: str) -> List[Dict[str, Any]]:
        """
        Get search results from cache or database.
        Implementation of the method expected by JobSearcher.

        This method follows this process:
        1. Try to get results from Redis cache
        2. If not found, try to get from database
        3. If found in database, update Redis cache

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            user_id: ID of the user who performed the search

        Returns:
            List of job details dictionaries
        """
        # Skip if no caching is available
        if not self.cache and not self.db:
            return []

        # Try Redis cache first (fastest)
        if self.cache:
            cached_job_ids = self.cache.get_search_results(keywords, location, filters, user_id)
            if cached_job_ids:
                # Get full job details from job cache
                results = []
                for job_id in cached_job_ids:
                    job = self.cache.get_job(job_id, user_id)
                    if job:
                        results.append(job.to_dict())

                # If we have all the jobs in the cache, return them
                if len(results) == len(cached_job_ids):
                    return results

        # If not in cache or incomplete, try database
        if self.db:
            try:
                db_results = await self.db.get_cached_search_results(keywords, location, filters, user_id)
                if db_results:
                    # Update Redis cache with these results
                    if self.cache:
                        job_ids = [job["id"] for job in db_results if job.get("id")]
                        self.cache.add_search_results(keywords, location, filters, job_ids, user_id)

                        # Add individual jobs to cache
                        for job_dict in db_results:
                            from dataModels.data_models import Job
                            job = Job.from_dict(job_dict)
                            self.cache.add_job(job, user_id)

                    return db_results
            except Exception as e:
                logger.error(f"Error retrieving cached search results for user {user_id}: {e}")

        # No results found in either cache
        return []

    async def save_search_results(self, keywords: str, location: str, filters: Dict[str, Any],
                                  job_listings: List[Dict[str, Any]], user_id: str) -> bool:
        """
        Save search results to both database and Redis cache.

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
        if not self.cache and not self.db:
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

        # Save to Redis cache
        try:
            if self.cache:
                self.cache.add_search_results(keywords, location, filters, job_ids, user_id)

                # Add individual jobs to cache
                for job_dict in job_listings:
                    from dataModels.data_models import Job
                    try:
                        job = Job.from_dict(job_dict)
                        self.cache.add_job(job, user_id)
                    except Exception as e:
                        logger.error(f"Error adding job to cache for user {user_id}: {e}")
                        success = False
        except Exception as e:
            logger.error(f"Error updating Redis cache for user {user_id}: {e}")
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
        # Try Redis cache first (fastest)
        if self.cache:
            job = self.cache.get_job_by_url(url, user_id)
            if job:
                return job.id

        # If not in cache, try database
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
        # Try Redis cache first (fastest)
        if self.cache:
            job = self.cache.get_job(job_id, user_id)
            if job:
                return job.to_dict()

        # If not in cache, try database
        if self.db:
            try:
                job = await self.db.get_job(job_id, user_id)
                if job:
                    # Update cache with this job
                    if self.cache:
                        self.cache.add_job(job, user_id)
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

        # Save to Redis cache
        try:
            if self.cache:
                self.cache.add_job(job, user_id)
        except Exception as e:
            logger.error(f"Error adding job to Redis cache for user {user_id}: {e}")
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

        # Update in Redis cache - get job, update status, and re-add to cache
        if self.cache and success:
            try:
                job = self.cache.get_job(job_id, user_id)
                if job:
                    job.status = status
                    self.cache.add_job(job, user_id)
            except Exception as e:
                logger.error(f"Error updating job status in Redis cache for user {user_id}: {e}")
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

        # Remove from Redis cache
        if self.cache and success:
            try:
                self.cache.remove_job(job_id, user_id)
            except Exception as e:
                logger.error(f"Error removing job from Redis cache for user {user_id}: {e}")
                # Don't mark as failure if cache removal fails but DB succeeded

        return success

    # Resume Management Methods
    async def save_resume(self, resume, user_id: str) -> bool:
        """
        Save a resume to both database and Redis cache.

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

        # Update Redis cache
        if success:
            try:
                if self.cache:
                    self.cache.add_resume(resume, user_id)
                    logger.info(f"Updated resume {resume.id} in Redis cache for user {user_id}")
            except Exception as e:
                logger.error(f"Error adding resume to Redis cache for user {user_id}: {e}")
                # Don't mark as failure if cache update fails but DB succeeded

        return success

    async def get_resume(self, resume_id: str, user_id: str):
        """
        Get a resume by ID from cache or database.

        Args:
            resume_id: ID of the resume to retrieve
            user_id: ID of the user who owns the resume

        Returns:
            Resume object if found, None otherwise
        """
        # Try Redis cache first
        if self.cache:
            cached_resume = self.cache.get_resume(resume_id, user_id)
            if cached_resume:
                return cached_resume

        # If not in cache, try database
        if self.db:
            try:
                resume = await self.db.get_resume(resume_id, user_id)
                if resume:
                    # Update cache with this resume
                    if self.cache:
                        self.cache.add_resume(resume, user_id)
                    return resume
            except Exception as e:
                logger.error(f"Error getting resume from database for user {user_id}: {e}")
                return None
        return None

    async def update_resume(self, resume, user_id: str) -> bool:
        """
        Update a resume in both database and cache.

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

        # Update Redis cache
        if success:
            try:
                if self.cache:
                    # Redis cache handles upserts automatically
                    self.cache.add_resume(resume, user_id)
                    logger.info(f"Updated resume {resume.id} in Redis cache for user {user_id}")
            except Exception as e:
                logger.error(f"Error updating resume in Redis cache for user {user_id}: {e}")
                # Don't mark as failure if cache update fails but DB succeeded

        return success

    async def delete_resume(self, resume_id: str, user_id: str) -> bool:
        """Delete a resume from both database and cache."""
        success = True

        # Delete from database first
        if self.db:
            try:
                success = await self.db.delete_resume(resume_id, user_id)
            except Exception as e:
                logger.error(f"Error deleting resume from database for user {user_id}: {e}")
                success = False

        # Remove from Redis cache
        if success:
            try:
                if self.cache:
                    self.cache.remove_resume(resume_id, user_id)
                    logger.info(f"Removed resume {resume_id} from Redis cache for user {user_id}")
            except Exception as e:
                logger.error(f"Error removing resume from Redis cache for user {user_id}: {e}")
                # Don't mark as failure if cache removal fails but DB succeeded

        return success

    async def get_resumes_for_job(self, job_id: str, user_id: str):
        """Get all resumes associated with a specific job."""
        if self.db:
            try:
                resumes = await self.db.get_resumes_for_job(job_id, user_id)
                # Cache the retrieved resumes for future use
                if self.cache:
                    for resume in resumes:
                        self.cache.add_resume(resume, user_id)
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
                if self.cache:
                    for resume in resumes:
                        self.cache.add_resume(resume, user_id)
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

        # Update in Redis cache - get job, update resume_id, and re-add to cache
        if self.cache and success:
            try:
                job = self.cache.get_job(job_id, user_id)
                if job:
                    job.resume_id = resume_id
                    self.cache.add_job(job, user_id)
                    logger.info(f"Updated job {job_id} with resume_id {resume_id} in Redis cache")
            except Exception as e:
                logger.error(f"Error updating job resume_id in Redis cache for user {user_id}: {e}")
                # Don't mark as failure if cache update fails but DB succeeded

        return success

    # Resume Generation Status Management
    async def get_resume_status(self, resume_id: str, user_id: str) -> Optional[Dict]:
        """Get resume generation status from Redis cache."""
        try:
            if self.cache:
                return await self.cache.get_resume_status(resume_id, user_id)
            return None
        except Exception as e:
            logger.error(f"Error getting resume status for user {user_id}: {e}")
            return None

    async def set_resume_status(self, resume_id: str, user_id: str, status, data=None, error=None):
        """Set resume generation status in Redis cache."""
        try:
            if self.cache:
                await self.cache.set_resume_status(resume_id, user_id, status, data, error)
        except Exception as e:
            logger.error(f"Error setting resume status for user {user_id}: {e}")

    async def remove_resume_status(self, resume_id: str, user_id: str):
        """Remove resume generation status from Redis cache."""
        try:
            if self.cache:
                await self.cache.remove_resume_status(resume_id, user_id)
        except Exception as e:
            logger.error(f"Error removing resume status for user {user_id}: {e}")

    # Cache Management Methods
    async def clear_user_cache(self, user_id: str):
        """Clear all cache data for a user."""
        try:
            if self.cache:
                self.cache.clear_user_cache(user_id)
                logger.info(f"Cleared all Redis cache data for user {user_id}")
        except Exception as e:
            logger.error(f"Error clearing Redis cache for user {user_id}: {e}")

    async def cleanup_expired_cache(self):
        """Clean up expired cache entries (Redis handles TTL automatically)."""
        try:
            if self.cache:
                await self.cache.cleanup_expired_cache()
                logger.info("Redis cache cleanup completed")
        except Exception as e:
            logger.error(f"Error cleaning up Redis cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if self.cache:
            return self.cache.get_cache_stats()
        return {"error": "No cache available"}

    # Database Health Check
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on database and Redis cache."""
        health_info = {
            "database": "unavailable",
            "redis_cache": "unavailable"
        }

        if self.db:
            try:
                db_health = await self.db.health_check()
                health_info["database"] = db_health
            except Exception as e:
                health_info["database"] = {"status": "error", "error": str(e)}

        if self.cache:
            try:
                cache_health = await self.cache.health_check()
                health_info["redis_cache"] = cache_health
            except Exception as e:
                health_info["redis_cache"] = {"status": "error", "error": str(e)}

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