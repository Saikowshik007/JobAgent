from typing import Dict, List, Optional, Any
import hashlib
import json
import logging
import threading

# Configure logging
logger = logging.getLogger(__name__)

class DBCacheManager:
    """
    Integration class that connects the database with the in-memory cache system.
    This manager handles interactions between the database and the cache system,
    ensuring data consistency and optimizing performance in a thread-safe manner.
    """

    # Thread local storage to ensure thread safety
    _thread_local = threading.local()

    def __init__(self, database=None, job_cache=None, search_cache=None):
        """
        Initialize the cache manager with database and cache instances.

        Args:
            database: Database instance
            job_cache: JobCache instance
            search_cache: SearchCache instance
        """
        self.db = database
        self.job_cache = job_cache
        self.search_cache = search_cache

    def get_cached_search_results(self, keywords: str, location: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
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

        Returns:
            List of job details dictionaries
        """
        # Skip if no caching is available
        if not self.search_cache and not self.db:
            return []

        # Generate a search key for consistent identification
        search_key = self._generate_search_key(keywords, location, filters)

        # Try in-memory cache first (fastest)
        if self.search_cache:
            cache_results = self.search_cache.get_search_results(keywords, location, filters)
            if cache_results:
                # Get full job details from job cache
                results = []
                for job_id in cache_results:
                    if self.job_cache:
                        job = self.job_cache.get_job(job_id)
                        if job:
                            results.append(job.to_dict())

                # If we have all the jobs in the cache, return them
                if len(results) == len(cache_results):
                    return results

        # If not in memory cache or incomplete, try database
        if self.db:
            try:
                db_results = self.db.get_cached_search_results(keywords, location, filters)
                if db_results:
                    # Update in-memory caches with these results
                    if self.search_cache:
                        job_ids = [job["id"] for job in db_results if job.get("id")]
                        self.search_cache.add_search_results(keywords, location, filters, job_ids)

                    # Add individual jobs to job cache
                    if self.job_cache:
                        for job_dict in db_results:
                            # Convert dict to Job object
                            from data_models import Job
                            job = Job.from_dict(job_dict)
                            self.job_cache.add_job(job)

                    return db_results
            except Exception as e:
                logger.error(f"Error retrieving cached search results: {e}")

        # No results found in either cache
        return []

    def save_search_results(self, keywords: str, location: str, filters: Dict[str, Any],
                            job_listings: List[Dict[str, Any]]) -> bool:
        """
        Save search results to both database and in-memory cache.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            job_listings: Job listing dictionaries

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
                search_id = self._generate_search_key(keywords, location, filters)
                # Save individual jobs first to ensure they exist
                for job_dict in job_listings:
                    # Convert dict to Job object
                    from data_models import Job
                    try:
                        job = Job.from_dict(job_dict)
                        self.db.save_job(job)
                    except Exception as e:
                        logger.error(f"Error saving job to database: {e}")
                        success = False

                # Then save the search history
                success = success and self.db.save_search_history(keywords, location, filters, job_ids, search_id)
            except Exception as e:
                logger.error(f"Error saving search history to database: {e}")
                success = False

        # Save to in-memory cache
        try:
            if self.search_cache:
                self.search_cache.add_search_results(keywords, location, filters, job_ids)

            # Add individual jobs to job cache
            if self.job_cache:
                for job_dict in job_listings:
                    # Convert dict to Job object
                    from data_models import Job
                    try:
                        job = Job.from_dict(job_dict)
                        self.job_cache.add_job(job)
                    except Exception as e:
                        logger.error(f"Error adding job to cache: {e}")
                        success = False
        except Exception as e:
            logger.error(f"Error updating in-memory caches: {e}")
            success = False

        return success

    def job_exists(self, url: str) -> Optional[str]:
        """
        Check if a job with the given URL exists in cache or database.

        Args:
            url: URL of the job posting

        Returns:
            Job ID if it exists, None otherwise
        """
        # Try in-memory cache first (fastest)
        if self.job_cache:
            job = self.job_cache.get_job_by_url(url)
            if job:
                return job.id

        # If not in memory cache, try database
        if self.db:
            try:
                return self.db.job_exists(url)
            except Exception as e:
                logger.error(f"Error checking if job exists in database: {e}")

        return None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job by ID from cache or database.

        Args:
            job_id: ID of the job to retrieve

        Returns:
            Job dictionary if found, None otherwise
        """
        # Try in-memory cache first (fastest)
        if self.job_cache:
            job = self.job_cache.get_job(job_id)
            if job:
                return job.to_dict()

        # If not in memory cache, try database
        if self.db:
            try:
                job = self.db.get_job(job_id)
                if job:
                    # Update cache with this job
                    if self.job_cache:
                        self.job_cache.add_job(job)
                    return job.to_dict()
            except Exception as e:
                logger.error(f"Error getting job from database: {e}")

        return None

    def save_job(self, job) -> bool:
        """
        Save a job to both database and cache.

        Args:
            job: Job object to save

        Returns:
            bool: True if successful, False otherwise
        """
        success = True

        # Save to database
        if self.db:
            try:
                success = self.db.save_job(job)
            except Exception as e:
                logger.error(f"Error saving job to database: {e}")
                success = False

        # Save to cache
        try:
            if self.job_cache:
                self.job_cache.add_job(job)
        except Exception as e:
            logger.error(f"Error adding job to cache: {e}")
            success = False

        return success

    def _generate_search_key(self, keywords: str, location: str, filters: Dict[str, Any]) -> str:
        """
        Generate a unique identifier for a search based on its parameters.
        
        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            
        Returns:
            String hash of the search parameters
        """
        # Normalize the strings
        keywords_norm = ' '.join(keywords.lower().split())
        location_norm = ' '.join(location.lower().split())

        # Create a stable string representation of filters
        filters_str = json.dumps(filters, sort_keys=True)

        # Create the key string
        key_str = f"{keywords_norm}|{location_norm}|{filters_str}"

        # Hash for consistent ID
        return hashlib.md5(key_str.encode()).hexdigest()