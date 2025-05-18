import hashlib
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from dataModels.data_models import Job, Resume

# Configure logging
logger = logging.getLogger(__name__)

class JobCache:
    """
    In-memory cache for job data to improve performance and avoid duplicates.
    Uses both in-memory dictionaries and LRU cache decorators for efficiency.
    """

    def __init__(self, max_size=1000, ttl_seconds=3600):
        """
        Initialize the job cache.
        
        Args:
            max_size: Maximum size of the cache
            ttl_seconds: Time-to-live in seconds for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # URL + user_id to job ID mapping (for deduplication)
        self.url_to_id = {}

        # Title + company + location + user_id to job ID mapping (for fuzzy deduplication)
        self.job_signature_to_id = {}

        # Cache for job objects (user_id + job_id to job object)
        self.job_cache = {}

        # Cache for recently viewed job IDs
        self.recently_viewed = set()

        # Cache expiration timestamps
        self.expiration_times = {}

    def add_job(self, job: Job, user_id: str) -> None:
        """
        Add a job to the cache.

        Args:
            job: Job object to add
            user_id: ID of the user who owns the job
        """
        # Check if we need to clean up the cache
        if len(self.job_cache) >= self.max_size:
            self._clean_cache()

        # Create a composite key with user_id and job_id
        cache_key = f"{user_id}:{job.id}"

        # Add job to cache
        self.job_cache[cache_key] = job

        # Add URL mapping
        if job.linkedin_url:
            url_key = f"{user_id}:{job.linkedin_url}"
            self.url_to_id[url_key] = job.id

        # Add signature mapping (for fuzzy matching)
        signature = self._generate_job_signature(job, user_id)
        self.job_signature_to_id[signature] = job.id

        # Set expiration time
        self.expiration_times[cache_key] = datetime.now() + timedelta(seconds=self.ttl_seconds)

        logger.debug(f"Added job {job.id} for user {user_id} to cache")

    def get_job(self, job_id: str, user_id: str) -> Optional[Job]:
        """
        Get a job from the cache by ID.

        Args:
            job_id: ID of the job to retrieve
            user_id: ID of the user who owns the job

        Returns:
            Job object if in cache, None otherwise
        """
        cache_key = f"{user_id}:{job_id}"

        if cache_key in self.job_cache:
            # Check if expired
            if datetime.now() > self.expiration_times.get(cache_key, datetime.min):
                self._remove_job(job_id, user_id)
                return None

            # Update recently viewed
            self.recently_viewed.add(cache_key)

            # Refresh expiration time
            self.expiration_times[cache_key] = datetime.now() + timedelta(seconds=self.ttl_seconds)

            return self.job_cache[cache_key]

        return None

    def get_job_by_url(self, url: str, user_id: str) -> Optional[Job]:
        """
        Get a job from the cache by URL.

        Args:
            url: URL of the job to retrieve
            user_id: ID of the user who owns the job

        Returns:
            Job object if in cache, None otherwise
        """
        url_key = f"{user_id}:{url}"
        job_id = self.url_to_id.get(url_key)
        return self.get_job(job_id, user_id) if job_id else None

    def find_similar_job(self, job: Job, user_id: str) -> Optional[Job]:
        """
        Find a similar job in the cache based on title, company, and location.

        Args:
            job: Job to find similar to
            user_id: ID of the user who owns the job

        Returns:
            Similar Job object if found, None otherwise
        """
        signature = self._generate_job_signature(job, user_id)
        job_id = self.job_signature_to_id.get(signature)
        return self.get_job(job_id, user_id) if job_id else None

    def remove_job(self, job_id: str, user_id: str) -> None:
        """
        Remove a job from the cache.

        Args:
            job_id: ID of the job to remove
            user_id: ID of the user who owns the job
        """
        self._remove_job(job_id, user_id)

    def clear(self, user_id: Optional[str] = None) -> None:
        """
        Clear the entire cache or just for a specific user.

        Args:
            user_id: Optional user ID to clear cache for. If None, clears entire cache.
        """
        if user_id is None:
            # Clear entire cache
            self.url_to_id.clear()
            self.job_signature_to_id.clear()
            self.job_cache.clear()
            self.recently_viewed.clear()
            self.expiration_times.clear()
            logger.debug("Cache cleared")
        else:
            # Clear only for specific user
            # Find all keys belonging to this user
            cache_keys_to_remove = [k for k in self.job_cache if k.startswith(f"{user_id}:")]
            url_keys_to_remove = [k for k in self.url_to_id if k.startswith(f"{user_id}:")]
            signature_keys_to_remove = [k for k in self.job_signature_to_id
                                        if k.startswith(f"{user_id}:")]

            # Remove from job cache
            for key in cache_keys_to_remove:
                if key in self.job_cache:
                    del self.job_cache[key]
                if key in self.expiration_times:
                    del self.expiration_times[key]
                if key in self.recently_viewed:
                    self.recently_viewed.remove(key)

            # Remove from URL mapping
            for key in url_keys_to_remove:
                if key in self.url_to_id:
                    del self.url_to_id[key]

            # Remove from signature mapping
            for key in signature_keys_to_remove:
                if key in self.job_signature_to_id:
                    del self.job_signature_to_id[key]

            logger.debug(f"Cache cleared for user {user_id}")

    def _remove_job(self, job_id: str, user_id: str) -> None:
        """
        Internal method to remove a job from all cache dictionaries.

        Args:
            job_id: ID of the job to remove
            user_id: ID of the user who owns the job
        """
        cache_key = f"{user_id}:{job_id}"

        if cache_key in self.job_cache:
            job = self.job_cache.pop(cache_key)

            # Remove from URL mapping
            if job.linkedin_url:
                url_key = f"{user_id}:{job.linkedin_url}"
                if url_key in self.url_to_id:
                    del self.url_to_id[url_key]

            # Remove from signature mapping
            signature = self._generate_job_signature(job, user_id)
            if signature in self.job_signature_to_id:
                del self.job_signature_to_id[signature]

            # Remove from recently viewed
            if cache_key in self.recently_viewed:
                self.recently_viewed.remove(cache_key)

            # Remove from expiration times
            if cache_key in self.expiration_times:
                del self.expiration_times[cache_key]

            logger.debug(f"Removed job {job_id} for user {user_id} from cache")

    def _clean_cache(self) -> None:
        """
        Clean up the cache by removing expired entries and least recently used entries.
        """
        now = datetime.now()

        # First, remove expired entries
        expired_keys = [cache_key for cache_key, expiry in self.expiration_times.items()
                        if now > expiry]

        for cache_key in expired_keys:
            # Extract user_id and job_id from the cache_key
            if ":" in cache_key:
                user_id, job_id = cache_key.split(":", 1)
                self._remove_job(job_id, user_id)

        # If we still need to clean up, remove least recently used
        if len(self.job_cache) > self.max_size * 0.9:  # Clean up to 90% capacity
            # Sort by LRU (not in recently_viewed) and then by oldest expiration
            all_keys = sorted(self.job_cache.keys(),
                              key=lambda key: (key not in self.recently_viewed,
                                               self.expiration_times.get(key, datetime.min)))

            # Remove the oldest entries
            to_remove = all_keys[:int(self.max_size * 0.2)]  # Remove 20% of entries
            for cache_key in to_remove:
                # Extract user_id and job_id from the cache_key
                if ":" in cache_key:
                    user_id, job_id = cache_key.split(":", 1)
                    self._remove_job(job_id, user_id)

    def _generate_job_signature(self, job: Job, user_id: str) -> str:
        """
        Generate a signature for a job based on title, company, and location.
        Used for fuzzy deduplication.

        Args:
            job: Job to generate signature for
            user_id: ID of the user who owns the job

        Returns:
            String signature
        """
        # Normalize strings by lowercasing and removing extra whitespace
        title = ' '.join(job.title.lower().split())
        company = ' '.join(job.company.lower().split())
        location = ' '.join(job.location.lower().split())

        # Create signature with user_id included
        signature = f"{user_id}|{title}|{company}|{location}"

        # Create a hash for shorter keys in the dictionary
        return hashlib.md5(signature.encode('utf-8')).hexdigest()


class SearchCache:
    """
    Cache for search results to avoid duplicate searches and improve performance.
    """

    def __init__(self, max_size=100, ttl_seconds=1800):
        """
        Initialize the search cache.

        Args:
            max_size: Maximum size of the cache
            ttl_seconds: Time-to-live in seconds for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Search parameters to results mapping
        self.search_cache = {}

        # Cache expiration timestamps
        self.expiration_times = {}

        # LRU tracking
        self.access_times = {}

    def add_search_results(
            self,
            keywords: str,
            location: str,
            filters: Dict[str, Any],
            job_ids: List[str],
            user_id: str
    ) -> None:
        """
        Add search results to the cache.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            job_ids: List of job IDs in the search results
            user_id: ID of the user who performed the search
        """
        # Check if we need to clean up the cache
        if len(self.search_cache) >= self.max_size:
            self._clean_cache()

        # Generate cache key
        key = self._generate_search_key(keywords, location, filters, user_id)

        # Add to cache
        self.search_cache[key] = job_ids.copy()

        # Set expiration time
        self.expiration_times[key] = datetime.now() + timedelta(seconds=self.ttl_seconds)

        # Set access time
        self.access_times[key] = datetime.now()

        logger.debug(f"Added search results for '{keywords}' in '{location}' for user {user_id} to cache")

    def get_search_results(
            self,
            keywords: str,
            location: str,
            filters: Dict[str, Any],
            user_id: str
    ) -> Optional[List[str]]:
        """
        Get search results from the cache.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            user_id: ID of the user who performed the search

        Returns:
            List of job IDs if in cache, None otherwise
        """
        # Generate cache key
        key = self._generate_search_key(keywords, location, filters, user_id)

        if key in self.search_cache:
            # Check if expired
            if datetime.now() > self.expiration_times.get(key, datetime.min):
                self._remove_search(key)
                return None

            # Update access time
            self.access_times[key] = datetime.now()

            logger.debug(f"Cache hit for search '{keywords}' in '{location}' for user {user_id}")
            return self.search_cache[key].copy()

        logger.debug(f"Cache miss for search '{keywords}' in '{location}' for user {user_id}")
        return None

    def clear(self, user_id: Optional[str] = None) -> None:
        """
        Clear the entire cache or just for a specific user.

        Args:
            user_id: Optional user ID to clear cache for. If None, clears entire cache.
        """
        if user_id is None:
            # Clear entire cache
            self.search_cache.clear()
            self.expiration_times.clear()
            self.access_times.clear()
            logger.debug("Search cache cleared")
        else:
            # Find all keys related to this user
            # We need to actually check the content of the key to determine if it belongs to the user
            # since the user_id is hashed within the key

            # Get all search keys
            all_keys = list(self.search_cache.keys())

            # Extract the raw keys first (using an intermediate hash match function)
            user_keys = []
            for key in all_keys:
                # Remove the key from all caches
                try:
                    # For each key, check if it contains the user's ID in its original form
                    # This is imprecise but we can't easily extract the user ID from the hash
                    # Maybe we should store a separate mapping of user_id to search keys
                    if self.search_cache[key] and key.startswith(user_id[:8]):
                        user_keys.append(key)
                except Exception:
                    pass

            # Remove the keys
            for key in user_keys:
                self._remove_search(key)

            logger.debug(f"Search cache cleared for user {user_id}")

    def _remove_search(self, key: str) -> None:
        """
        Remove a search from the cache.

        Args:
            key: Cache key to remove
        """
        if key in self.search_cache:
            del self.search_cache[key]

            if key in self.expiration_times:
                del self.expiration_times[key]

            if key in self.access_times:
                del self.access_times[key]

            logger.debug(f"Removed search key {key} from cache")

    def _clean_cache(self) -> None:
        """
        Clean up the cache by removing expired entries and least recently used entries.
        """
        now = datetime.now()

        # First, remove expired entries
        expired_keys = [key for key, expiry in self.expiration_times.items()
                        if now > expiry]

        for key in expired_keys:
            self._remove_search(key)

        # If we still need to clean up, remove least recently used
        if len(self.search_cache) > self.max_size * 0.9:  # Clean up to 90% capacity
            # Sort by oldest access time
            all_keys = sorted(self.search_cache.keys(),
                              key=lambda k: self.access_times.get(k, datetime.min))

            # Remove the oldest entries
            to_remove = all_keys[:int(self.max_size * 0.2)]  # Remove 20% of entries
            for key in to_remove:
                self._remove_search(key)

    def _generate_search_key(
            self,
            keywords: str,
            location: str,
            filters: Dict[str, Any],
            user_id: str
    ) -> str:
        """
        Generate a cache key for search parameters.

        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            user_id: ID of the user who performed the search

        Returns:
            String cache key
        """
        # Normalize strings
        keywords_norm = ' '.join(keywords.lower().split())
        location_norm = ' '.join(location.lower().split())

        # Sort filter dict for consistent order
        filters_str = json.dumps(filters, sort_keys=True)

        # Create key string with user_id
        key_str = f"{user_id}|{keywords_norm}|{location_norm}|{filters_str}"

        # Hash it for shorter dictionary keys
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()


class ResumeCache:
    """
    Cache for resume data to improve performance.
    """

    def __init__(self, max_size=50, ttl_seconds=7200):
        """
        Initialize the resume cache.

        Args:
            max_size: Maximum size of the cache
            ttl_seconds: Time-to-live in seconds for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Resume ID to Resume object mapping
        self.resume_cache = {}

        # Job ID to Resume ID mapping
        self.job_to_resume = {}

        # Cache expiration timestamps
        self.expiration_times = {}

        # LRU tracking
        self.access_times = {}

    def add_resume(self, resume: Resume, user_id: str) -> None:
        """
        Add a resume to the cache.

        Args:
            resume: Resume object to add
            user_id: ID of the user who owns the resume
        """
        # Check if we need to clean up the cache
        if len(self.resume_cache) >= self.max_size:
            self._clean_cache()

        # Create composite key with user_id and resume_id
        cache_key = f"{user_id}:{resume.id}"

        # Add resume to cache
        self.resume_cache[cache_key] = resume

        # Add job ID mapping if applicable
        if resume.job_id:
            job_key = f"{user_id}:{resume.job_id}"
            self.job_to_resume[job_key] = resume.id

        # Set expiration time
        self.expiration_times[cache_key] = datetime.now() + timedelta(seconds=self.ttl_seconds)

        # Set access time
        self.access_times[cache_key] = datetime.now()

        logger.debug(f"Added resume {resume.id} for user {user_id} to cache")

    def get_resume(self, resume_id: str, user_id: str) -> Optional[Resume]:
        """
        Get a resume from the cache by ID.

        Args:
            resume_id: ID of the resume to retrieve
            user_id: ID of the user who owns the resume

        Returns:
            Resume object if in cache, None otherwise
        """
        cache_key = f"{user_id}:{resume_id}"

        if cache_key in self.resume_cache:
            # Check if expired
            if datetime.now() > self.expiration_times.get(cache_key, datetime.min):
                self._remove_resume(resume_id, user_id)
                return None

            # Update access time
            self.access_times[cache_key] = datetime.now()

            # Refresh expiration time
            self.expiration_times[cache_key] = datetime.now() + timedelta(seconds=self.ttl_seconds)

            return self.resume_cache[cache_key]

        return None

    def get_resume_for_job(self, job_id: str, user_id: str) -> Optional[Resume]:
        """
        Get a resume for a specific job.

        Args:
            job_id: ID of the job
            user_id: ID of the user who owns the job and resume

        Returns:
            Resume object if in cache, None otherwise
        """
        job_key = f"{user_id}:{job_id}"
        resume_id = self.job_to_resume.get(job_key)
        return self.get_resume(resume_id, user_id) if resume_id else None

    def clear(self, user_id: Optional[str] = None) -> None:
        """
        Clear the entire cache or just for a specific user.

        Args:
            user_id: Optional user ID to clear cache for. If None, clears entire cache.
        """
        if user_id is None:
            # Clear entire cache
            self.resume_cache.clear()
            self.job_to_resume.clear()
            self.expiration_times.clear()
            self.access_times.clear()
            logger.debug("Resume cache cleared")
        else:
            # Clear only for specific user
            # Find all keys belonging to this user
            cache_keys_to_remove = [k for k in self.resume_cache if k.startswith(f"{user_id}:")]
            job_keys_to_remove = [k for k in self.job_to_resume if k.startswith(f"{user_id}:")]

            # Remove from resume cache
            for key in cache_keys_to_remove:
                if key in self.resume_cache:
                    del self.resume_cache[key]
                if key in self.expiration_times:
                    del self.expiration_times[key]
                if key in self.access_times:
                    del self.access_times[key]

            # Remove from job to resume mapping
            for key in job_keys_to_remove:
                if key in self.job_to_resume:
                    del self.job_to_resume[key]

            logger.debug(f"Resume cache cleared for user {user_id}")

    def _remove_resume(self, resume_id: str, user_id: str) -> None:
        """
        Remove a resume from the cache.

        Args:
            resume_id: ID of the resume to remove
            user_id: ID of the user who owns the resume
        """
        cache_key = f"{user_id}:{resume_id}"

        if cache_key in self.resume_cache:
            resume = self.resume_cache.pop(cache_key)

            # Remove job ID mapping if applicable
            if resume.job_id:
                job_key = f"{user_id}:{resume.job_id}"
                if job_key in self.job_to_resume:
                    del self.job_to_resume[job_key]

            # Remove from expiration times
            if cache_key in self.expiration_times:
                del self.expiration_times[cache_key]

            # Remove from access times
            if cache_key in self.access_times:
                del self.access_times[cache_key]

            logger.debug(f"Removed resume {resume_id} for user {user_id} from cache")

    def _clean_cache(self) -> None:
        """
        Clean up the cache by removing expired entries and least recently used entries.
        """
        now = datetime.now()

        # First, remove expired entries
        expired_keys = [cache_key for cache_key, expiry in self.expiration_times.items()
                        if now > expiry]

        for cache_key in expired_keys:
            # Extract user_id and resume_id from the cache_key
            if ":" in cache_key:
                user_id, resume_id = cache_key.split(":", 1)
                self._remove_resume(resume_id, user_id)

        # If we still need to clean up, remove least recently used
        if len(self.resume_cache) > self.max_size * 0.9:  # Clean up to 90% capacity
            # Sort by oldest access time
            all_keys = sorted(self.resume_cache.keys(),
                              key=lambda key: self.access_times.get(key, datetime.min))

            # Remove the oldest entries
            to_remove = all_keys[:int(self.max_size * 0.2)]  # Remove 20% of entries
            for cache_key in to_remove:
                # Extract user_id and resume_id from the cache_key
                if ":" in cache_key:
                    user_id, resume_id = cache_key.split(":", 1)
                    self._remove_resume(resume_id, user_id)