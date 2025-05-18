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
        
        # URL to job ID mapping (for deduplication)
        self.url_to_id = {}
        
        # Title + company + location to job ID mapping (for fuzzy deduplication)
        self.job_signature_to_id = {}
        
        # Cache for job objects
        self.job_cache = {}
        
        # Cache for recently viewed job IDs
        self.recently_viewed = set()
        
        # Cache expiration timestamps
        self.expiration_times = {}
    
    def add_job(self, job: Job) -> None:
        """
        Add a job to the cache.
        
        Args:
            job: Job object to add
        """
        # Check if we need to clean up the cache
        if len(self.job_cache) >= self.max_size:
            self._clean_cache()
        
        # Add job to cache
        self.job_cache[job.id] = job
        
        # Add URL mapping
        if job.linkedin_url:
            self.url_to_id[job.linkedin_url] = job.id
        
        # Add signature mapping (for fuzzy matching)
        signature = self._generate_job_signature(job)
        self.job_signature_to_id[signature] = job.id
        
        # Set expiration time
        self.expiration_times[job.id] = datetime.now() + timedelta(seconds=self.ttl_seconds)
        
        logger.debug(f"Added job {job.id} to cache")
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job from the cache by ID.
        
        Args:
            job_id: ID of the job to retrieve
            
        Returns:
            Job object if in cache, None otherwise
        """
        if job_id in self.job_cache:
            # Check if expired
            if datetime.now() > self.expiration_times.get(job_id, datetime.min):
                self._remove_job(job_id)
                return None
            
            # Update recently viewed
            self.recently_viewed.add(job_id)
            
            # Refresh expiration time
            self.expiration_times[job_id] = datetime.now() + timedelta(seconds=self.ttl_seconds)
            
            return self.job_cache[job_id]
        
        return None
    
    def get_job_by_url(self, url: str) -> Optional[Job]:
        """
        Get a job from the cache by URL.
        
        Args:
            url: URL of the job to retrieve
            
        Returns:
            Job object if in cache, None otherwise
        """
        job_id = self.url_to_id.get(url)
        return self.get_job(job_id) if job_id else None
    
    def find_similar_job(self, job: Job) -> Optional[Job]:
        """
        Find a similar job in the cache based on title, company, and location.
        
        Args:
            job: Job to find similar to
            
        Returns:
            Similar Job object if found, None otherwise
        """
        signature = self._generate_job_signature(job)
        job_id = self.job_signature_to_id.get(signature)
        return self.get_job(job_id) if job_id else None
    
    def remove_job(self, job_id: str) -> None:
        """
        Remove a job from the cache.
        
        Args:
            job_id: ID of the job to remove
        """
        self._remove_job(job_id)
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.url_to_id.clear()
        self.job_signature_to_id.clear()
        self.job_cache.clear()
        self.recently_viewed.clear()
        self.expiration_times.clear()
        logger.debug("Cache cleared")
    
    def _remove_job(self, job_id: str) -> None:
        """
        Internal method to remove a job from all cache dictionaries.
        
        Args:
            job_id: ID of the job to remove
        """
        if job_id in self.job_cache:
            job = self.job_cache.pop(job_id)
            
            # Remove from URL mapping
            if job.url and job.url in self.url_to_id:
                del self.url_to_id[job.url]
            
            # Remove from signature mapping
            signature = self._generate_job_signature(job)
            if signature in self.job_signature_to_id:
                del self.job_signature_to_id[signature]
            
            # Remove from recently viewed
            if job_id in self.recently_viewed:
                self.recently_viewed.remove(job_id)
            
            # Remove from expiration times
            if job_id in self.expiration_times:
                del self.expiration_times[job_id]
            
            logger.debug(f"Removed job {job_id} from cache")
    
    def _clean_cache(self) -> None:
        """
        Clean up the cache by removing expired entries and least recently used entries.
        """
        now = datetime.now()
        
        # First, remove expired entries
        expired_ids = [job_id for job_id, expiry in self.expiration_times.items() 
                      if now > expiry]
        
        for job_id in expired_ids:
            self._remove_job(job_id)
        
        # If we still need to clean up, remove least recently used
        if len(self.job_cache) > self.max_size * 0.9:  # Clean up to 90% capacity
            # Sort by LRU (not in recently_viewed) and then by oldest expiration
            all_ids = sorted(self.job_cache.keys(),
                            key=lambda jid: (jid not in self.recently_viewed,
                                           self.expiration_times.get(jid, datetime.min)))
            
            # Remove the oldest entries
            to_remove = all_ids[:int(self.max_size * 0.2)]  # Remove 20% of entries
            for job_id in to_remove:
                self._remove_job(job_id)
    
    def _generate_job_signature(self, job: Job) -> str:
        """
        Generate a signature for a job based on title, company, and location.
        Used for fuzzy deduplication.
        
        Args:
            job: Job to generate signature for
            
        Returns:
            String signature
        """
        # Normalize strings by lowercasing and removing extra whitespace
        title = ' '.join(job.title.lower().split())
        company = ' '.join(job.company.lower().split())
        location = ' '.join(job.location.lower().split())
        
        # Create signature
        signature = f"{title}|{company}|{location}"
        
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
        job_ids: List[str]
    ) -> None:
        """
        Add search results to the cache.
        
        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            job_ids: List of job IDs in the search results
        """
        # Check if we need to clean up the cache
        if len(self.search_cache) >= self.max_size:
            self._clean_cache()
        
        # Generate cache key
        key = self._generate_search_key(keywords, location, filters)
        
        # Add to cache
        self.search_cache[key] = job_ids.copy()
        
        # Set expiration time
        self.expiration_times[key] = datetime.now() + timedelta(seconds=self.ttl_seconds)
        
        # Set access time
        self.access_times[key] = datetime.now()
        
        logger.debug(f"Added search results for '{keywords}' in '{location}' to cache")
    
    def get_search_results(
        self,
        keywords: str,
        location: str,
        filters: Dict[str, Any]
    ) -> Optional[List[str]]:
        """
        Get search results from the cache.
        
        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            
        Returns:
            List of job IDs if in cache, None otherwise
        """
        # Generate cache key
        key = self._generate_search_key(keywords, location, filters)
        
        if key in self.search_cache:
            # Check if expired
            if datetime.now() > self.expiration_times.get(key, datetime.min):
                self._remove_search(key)
                return None
            
            # Update access time
            self.access_times[key] = datetime.now()
            
            logger.debug(f"Cache hit for search '{keywords}' in '{location}'")
            return self.search_cache[key].copy()
        
        logger.debug(f"Cache miss for search '{keywords}' in '{location}'")
        return None
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.search_cache.clear()
        self.expiration_times.clear()
        self.access_times.clear()
        logger.debug("Search cache cleared")
    
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
        filters: Dict[str, Any]
    ) -> str:
        """
        Generate a cache key for search parameters.
        
        Args:
            keywords: Search keywords
            location: Search location
            filters: Search filters
            
        Returns:
            String cache key
        """
        # Normalize strings
        keywords_norm = ' '.join(keywords.lower().split())
        location_norm = ' '.join(location.lower().split())
        
        # Sort filter dict for consistent order
        filters_str = json.dumps(filters, sort_keys=True)
        
        # Create key string
        key_str = f"{keywords_norm}|{location_norm}|{filters_str}"
        
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
    
    def add_resume(self, resume: Resume) -> None:
        """
        Add a resume to the cache.
        
        Args:
            resume: Resume object to add
        """
        # Check if we need to clean up the cache
        if len(self.resume_cache) >= self.max_size:
            self._clean_cache()
        
        # Add resume to cache
        self.resume_cache[resume.id] = resume
        
        # Add job ID mapping if applicable
        if resume.job_id:
            self.job_to_resume[resume.job_id] = resume.id
        
        # Set expiration time
        self.expiration_times[resume.id] = datetime.now() + timedelta(seconds=self.ttl_seconds)
        
        # Set access time
        self.access_times[resume.id] = datetime.now()
        
        logger.debug(f"Added resume {resume.id} to cache")
    
    def get_resume(self, resume_id: str) -> Optional[Resume]:
        """
        Get a resume from the cache by ID.
        
        Args:
            resume_id: ID of the resume to retrieve
            
        Returns:
            Resume object if in cache, None otherwise
        """
        if resume_id in self.resume_cache:
            # Check if expired
            if datetime.now() > self.expiration_times.get(resume_id, datetime.min):
                self._remove_resume(resume_id)
                return None
            
            # Update access time
            self.access_times[resume_id] = datetime.now()
            
            # Refresh expiration time
            self.expiration_times[resume_id] = datetime.now() + timedelta(seconds=self.ttl_seconds)
            
            return self.resume_cache[resume_id]
        
        return None
    
    def get_resume_for_job(self, job_id: str) -> Optional[Resume]:
        """
        Get a resume for a specific job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Resume object if in cache, None otherwise
        """
        resume_id = self.job_to_resume.get(job_id)
        return self.get_resume(resume_id) if resume_id else None
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.resume_cache.clear()
        self.job_to_resume.clear()
        self.expiration_times.clear()
        self.access_times.clear()
        logger.debug("Resume cache cleared")
    
    def _remove_resume(self, resume_id: str) -> None:
        """
        Remove a resume from the cache.
        
        Args:
            resume_id: ID of the resume to remove
        """
        if resume_id in self.resume_cache:
            resume = self.resume_cache.pop(resume_id)
            
            # Remove job ID mapping if applicable
            if resume.job_id and resume.job_id in self.job_to_resume:
                del self.job_to_resume[resume.job_id]
            
            # Remove from expiration times
            if resume_id in self.expiration_times:
                del self.expiration_times[resume_id]
            
            # Remove from access times
            if resume_id in self.access_times:
                del self.access_times[resume_id]
            
            logger.debug(f"Removed resume {resume_id} from cache")
    
    def _clean_cache(self) -> None:
        """
        Clean up the cache by removing expired entries and least recently used entries.
        """
        now = datetime.now()
        
        # First, remove expired entries
        expired_ids = [rid for rid, expiry in self.expiration_times.items() 
                      if now > expiry]
        
        for rid in expired_ids:
            self._remove_resume(rid)
        
        # If we still need to clean up, remove least recently used
        if len(self.resume_cache) > self.max_size * 0.9:  # Clean up to 90% capacity
            # Sort by oldest access time
            all_ids = sorted(self.resume_cache.keys(),
                            key=lambda rid: self.access_times.get(rid, datetime.min))
            
            # Remove the oldest entries
            to_remove = all_ids[:int(self.max_size * 0.2)]  # Remove 20% of entries
            for rid in to_remove:
                self._remove_resume(rid)
