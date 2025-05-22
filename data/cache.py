import asyncio
import hashlib
import logging
import json
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import threading
from dataclasses import dataclass

from dataModels.data_models import Job, Resume

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """ cache entry with metadata."""
    data: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int

class JobCache:
    """
    High-performance job cache with memory management and efficient lookup.
    """

    def __init__(self, max_size=1000, ttl_seconds=3600, max_memory_mb=100):
        """
        Initialize job cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

        # Thread-safe data structures
        self._lock = threading.RLock()

        # Main cache storage - LRU ordered
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Efficient lookup indexes
        self._url_index: Dict[str, str] = {}  # url_key -> cache_key
        self._signature_index: Dict[str, str] = {}  # signature -> cache_key
        self._user_index: Dict[str, Set[str]] = {}  # user_id -> set of cache_keys

        # Memory and statistics tracking
        self._total_memory = 0
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        # Background cleanup
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

    def add_job(self, job: Job, user_id: str) -> None:
        """Add job to cache with memory management."""
        with self._lock:
            cache_key = f"{user_id}:{job.id}"

            # Calculate memory size
            job_size = self._calculate_size(job)

            # Remove old entry if exists
            if cache_key in self._cache:
                self._remove_entry(cache_key)

            # Check memory limits and cleanup if needed
            if self._total_memory + job_size > self.max_memory_bytes:
                self._cleanup_memory(job_size)

            # Create cache entry
            now = time.time()
            entry = CacheEntry(
                data=job,
                created_at=now,
                last_accessed=now,
                access_count=0,
                size_bytes=job_size
            )

            # Add to main cache
            self._cache[cache_key] = entry
            self._total_memory += job_size

            # Update indexes
            if job.job_url:
                url_key = f"{user_id}:{job.job_url}"
                self._url_index[url_key] = cache_key

            signature = self._generate_job_signature(job, user_id)
            self._signature_index[signature] = cache_key

            if user_id not in self._user_index:
                self._user_index[user_id] = set()
            self._user_index[user_id].add(cache_key)

            # Periodic cleanup
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_expired()

    def get_job(self, job_id: str, user_id: str) -> Optional[Job]:
        """Get job from cache with efficient lookup."""
        with self._lock:
            cache_key = f"{user_id}:{job_id}"

            if cache_key not in self._cache:
                self._miss_count += 1
                return None

            entry = self._cache[cache_key]

            # Check expiration
            if self._is_expired(entry):
                self._remove_entry(cache_key)
                self._miss_count += 1
                return None

            # Update access statistics
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._hit_count += 1

            # Move to end (LRU)
            self._cache.move_to_end(cache_key)

            return entry.data

    def get_job_by_url(self, url: str, user_id: str) -> Optional[Job]:
        """Get job by URL using efficient index."""
        with self._lock:
            url_key = f"{user_id}:{url}"
            cache_key = self._url_index.get(url_key)

            if not cache_key:
                self._miss_count += 1
                return None

            # Extract job_id from cache_key
            job_id = cache_key.split(':', 1)[1]
            return self.get_job(job_id, user_id)

    def find_similar_job(self, job: Job, user_id: str) -> Optional[Job]:
        """Find similar job using signature index."""
        with self._lock:
            signature = self._generate_job_signature(job, user_id)
            cache_key = self._signature_index.get(signature)

            if not cache_key:
                return None

            job_id = cache_key.split(':', 1)[1]
            return self.get_job(job_id, user_id)

    def get_user_jobs(self, user_id: str, limit: int = None) -> List[Job]:
        """Get all jobs for a user."""
        with self._lock:
            user_keys = self._user_index.get(user_id, set())
            jobs = []

            for cache_key in user_keys:
                if cache_key in self._cache:
                    entry = self._cache[cache_key]
                    if not self._is_expired(entry):
                        jobs.append(entry.data)
                        if limit and len(jobs) >= limit:
                            break

            return jobs

    def remove_job(self, job_id: str, user_id: str) -> None:
        """Remove job from cache."""
        with self._lock:
            cache_key = f"{user_id}:{job_id}"
            self._remove_entry(cache_key)

    def clear_user(self, user_id: str) -> None:
        """Clear all jobs for a user."""
        with self._lock:
            user_keys = self._user_index.get(user_id, set()).copy()
            for cache_key in user_keys:
                self._remove_entry(cache_key)

            if user_id in self._user_index:
                del self._user_index[user_id]

    def clear_all(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._url_index.clear()
            self._signature_index.clear()
            self._user_index.clear()
            self._total_memory = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0

            return {
                "entries": len(self._cache),
                "memory_mb": round(self._total_memory / (1024 * 1024), 2),
                "hit_rate": round(hit_rate * 100, 2),
                "hits": self._hit_count,
                "misses": self._miss_count,
                "evictions": self._eviction_count,
                "users": len(self._user_index)
            }

    def _remove_entry(self, cache_key: str) -> None:
        """Remove entry and update all indexes."""
        if cache_key not in self._cache:
            return

        entry = self._cache[cache_key]
        job = entry.data
        user_id = cache_key.split(':', 1)[0]

        # Remove from main cache
        del self._cache[cache_key]
        self._total_memory -= entry.size_bytes

        # Remove from URL index
        if job.job_url:
            url_key = f"{user_id}:{job.job_url}"
            if url_key in self._url_index:
                del self._url_index[url_key]

        # Remove from signature index
        signature = self._generate_job_signature(job, user_id)
        if signature in self._signature_index:
            del self._signature_index[signature]

        # Remove from user index
        if user_id in self._user_index:
            self._user_index[user_id].discard(cache_key)
            if not self._user_index[user_id]:
                del self._user_index[user_id]

    def _cleanup_memory(self, needed_bytes: int) -> None:
        """Free memory by evicting LRU entries."""
        target_memory = self.max_memory_bytes - needed_bytes

        # Remove expired entries first
        self._cleanup_expired()

        # If still need space, remove LRU entries
        while self._total_memory > target_memory and self._cache:
            # Get least recently used (first in OrderedDict)
            lru_key = next(iter(self._cache))
            self._remove_entry(lru_key)
            self._eviction_count += 1

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = []

        for cache_key, entry in self._cache.items():
            if self._is_expired(entry):
                expired_keys.append(cache_key)

        for key in expired_keys:
            self._remove_entry(key)

        self._last_cleanup = now

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        return time.time() - entry.created_at > self.ttl_seconds

    def _calculate_size(self, job: Job) -> int:
        """Estimate memory size of job object."""
        # Simple estimation - in production, use memory_profiler or similar
        base_size = 200  # Base object overhead

        # Add size for strings
        if job.job_url:
            base_size += len(job.job_url.encode('utf-8'))

        # Add metadata size
        if job.metadata:
            base_size += len(json.dumps(job.metadata).encode('utf-8'))

        return base_size

    def _generate_job_signature(self, job: Job, user_id: str) -> str:
        """Generate job signature for similarity matching."""
        metadata = job.metadata or {}
        title = metadata.get('title', '').lower().strip()
        company = metadata.get('company', '').lower().strip()
        location = metadata.get('location', '').lower().strip()

        signature = f"{user_id}|{title}|{company}|{location}"
        return hashlib.md5(signature.encode('utf-8')).hexdigest()

class ResumeGenerationStatus(Enum):
    """Resume generation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ResumeCache:
    """High-performance resume generation cache with async support."""

    def __init__(self, ttl_seconds=7200):  # 2 hours default
        """Initialize resume cache."""
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def set_status(self, resume_id: str, user_id: str, status: ResumeGenerationStatus,
                         data: Optional[Dict] = None, error: Optional[str] = None) -> None:
        """Set resume generation status."""
        async with self._lock:
            cache_key = f"{user_id}:{resume_id}"
            self._cache[cache_key] = {
                "status": status,
                "data": data,
                "error": error,
                "updated_at": time.time(),
                "user_id": user_id,
                "resume_id": resume_id
            }

    async def get_status(self, resume_id: str, user_id: str) -> Optional[Dict]:
        """Get resume generation status."""
        async with self._lock:
            cache_key = f"{user_id}:{resume_id}"
            entry = self._cache.get(cache_key)

            if not entry:
                return None

            # Check expiration
            if time.time() - entry["updated_at"] > self.ttl_seconds:
                del self._cache[cache_key]
                return None

            return entry.copy()

    async def remove(self, resume_id: str, user_id: str) -> None:
        """Remove resume from cache."""
        async with self._lock:
            cache_key = f"{user_id}:{resume_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]

    async def clear_user_cache(self, user_id: str) -> None:
        """Clear all cache entries for a user."""
        async with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self._cache[key]

    async def cleanup_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._cache.items()
                if now - v["updated_at"] > self.ttl_seconds
            ]

            for key in expired_keys:
                del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "memory_estimate_kb": len(str(self._cache)) // 1024
        }


class CacheMetrics:
    """Centralized cache metrics and monitoring."""

    def __init__(self):
        self.job_cache_stats = {}
        self.search_cache_stats = {}
        self.resume_cache_stats = {}
        self._lock = threading.Lock()

    def update_stats(self, cache_type: str, stats: Dict[str, Any]) -> None:
        """Update cache statistics."""
        with self._lock:
            if cache_type == "job":
                self.job_cache_stats = stats
            elif cache_type == "search":
                self.search_cache_stats = stats
            elif cache_type == "resume":
                self.resume_cache_stats = stats

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall cache statistics."""
        with self._lock:
            return {
                "job_cache": self.job_cache_stats,
                "search_cache": self.search_cache_stats,
                "resume_cache": self.resume_cache_stats,
                "timestamp": datetime.now().isoformat()
            }