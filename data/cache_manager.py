# cache_manager.py - Add this as a new file
import asyncio
from typing import Dict, Optional
from services.resume_generator import ResumeGenerationCache

class GlobalCacheManager:
    """Singleton cache manager for the application."""

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.resume_cache = ResumeGenerationCache()
            self._initialized = True

    async def get_resume_status(self, resume_id: str, user_id: str) -> Optional[Dict]:
        """Get resume generation status."""
        return await self.resume_cache.get_status(resume_id, user_id)

    async def set_resume_status(self, resume_id: str, user_id: str, status, data=None, error=None):
        """Set resume generation status."""
        await self.resume_cache.set_status(resume_id, user_id, status, data, error)

    async def remove_resume(self, resume_id: str, user_id: str):
        """Remove resume from cache."""
        await self.resume_cache.remove(resume_id, user_id)

    async def clear_user_cache(self, user_id: str):
        """Clear all cache for a user."""
        await self.resume_cache.clear_user_cache(user_id)

# Global instance
cache_manager = GlobalCacheManager()