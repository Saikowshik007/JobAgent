"""
System management routes for health checks, cache management, and utilities.
"""
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import logging

from core.dependencies import get_cache_manager, get_user_id
from data.dbcache_manager import DBCacheManager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle all OPTIONS requests explicitly"""
    return {"message": "OK"}

@router.get("/status")
async def get_system_status(
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get the overall status of the job tracking system."""
    try:
        # Get health check from unified cache manager
        health_info = await cache_manager.health_check()

        # Get job statistics
        job_stats = await cache_manager.get_job_stats(user_id)

        # Get cache statistics
        cache_stats = cache_manager.get_cache_stats()

        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "health": health_info,
            "job_stats": job_stats,
            "cache_stats": cache_stats
        }
    except Exception as e:
        logger.error(f"Error getting system status for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache/clear")
async def clear_cache(
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Clear user's cache data."""
    try:
        # Clear all cache data using unified cache manager
        await cache_manager.clear_user_cache(user_id)

        return {
            "message": "Cache cleared successfully",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/cleanup")
async def cleanup_cache(
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Clean up expired cache entries."""
    try:
        await cache_manager.cleanup_expired_cache()

        return {
            "message": "Cache cleanup completed",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def get_cache_stats(
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get detailed cache statistics."""
    try:
        stats = cache_manager.get_cache_stats()
        health = await cache_manager.health_check()

        return {
            "cache_stats": stats,
            "health": health,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))