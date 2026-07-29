"""
System management routes for health checks, cache management, and utilities.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from datetime import datetime
import logging

from core.dependencies import get_cache_manager
from data.dbcache_manager import DBCacheManager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle all OPTIONS requests explicitly"""
    return {"message": "OK"}

@router.get("/{user_id}/status")
async def get_system_status(
        user_id: str,
        request: Request,
        cache_manager: DBCacheManager = Depends(get_cache_manager)
):
    """Get the overall status of the job tracking system."""
    try:
        logger.debug("system_status_requested")

        # Get health check from unified cache manager
        try:
            health_info = await cache_manager.health_check()
            logger.debug("system_health_check_completed")
        except Exception as e:
            logger.warning("system_health_check_failed", extra={"error.reason": str(e)})
            health_info = {"status": "degraded", "error": str(e)}

        # Get job statistics
        try:
            job_stats = await cache_manager.get_job_stats(user_id)
            logger.debug("system_job_stats_loaded")
        except Exception as e:
            logger.warning("system_job_stats_load_failed", extra={"error.reason": str(e)})
            job_stats = {"error": str(e)}

        # Get cache statistics
        try:
            cache_stats = cache_manager.get_cache_stats()
            logger.debug("system_cache_stats_loaded")
        except Exception as e:
            logger.warning("system_cache_stats_load_failed", extra={"error.reason": str(e)})
            cache_stats = {"error": str(e)}

        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "health": health_info,
            "job_stats": job_stats,
            "cache_stats": cache_stats,
            "initialization_status": {
                "cache_manager": cache_manager is not None,
                "database": hasattr(request.app.state, 'db'),
                "job_cache": hasattr(request.app.state, 'job_cache'),
                "resume_cache": hasattr(request.app.state, 'resume_cache')
            }
        }
    except Exception as e:
        logger.exception("system_status_failed")
        raise HTTPException(status_code=500, detail=f"System status error: {str(e)}")

@router.delete("/{user_id}/cache/clear")
async def clear_cache(
        user_id: str,
        request: Request,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
):
    """Clear user's cache data."""
    try:
        logger.info("cache_clear_started")

        # Clear all cache data using unified cache manager
        await cache_manager.clear_user_cache(user_id)

        logger.info("cache_clear_completed")
        return {
            "message": "Cache cleared successfully",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("cache_clear_failed")
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")

@router.post("/{user_id}/cache/cleanup")
async def cleanup_cache(
        user_id: str,
        request: Request,
        cache_manager: DBCacheManager = Depends(get_cache_manager)
):
    """Clean up expired cache entries."""
    try:
        logger.info("cache_cleanup_started")

        await cache_manager.cleanup_expired_cache()

        logger.info("cache_cleanup_completed")
        return {
            "message": "Cache cleanup completed",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        }
    except Exception as e:
        logger.exception("cache_cleanup_failed")
        raise HTTPException(status_code=500, detail=f"Cache cleanup error: {str(e)}")

@router.get("/{user_id}/cache/stats")
async def get_cache_stats(
        user_id: str,
        request: Request,
        cache_manager: DBCacheManager = Depends(get_cache_manager)
):
    """Get detailed cache statistics."""
    try:
        logger.debug("cache_stats_requested")

        stats = cache_manager.get_cache_stats()
        health = await cache_manager.health_check()

        logger.debug("cache_stats_loaded")
        return {
            "cache_stats": stats,
            "health": health,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("cache_stats_failed")
        raise HTTPException(status_code=500, detail=f"Cache stats error: {str(e)}")
