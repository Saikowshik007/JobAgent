"""
System management routes for health checks, cache management, and utilities.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from datetime import datetime
import logging
import traceback

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
        request: Request,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get the overall status of the job tracking system."""
    try:
        logger.info(f"Getting system status for user: {user_id}")

        # Get health check from unified cache manager
        try:
            health_info = await cache_manager.health_check()
            logger.info("✓ Health check completed")
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            health_info = {"status": "degraded", "error": str(e)}

        # Get job statistics
        try:
            job_stats = await cache_manager.get_job_stats(user_id)
            logger.info("✓ Job stats retrieved")
        except Exception as e:
            logger.warning(f"Job stats failed: {e}")
            job_stats = {"error": str(e)}

        # Get cache statistics
        try:
            cache_stats = cache_manager.get_cache_stats()
            logger.info("✓ Cache stats retrieved")
        except Exception as e:
            logger.warning(f"Cache stats failed: {e}")
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
        logger.error(f"Error getting system status for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"System status error: {str(e)}")

@router.delete("/cache/clear")
async def clear_cache(
        request: Request,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Clear user's cache data."""
    try:
        logger.info(f"Clearing cache for user: {user_id}")

        # Clear all cache data using unified cache manager
        await cache_manager.clear_user_cache(user_id)

        logger.info(f"✓ Cache cleared successfully for user: {user_id}")
        return {
            "message": "Cache cleared successfully",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")

@router.post("/cache/cleanup")
async def cleanup_cache(
        request: Request,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Clean up expired cache entries."""
    try:
        logger.info("Starting cache cleanup")

        await cache_manager.cleanup_expired_cache()

        logger.info("✓ Cache cleanup completed successfully")
        return {
            "message": "Cache cleanup completed",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Cache cleanup error: {str(e)}")

@router.get("/cache/stats")
async def get_cache_stats(
        request: Request,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get detailed cache statistics."""
    try:
        logger.info("Getting cache statistics")

        stats = cache_manager.get_cache_stats()
        health = await cache_manager.health_check()

        logger.info("✓ Cache statistics retrieved successfully")
        return {
            "cache_stats": stats,
            "health": health,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Cache stats error: {str(e)}")
