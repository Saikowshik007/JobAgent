"""
Dependency injection functions for FastAPI endpoints.
"""
from fastapi import HTTPException, Header
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Default user ID if none provided
DEFAULT_USER_ID = "default_user"

def get_cache_manager():
    """Get the unified cache manager instance."""
    from main import app

    if not hasattr(app.state, "cache_manager"):
        raise HTTPException(
            status_code=500,
            detail="Cache manager not initialized. Please check server configuration."
        )
    return app.state.cache_manager

async def get_user_id(x_user_id: Optional[str] = Header(None)):
    """Get user_id from header or use default if not provided."""
    if not x_user_id:
        logger.warning("No user_id provided in header, using default")
        return DEFAULT_USER_ID
    logger.info(f"Received user_id: {x_user_id}")
    return x_user_id

async def get_user_key(x_api_key: Optional[str] = Header(None)):
    """Get API key from header."""
    if not x_api_key:
        logger.warning("No api_key provided in header")
        raise HTTPException(
            status_code=401,
            detail="No API key provided. Please include X-Api-Key header."
        )
    logger.info("Received API key")
    return x_api_key