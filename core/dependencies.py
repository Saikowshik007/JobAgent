"""
Dependency injection functions for FastAPI endpoints.
"""
from fastapi import HTTPException, Header, Request
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Default user ID if none provided
DEFAULT_USER_ID = "default_user"

def get_cache_manager(request: Request):
    """Get the unified cache manager instance."""
    app = request.app

    # Check if cache manager exists
    if not hasattr(app.state, "cache_manager"):
        logger.error("Cache manager not found in app.state")
        logger.error(f"Available app.state attributes: {dir(app.state)}")

        # Try to get more detailed information
        state_dict = {}
        for attr in dir(app.state):
            if not attr.startswith('_'):
                try:
                    state_dict[attr] = getattr(app.state, attr)
                except Exception as e:
                    state_dict[attr] = f"Error accessing: {e}"

        logger.error(f"App state contents: {state_dict}")

        raise HTTPException(
            status_code=500,
            detail="Cache manager not initialized. Please check server configuration and logs."
        )

    cache_manager = app.state.cache_manager
    if cache_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Cache manager is None. Initialization may have failed."
        )

    return cache_manager

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

async def get_optional_user_key(x_api_key: Optional[str] = Header(None)):
    """Get API key from header, but don't require it."""
    if not x_api_key:
        logger.info("No api_key provided in header (optional)")
        return None
    logger.info("Received API key")
    return x_api_key