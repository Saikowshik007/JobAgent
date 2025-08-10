"""
Dependency injection functions for FastAPI endpoints.
"""
import json

from fastapi import HTTPException, Header, Request, Form
from typing import Optional
import logging

from dataModels.user_model import User

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


async def get_user_from_form(user: str = Form(...)):
    """
    Extracts the 'user' field from multipart/form-data and parses it into a User object.
    The 'user' field must be a JSON-encoded string.
    """
    try:
        user_data = json.loads(user)
        return User(**user_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'user' form field")
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid user data: {e}")
