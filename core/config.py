"""
Configuration management for the JobTrak API.
"""
import os
import logging

from starlette.middleware.cors import CORSMiddleware

from main import app

logger = logging.getLogger(__name__)

def get_allowed_origins():
    """Get the list of allowed CORS origins."""
    # Base origins that are always allowed
    origins = [
        "https://job-agent-ui.vercel.app",
        "https://jobtrackai.duckdns.org",
        "http://jobtrackai.duckdns.org",
        "http://localhost:3000",
        "https://localhost:3000",
        "http://127.0.0.1:3000",
        "https://127.0.0.1:3000",
        "http://localhost:3001",
        "https://localhost:3001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://simplify.jobs"
    ]

    # Add debug origins if in debug mode
    if os.environ.get('API_DEBUG'):
        debug_origins = [
            "http://localhost:3002",
            "http://localhost:5000",
            "http://192.168.1.100:3000",
            "http://192.168.1.101:3000",
        ]
        origins.extend(debug_origins)

    logger.info(f"CORS allowed origins: {origins}")
    return origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),  # NO WILDCARD with credentials
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=[
        # Standard headers
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "Cache-Control",
        "DNT",
        "If-Modified-Since",
        "Keep-Alive",
        "Origin",
        "User-Agent",
        "X-Requested-With",
        "Range",

        # Custom API headers
        "X-Api-Key",
        "x-api-key",
        "X-User-Id",
        "x-user-id",
        "x_user_id",

        # Additional headers
        "X-CSRF-Token",
        "X-Forwarded-For",
        "X-Forwarded-Proto",
        "X-Real-IP",
    ],
    expose_headers=[
        "Content-Range",
        "X-Content-Range",
        "X-Total-Count",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Credentials"
    ],
    max_age=3600,  # Cache preflight requests for 1 hour
)