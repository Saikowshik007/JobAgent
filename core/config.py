"""
Configuration management for the JobTrak API.
"""
import os
import logging

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