"""
Main FastAPI application with proper initialization and enhanced CORS configuration.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware  # Use Starlette directly

from core.database import initialize_app
from routers import system, jobs, resume, simplify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_allowed_origins():
    """Get the list of allowed CORS origins."""
    # Base origins that are always allowed
    origins = [
        # Production origins - EXACT MATCHES (most important)
        "https://job-agent-ui.vercel.app",
        "https://jobtrackai.duckdns.org",
        "http://jobtrackai.duckdns.org",

        # Vercel preview deployments (common pattern)
        "https://job-agent-ui-git-main-your-username.vercel.app",
        "https://job-agent-ui-git-dev-your-username.vercel.app",

        # External integrations
        "https://simplify.jobs",

        # Development origins
        "http://localhost:3000",
        "https://localhost:3000",
        "http://127.0.0.1:3000",
        "https://127.0.0.1:3000",
        "http://localhost:3001",
        "https://localhost:3001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Add debug origins if in debug mode
    if os.environ.get('API_DEBUG') or os.environ.get('DEBUG'):
        debug_origins = [
            "http://localhost:3002",
            "http://localhost:5000",
            "http://192.168.1.100:3000",
            "http://192.168.1.101:3000",
            # Add wildcard for Vercel previews in debug mode
            "https://*.vercel.app",
        ]
        origins.extend(debug_origins)

    # Add any additional origins from environment variable
    env_origins = os.environ.get('ADDITIONAL_CORS_ORIGINS', '')
    if env_origins:
        additional_origins = [origin.strip() for origin in env_origins.split(',')]
        origins.extend(additional_origins)

    logger.info(f"CORS allowed origins: {origins}")
    return origins

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting application initialization...")

    try:
        # Initialize the application with all required components
        success = await initialize_app(app)

        if not success:
            logger.error("Failed to initialize application")
            raise RuntimeError("Application initialization failed")

        logger.info("Application initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise
    finally:
        # Cleanup code here if needed
        logger.info("Application shutting down...")

        # Close database connections if they exist
        if hasattr(app.state, 'db'):
            try:
                await app.state.db.close_pool()
                logger.info("Database connections closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}")

# Create FastAPI app with lifespan
app = FastAPI(
    title="JobTrak API",
    description="API for analyzing job postings and tracking job applications",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS IMMEDIATELY after app creation - this is critical!
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),  # Specific origins
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
        "Referer",
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
        # Common frontend framework headers
        "X-Vercel-Id",
        "X-Deployment-Id",
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

# Include routers AFTER CORS middleware
app.include_router(system.router, prefix="/api", tags=["System"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(resume.router, prefix="/api/resume", tags=["Resume"])
app.include_router(simplify.router, prefix="/api/simplify", tags=["Simplify"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "JobTrak API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": "2025-06-08"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint."""
    return {
        "status": "healthy",
        "cache_manager_initialized": hasattr(app.state, "cache_manager"),
        "database_initialized": hasattr(app.state, "db"),
        "timestamp": "2025-06-08",
        "version": "1.0.0"
    }

# Add a specific CORS test endpoint
@app.get("/api/cors-test")
async def cors_test():
    """Test endpoint specifically for CORS debugging."""
    return {
        "message": "CORS test successful",
        "timestamp": "2025-06-08",
        "origin_allowed": True,
        "cors_origins": get_allowed_origins()
    }

# Add OPTIONS handler for preflight requests
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle OPTIONS requests for CORS preflight."""
    return {"message": "OK"}

if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info"
    )