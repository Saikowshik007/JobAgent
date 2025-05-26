"""
Main FastAPI application with proper initialization.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.database import initialize_app
from routers import system, jobs, resume, simplify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Include routers
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
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "cache_manager_initialized": hasattr(app.state, "cache_manager"),
        "database_initialized": hasattr(app.state, "db")
    }

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