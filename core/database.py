"""
Database initialization and setup functions.
"""
import os
import logging
import config
from data.database import Database
from data.dbcache_manager import DBCacheManager
from data.cache import JobCache, ResumeCache

logger = logging.getLogger(__name__)

async def initialize_app(app, db_url=None, job_cache_size=None, search_cache_size=None):
    """Initialize the FastAPI application with required components."""
    try:
        # Get configuration from environment variables if not provided
        if db_url is None:
            db_url = config.get("database.url") or os.environ.get('DATABASE_URL')

        if job_cache_size is None:
            job_cache_size = int(config.get("cache.job_cache_size", 1000))

        if search_cache_size is None:
            search_cache_size = int(config.get("cache.search_cache_size", 1000))

        # Initialize database
        db = Database(db_url)
        await db.initialize_pool()
        logger.info(f"Initialized database with URL {db_url}")

        # Initialize caches
        job_cache = JobCache(max_size=job_cache_size)
        resume_cache = ResumeCache()
        logger.info(f"Initialized caches with sizes - job: {job_cache_size}, search: {search_cache_size}")

        # Initialize unified cache manager
        cache_manager = DBCacheManager(
            database=db,
            job_cache=job_cache,
            resume_cache=resume_cache
        )
        logger.info("Initialized unified cache manager")

        # Store in application state
        app.state.db = db
        app.state.job_cache = job_cache
        app.state.resume_cache = resume_cache
        app.state.cache_manager = cache_manager

        logger.info("Application initialization complete")
        return True
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        return False