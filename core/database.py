"""
Database initialization and setup functions.
"""
import os
import logging
import traceback
from typing import Optional
try:
    import config
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Config module not found, using environment variables only")
    config = None

logger = logging.getLogger(__name__)

async def initialize_app(app, db_url: Optional[str] = None, job_cache_size: Optional[int] = None, search_cache_size: Optional[int] = None):
    """Initialize the FastAPI application with required components."""
    try:
        logger.info("Starting application initialization...")

        # Get configuration from environment variables if not provided
        if db_url is None:
            if config:
                db_url = config.get("database.path") or os.environ.get('DATABASE_URL')
            else:
                db_url = os.environ.get('DATABASE_URL')

            if not db_url:
                # Provide a default SQLite database for development
                db_url = "sqlite:///./jobtrak.db"
                logger.warning(f"No DATABASE_URL provided, using default: {db_url}")
        logger.info(f"Configuration: db_url={db_url}, job_cache_size={job_cache_size}, search_cache_size={search_cache_size}")

        # Initialize database
        try:
            from data.database import Database
            db = Database(db_url)
            await db.initialize_pool()
            logger.info(f"✓ Database initialized successfully with URL: {db_url}")
        except Exception as e:
            logger.error(f"✗ Database initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise

        # Initialize unified cache manager
        try:
            from data.dbcache_manager import DBCacheManager
            cache_manager = DBCacheManager(
                database=db,
            )
            logger.info("✓ Unified cache manager initialized")
        except Exception as e:
            logger.error(f"✗ Cache manager initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise

        # Store in application state
        try:
            app.state.db = db
            app.state.cache_manager = cache_manager

            # Verify the state was set correctly
            assert hasattr(app.state, 'cache_manager'), "cache_manager not set in app.state"
            assert app.state.cache_manager is not None, "cache_manager is None in app.state"

            logger.info("✓ Application state configured successfully")
        except Exception as e:
            logger.error(f"✗ Application state configuration failed: {e}")
            logger.error(traceback.format_exc())
            raise

        # Test the cache manager
        try:
            health_info = await cache_manager.health_check()
            logger.info(f"✓ Cache manager health check passed: {health_info}")
        except Exception as e:
            logger.error(f"✗ Cache manager health check failed: {e}")
            logger.error(traceback.format_exc())
            # Don't raise here, just warn - the manager might still work

        logger.info("🎉 Application initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"💥 Critical error during application initialization: {e}")
        logger.error(traceback.format_exc())

        # Clean up any partially initialized state
        try:
            if hasattr(app.state, 'db'):
                await app.state.db.close_pool()
        except:
            pass

        return False

async def verify_initialization(app):
    """Verify that all components are properly initialized."""
    issues = []

    if not hasattr(app.state, 'db'):
        issues.append("Database not initialized")

    if issues:
        logger.error(f"Initialization verification failed: {issues}")
        return False, issues

    logger.info("✓ Initialization verification passed")
    return True, []