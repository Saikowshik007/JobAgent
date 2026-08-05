"""
Database initialization and setup functions.
"""
import os
import logging
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
        logger.info("application_initialization_started")

        # Get configuration from environment variables if not provided
        if db_url is None:
            # Environment configuration must win in containers and production.
            # The YAML value remains a local-development fallback only.
            db_url = os.environ.get('DATABASE_URL')
            if not db_url and config:
                db_url = config.get("database.path")

            if not db_url:
                # Provide a default SQLite database for development
                db_url = "sqlite:///./jobtrak.db"
                logger.warning("database_url_missing_using_development_default")
        logger.info("application_configuration_loaded", extra={"cache.job_size": job_cache_size, "cache.search_size": search_cache_size})

        # Initialize database
        try:
            from data.database import Database
            db = Database(db_url)
            await db.initialize_pool()
            logger.info("database_initialized")
        except Exception:
            logger.exception("database_initialization_failed")
            raise

        # Initialize unified cache manager
        try:
            from data.dbcache_manager import DBCacheManager
            cache_manager = DBCacheManager(
                database=db,
            )
            logger.info("cache_manager_initialized")
        except Exception:
            logger.exception("cache_manager_initialization_failed")
            raise

        # Store in application state
        try:
            app.state.db = db
            app.state.cache_manager = cache_manager

            # Verify the state was set correctly
            assert hasattr(app.state, 'cache_manager'), "cache_manager not set in app.state"
            assert app.state.cache_manager is not None, "cache_manager is None in app.state"

            logger.info("application_state_configured")
        except Exception:
            logger.exception("application_state_configuration_failed")
            raise

        # Test the cache manager
        try:
            health_info = await cache_manager.health_check()
            logger.info("cache_manager_health_check_passed", extra={"health.status": health_info.get("status")})
        except Exception:
            logger.exception("cache_manager_health_check_failed")
            # Don't raise here, just warn - the manager might still work

        logger.info("application_initialization_completed")
        return True

    except Exception:
        logger.exception("application_initialization_failed")

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
        logger.error("application_initialization_verification_failed", extra={"error.reason": "; ".join(issues)})
        return False, issues

    logger.info("application_initialization_verification_passed")
    return True, []
