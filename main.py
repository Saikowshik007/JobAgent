import logging
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import routers
from routers import jobs, resume, system, simplify
from core.config import get_allowed_origins
from core.database import initialize_app
import config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=config.get("api.level"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="JobTrak API",
    description="API for analyzing job postings and tracking job applications",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=[
        "Accept", "Accept-Language", "Content-Language", "Content-Type",
        "Authorization", "Cache-Control", "DNT", "If-Modified-Since",
        "Keep-Alive", "Origin", "User-Agent", "X-Requested-With", "Range",
        "X-Api-Key", "x-api-key", "X-User-Id", "x-user-id", "x_user_id",
        "X-CSRF-Token", "X-Forwarded-For", "X-Forwarded-Proto", "X-Real-IP",
    ],
    expose_headers=[
        "Content-Range", "X-Content-Range", "X-Total-Count",
        "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"
    ],
    max_age=3600,
)

# Include routers
app.include_router(system.router, prefix="/api", tags=["System"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(resume.router, prefix="/api/resume", tags=["Resume"])
app.include_router(simplify.router, prefix="/api/simplify", tags=["Simplify"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        import requests
        public_ip = requests.get('https://ipinfo.io/ip', timeout=5).text.strip()
        logger.info(f"🌐 Server starting - Public IP: {public_ip}")
        logger.info(f"🔗 API accessible at: https://{public_ip}/api/")
        app.state.public_ip = public_ip
    except Exception as e:
        logger.warning(f"Could not detect public IP: {e}")

    success = await initialize_app(app)
    if not success:
        logger.error("Failed to initialize application during startup")
        raise Exception("Application initialization failed")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down."""
    try:
        if hasattr(app.state, 'db'):
            await app.state.db.close_pool()
        logger.info("Successfully cleaned up resources on shutdown")
    except Exception as e:
        logger.error(f"Error cleaning up resources: {e}")

if __name__ == "__main__":
    host = os.environ.get('API_HOST', '0.0.0.0')
    port = int(os.environ.get('API_PORT', 8000))
    debug = os.environ.get('API_DEBUG', '').lower() in ('true', '1', 'yes')

    logger.info(f"Starting API server on {host}:{port} (debug={debug})")
    uvicorn.run(app, host=host, port=port, log_level="debug" if debug else "info")