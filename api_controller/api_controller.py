from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os
import time
import hashlib
from enum import Enum
from datetime import datetime
import uvicorn
from dotenv import load_dotenv

import config
from dataModels.api_models import JobSearchRequest, JobStatusUpdateRequest, GenerateResumeRequest, \
    UploadToSimplifyRequest

# Load environment variables from .env file if it exists
load_dotenv()

# Check if we're running in Docker and import the docker_webdriver
if os.environ.get('SELENIUM_REMOTE_URL'):
    pass

from data.database import Database
from drivers.job_searcher import JobSearcher
from drivers.linkedin_driver import LinkedInIntegration
from drivers.browser_manager import BrowserDriverManager
from dataModels.data_models import JobStatus
from data.dbcache_manager import DBCacheManager
from data.cache import JobCache, SearchCache

# Configure logging
logging.basicConfig(
    level=config.get("api.level"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="JobTrak API",
    description="API for searching and tracking job applications from LinkedIn",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Global LinkedIn driver instance - initialized when needed
_linkedin_driver = None

# Initialize function to set up application state
def initialize(db_path=None, job_cache_size=None, search_cache_size=None):
    """Initialize the FastAPI application with required components."""
    try:
        # Get configuration from environment variables if not provided
        if db_path is None:
            db_path = config.get("database.path")

        if job_cache_size is None:
            job_cache_size = int(config.get("cache.job_cache_size",1000))

        if search_cache_size is None:
            search_cache_size = int(config.get("cache.search"
                                               "_cache_size",1000))

        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path) if '/' in db_path else '.', exist_ok=True)

        # Initialize database
        db = Database(db_path)
        logger.info(f"Initialized database at {db_path}")

        # Initialize caches
        job_cache = JobCache(max_size=job_cache_size)
        search_cache = SearchCache(max_size=search_cache_size)
        logger.info(f"Initialized caches with sizes - job: {job_cache_size}, search: {search_cache_size}")

        # Initialize cache manager
        db_manager = DBCacheManager(
            database=db,
            job_cache=job_cache,
            search_cache=search_cache
        )
        logger.info("Initialized database cache manager")

        # Store in application state
        app.state.db = db
        app.state.job_cache = job_cache
        app.state.search_cache = search_cache
        app.state.db_manager = db_manager

        logger.info("Application initialization complete")
        return True
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        return False

# Initialize the application when this module is loaded
initialize()

# Dependency to get database manager
def get_db_manager():
    """Get the database manager instance."""
    if not hasattr(app.state, "db_manager"):
        # If db_manager is not in app.state, initialize the application
        logger.warning("Database manager not found in app state. Initializing the application...")
        initialize()

        if not hasattr(app.state, "db_manager"):
            # If still not available, raise an exception
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize database manager. Please check server configuration."
            )

    return app.state.db_manager

# System status endpoint
@app.get("/api/status", tags=["System"])
async def get_system_status(db_manager: DBCacheManager = Depends(get_db_manager)):
    """Get the overall status of the job tracking system."""
    try:
        # Check database connection
        db = db_manager.db if db_manager else None
        db_status = "OK"
        job_count = 0
        job_stats = {}

        try:
            if db:
                job_count = len(db.get_all_jobs())
                job_stats = {
                    "total": job_count,
                    "new": len(db.get_all_jobs(status=JobStatus.NEW)),
                    "interested": len(db.get_all_jobs(status=JobStatus.INTERESTED)),
                    "applied": len(db.get_all_jobs(status=JobStatus.APPLIED)),
                    "interviews": len(db.get_all_jobs(status=JobStatus.INTERVIEW)),
                    "offers": len(db.get_all_jobs(status=JobStatus.OFFER)),
                    "rejected": len(db.get_all_jobs(status=JobStatus.REJECTED))
                }
        except Exception as e:
            db_status = f"Error: {str(e)}"
            job_stats = {
                "total": 0,
                "new": 0,
                "interested": 0,
                "applied": 0,
                "interviews": 0,
                "offers": 0,
                "rejected": 0
            }

        # Check LinkedIn connection
        linkedin_status = "Not Connected"
        global _linkedin_driver
        if _linkedin_driver and _linkedin_driver.logged_in:
            linkedin_status = "Connected"

        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": db_status,
                "linkedin": linkedin_status
            },
            "job_stats": job_stats
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_linkedin_driver(headless=True, force_new=False):
    """Initialize LinkedIn driver if not already initialized."""
    global _linkedin_driver

    try:
        if _linkedin_driver is None or force_new:
            # Close existing driver if forcing new
            if force_new and _linkedin_driver:
                try:
                    _linkedin_driver.driver.quit()
                except:
                    pass
                _linkedin_driver = None

            # Initialize LinkedIn driver with credentials from environment variables
            email = config.get("credentials.linkedin.email")
            password = config.get("credentials.linkedin.password")

            if not email or not password:
                raise ValueError("LinkedIn credentials not found. Set LINKEDIN_EMAIL and LINKEDIN_PASSWORD environment variables.")

            _linkedin_driver = LinkedInIntegration(
                email=email,
                password=password,
                headless=headless
            )

            # Login to LinkedIn
            if not _linkedin_driver.login():
                raise ValueError("Failed to login to LinkedIn. Check credentials.")

        return _linkedin_driver
    except Exception as e:
        logger.error(f"Error initializing LinkedIn driver: {e}")
        raise HTTPException(status_code=500, detail=f"LinkedIn driver error: {str(e)}")

# Job search endpoint
@app.post("/api/jobs/search", tags=["Jobs"])
async def search_jobs(
        request: JobSearchRequest,
        background_tasks: BackgroundTasks,
        db_manager: DBCacheManager = Depends(get_db_manager)
):
    """Search for jobs on LinkedIn based on provided criteria."""
    try:
        # Get LinkedIn driver
        linkedin_driver = get_linkedin_driver(headless=request.headless)

        # Create job searcher
        job_searcher = JobSearcher(linkedin_driver, db_manager=db_manager)

        # Perform job search
        logger.info(f"Searching for jobs: {request.keywords} in {request.location}")
        start_time = time.time()

        # Extract filters from request
        filters = {}
        if request.filters:
            filters = {k: v for k, v in request.filters.dict().items() if v is not None}

        job_listings = job_searcher.search_jobs(
            keywords=request.keywords,
            location=request.location,
            filters=filters,
            max_listings=request.max_jobs
        )

        duration = time.time() - start_time

        if not job_listings:
            return {
                "message": "No jobs found matching your criteria",
                "count": 0,
                "duration_seconds": duration,
                "jobs": []
            }

        # Generate search ID
        search_id = hashlib.md5(f"{request.keywords}|{request.location}|{str(filters)}".encode()).hexdigest()

        # Save job IDs for reference
        job_ids = [job.get('id') for job in job_listings if job.get('id')]

        db_manager.save_search_results(request.keywords, request.location, filters, job_listings)

        return {
            "message": f"Found {len(job_listings)} jobs matching your criteria",
            "count": len(job_listings),
            "duration_seconds": duration,
            "search_id": search_id,
            "jobs": job_listings
        }

    except Exception as e:
        logger.error(f"Error searching for jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get jobs endpoint
@app.get("/api/jobs", tags=["Jobs"])
async def get_jobs(
        status: Optional[str] = None,
        company: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        db_manager: DBCacheManager = Depends(get_db_manager)
):
    """Get jobs from the database with optional filtering."""
    try:
        db = db_manager.db

        # Get all jobs first
        jobs = db.get_all_jobs(status=status if status else None)

        # Apply additional filters
        if company:
            jobs = [job for job in jobs if company.lower() in job.company.lower()]

        # Get total count
        total_count = len(jobs)

        # Apply pagination
        jobs = jobs[offset:offset+limit]

        # Convert to dict for JSON response
        jobs_dict = [job.to_dict() for job in jobs]

        return {
            "count": len(jobs_dict),
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "jobs": jobs_dict
        }

    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get specific job endpoint
@app.get("/api/jobs/{job_id}", tags=["Jobs"])
async def get_job(
        job_id: str,
        db_manager: DBCacheManager = Depends(get_db_manager)
):
    """Get a specific job by ID."""
    try:
        db = db_manager.db
        job = db.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found with ID: {job_id}")

        return job.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update job status endpoint
@app.put("/api/jobs/{job_id}/status", tags=["Jobs"])
async def update_job_status(
        job_id: str,
        request: JobStatusUpdateRequest,
        db_manager: DBCacheManager = Depends(get_db_manager)
):
    """Update the status of a job."""
    try:
        db = db_manager.db

        status = request.status.value

        # Update the job status
        try:
            status_enum = JobStatus(status)
        except ValueError:
            valid_statuses = [s.value for s in JobStatus]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status value: {status}. Valid values are: {valid_statuses}"
            )

        success = db.update_job_status(job_id, status_enum)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to update job status. Job may not exist: {job_id}"
            )

        # Get the updated job
        job = db.get_job(job_id)

        return {
            "message": f"Job status updated to {status}",
            "job": job.to_dict() if job else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Generate resume endpoint (placeholder, to be implemented later)
@app.post("/api/resume/generate", tags=["Resume"])
async def generate_resume(
        request: GenerateResumeRequest,
        db_manager: DBCacheManager = Depends(get_db_manager)
):
    """Generate a tailored resume for a specific job."""
    # This is a placeholder - you'll implement the actual resume generation later
    try:
        db = db_manager.db

        # Check if job exists
        job = db.get_job(request.job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found with ID: {request.job_id}")

        # Placeholder response
        return {
            "message": "Resume generation functionality will be implemented in a future update",
            "job_id": request.job_id,
            "template": request.template,
            "customize": request.customize,
            "job": job.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating resume for job {request.job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Upload to Simplify endpoint (placeholder, to be implemented later)
@app.post("/api/resume/upload-to-simplify", tags=["Resume"])
async def upload_to_simplify(
        request: UploadToSimplifyRequest,
        db_manager: DBCacheManager = Depends(get_db_manager)
):
    """Upload a resume to Simplify.jobs."""
    # This is a placeholder - you'll implement the actual Simplify integration later
    try:
        db = db_manager.db

        # Check if job exists
        job = db.get_job(request.job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found with ID: {request.job_id}")

        # Check if resume exists
        resume_id = request.resume_id or job.resume_id
        if not resume_id:
            raise HTTPException(status_code=400, detail="No resume ID provided or associated with this job")

        resume = db.get_resume(resume_id)
        if not resume:
            raise HTTPException(status_code=404, detail=f"Resume not found with ID: {resume_id}")

        # Placeholder response
        return {
            "message": "Simplify.jobs upload functionality will be implemented in a future update",
            "job_id": request.job_id,
            "resume_id": resume_id,
            "job": job.to_dict(),
            "resume": resume.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading resume to Simplify for job {request.job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Shutdown event - clean up resources
@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when the application shuts down."""
    try:
        global _linkedin_driver
        if _linkedin_driver:
            try:
                _linkedin_driver.driver.quit()
            except:
                pass
            finally:
                _linkedin_driver = None

        # Also make sure BrowserDriverManager releases any driver
        BrowserDriverManager.release_driver()
        BrowserDriverManager.close_all_drivers()

        logger.info("Successfully cleaned up resources on shutdown")

    except Exception as e:
        logger.error(f"Error cleaning up resources: {e}")

# Run the server if this file is executed directly
# Initialize the application when the module is loaded
initialize()
if __name__ == "__main__":
    # Initialize application
    if not initialize():
        logger.error("Failed to initialize application. Exiting.")
        exit(1)

    # Run the FastAPI app with Uvicorn
    host = os.environ.get('API_HOST', '0.0.0.0')
    port = int(os.environ.get('API_PORT', 8000))
    debug = os.environ.get('API_DEBUG', '').lower() in ('true', '1', 'yes')

    logger.info(f"Starting API server on {host}:{port} (debug={debug})")
    uvicorn.run(app, host=host, port=port, log_level="debug" if debug else "info")