import asyncio
import io
import hashlib
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, Header, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import logging
import os
import time
from datetime import datetime
import uvicorn
from dotenv import load_dotenv
from starlette.responses import StreamingResponse, FileResponse

import config
from data.cache_manager import cache_manager
from dataModels.api_models import JobStatusUpdateRequest, GenerateResumeRequest, \
    UploadToSimplifyRequest
from services.resume_generator import ResumeGenerator
from services.resume_improver import ResumeImprover  # Import the ResumeImprover class

# Load environment variables from .env file if it exists
load_dotenv()

from data.database import Database
from dataModels.data_models import JobStatus, Job
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
    description="API for analyzing job postings and tracking job applications",
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

# Default user ID if none provided
DEFAULT_USER_ID = "default_user"

# Initialize function to set up application state
def initialize(db_path=None, job_cache_size=None, search_cache_size=None):
    """Initialize the FastAPI application with required components."""
    try:
        # Get configuration from environment variables if not provided
        if db_path is None:
            db_path = config.get("database.path")

        if job_cache_size is None:
            job_cache_size = int(config.get("cache.job_cache_size", 1000))

        if search_cache_size is None:
            search_cache_size = int(config.get("cache.search_cache_size", 1000))

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
        app.state.cache_manager = cache_manager  # Add global cache manager

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

async def get_user_id(x_user_id: Optional[str] = Header(None, alias="X_user_id")):
    """Get user_id from header or use default if not provided."""
    if not x_user_id:
        logger.warning("No user_id provided in header, using default")
        return DEFAULT_USER_ID
    return x_user_id

async def get_user_key(x_api_key: Optional[str] = Header(None, alias="X-Api-Key")):
    if not x_api_key:
        logger.warning("No api_key provided in header, using default")
        raise HTTPException(
            status_code=500,
            detail="No api key provided"
        )
    return x_api_key

# System status endpoint
@app.get("/api/status", tags=["System"])
async def get_system_status(
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
):
    """Get the overall status of the job tracking system - now non-blocking."""
    try:
        # This should now be much faster since resume generation doesn't block
        db = db_manager.db if db_manager else None
        db_status = "OK"
        job_count = 0
        job_stats = {}

        try:
            if db:
                # Use asyncio.gather to run queries concurrently
                all_jobs, new_jobs, interested_jobs, applied_jobs, interview_jobs, offer_jobs, rejected_jobs = await asyncio.gather(
                    db.get_all_jobs(user_id),
                    db.get_all_jobs(user_id, status=JobStatus.NEW),
                    db.get_all_jobs(user_id, status=JobStatus.INTERESTED),
                    db.get_all_jobs(user_id, status=JobStatus.APPLIED),
                    db.get_all_jobs(user_id, status=JobStatus.INTERVIEW),
                    db.get_all_jobs(user_id, status=JobStatus.OFFER),
                    db.get_all_jobs(user_id, status=JobStatus.REJECTED)
                )

                job_count = len(all_jobs)
                job_stats = {
                    "total": job_count,
                    "new": len(new_jobs),
                    "interested": len(interested_jobs),
                    "applied": len(applied_jobs),
                    "interviews": len(interview_jobs),
                    "offers": len(offer_jobs),
                    "rejected": len(rejected_jobs)
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

        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "components": {
                "database": db_status
            },
            "job_stats": job_stats
        }
    except Exception as e:
        logger.error(f"Error getting system status for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Job analysis endpoint
@app.post("/api/jobs/analyze", tags=["Jobs"])
async def analyze_job(
        job_url: str = Form(...),
        status: Optional[str] = Form(None),
        resume_id: Optional[str] = Form(None),
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id),
        api_key: str = Depends(get_user_key)
):
    """Analyze a job posting from a URL."""
    try:
        job_exist = await db_manager.job_exists(url=job_url,user_id=user_id)
        if job_exist:
            raise HTTPException(status_code=409, detail="Job already exists!!")
        # Initialize ResumeImprover with the job URL
        resume_improver = ResumeImprover(url=job_url, api_key=api_key)
        resume_improver.download_and_parse_job_post()
        # The job is already parsed during initialization of ResumeImprover
        job_details = resume_improver.parsed_job

        # Create a job ID
        job_id = hashlib.md5(f"{user_id}|{job_url}|{datetime.now().isoformat()}".encode()).hexdigest()

        # Handle status if provided
        job_status = JobStatus.NEW
        if status:
            try:
                job_status = JobStatus(status)
            except ValueError:
                logger.warning(f"Invalid status '{status}' provided, using default NEW")

        job = Job(
            id=job_id,
            job_url=job_url,
            status=JobStatus.NEW,
            date_found=datetime.now(),
            metadata= job_details
        )

        # Associate the job with the user
        job.metadata["user_id"] = user_id
        await db_manager.db.save_job(job, user_id)

        return {
            "message": "Job analyzed successfully",
            "job_id": job_id,
            "job_url": job_url,
            "user_id": user_id,
            "job_details": job.to_dict()
        }

    except Exception as e:
        logger.error(f"Error analyzing job for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get jobs endpoint
@app.get("/api/jobs", tags=["Jobs"])
async def get_jobs(
        status: Optional[str] = None,
        company: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
):
    """Get jobs from the database with optional filtering."""
    try:
        db = db_manager.db

        # Get all jobs for this user
        jobs = await db.get_all_jobs(user_id, status=status if status else None)

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
            "user_id": user_id,
            "jobs": jobs_dict
        }

    except Exception as e:
        logger.error(f"Error getting jobs for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/cache/clear", tags=["System"])
async def clear_cache(
        user_id: str = Depends(get_user_id)
):
    """Clear user's cache data."""
    try:
        # Clear job cache
        if hasattr(app.state, 'job_cache'):
            app.state.job_cache.clear(user_id)

        # Clear search cache
        if hasattr(app.state, 'search_cache'):
            app.state.search_cache.clear(user_id)

        return {
            "message": "Cache cleared successfully",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add endpoint to get active resume generations
@app.get("/api/resume/active", tags=["Resume"])
async def get_active_resume_generations(
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
):
    """Get all active resume generations for a user."""
    try:
        # This would require modifying the cache to support listing all entries for a user
        # For now, return a simple response
        return {
            "message": "Active resume generations tracking",
            "user_id": user_id,
            "note": "Individual resume status should be checked using /api/resume/{resume_id}/status"
        }
    except Exception as e:
        logger.error(f"Error getting active resume generations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get specific job endpoint
@app.get("/api/jobs/{job_id}", tags=["Jobs"])
async def get_job(
        job_id: str,
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
):
    """Get a specific job by ID."""
    try:
        db = db_manager.db
        job = await db.get_job(job_id, user_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found with ID: {job_id} for user: {user_id}")

        return job.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job {job_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update job status endpoint
@app.put("/api/jobs/{job_id}/status", tags=["Jobs"])
async def update_job_status(
        job_id: str,
        request: JobStatusUpdateRequest,
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
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

        success = await db.update_job_status(job_id, user_id, status_enum)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to update job status. Job may not exist: {job_id} for user: {user_id}"
            )

        # Get the updated job
        job = await db.get_job(job_id, user_id)

        return {
            "message": f"Job status updated to {status}",
            "user_id": user_id,
            "job": job.to_dict() if job else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job status for {job_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/resume/generate", tags=["Resume"])
async def generate_resume(
        request: GenerateResumeRequest,
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id),
        api_key: str = Depends(get_user_key)
):
    """Generate a tailored resume for a specific job using the provided resume data."""
    try:
        # Initialize resume generator
        resume_generator = ResumeGenerator(db_manager, user_id, api_key)

        # Start the resume generation process with the provided resume data
        # This is now non-blocking and uses cache for status tracking
        resume_info = await resume_generator.generate_resume(
            job_id=request.job_id,
            template=request.template or "standard",
            customize=request.customize,
            resume_data=request.resume_data
        )

        return resume_info

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating resume for job {request.job_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/resume/{resume_id}/download", tags=["Resume"])
async def download_resume(
        resume_id: str,
        format: str = Query("yaml", regex="^(yaml)$"),
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
):
    """Download a generated resume in YAML format for client-side rendering."""
    try:
        # Initialize resume generator
        resume_generator = ResumeGenerator(db_manager, user_id, "")

        # Get resume content (checks cache first, then database)
        yaml_content = await resume_generator.get_resume_content(resume_id)

        # Return the YAML content directly
        return {
            "content": yaml_content,
            "format": format,
            "resume_id": resume_id
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error downloading resume {resume_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/resume/{resume_id}/status", tags=["Resume"])
async def check_resume_status(
        resume_id: str,
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
):
    """Check the status of a resume generation process using cache."""
    try:
        # Initialize resume generator
        resume_generator = ResumeGenerator(db_manager, user_id, "")

        # Check resume status (uses cache first, much faster)
        return await resume_generator.check_resume_status(resume_id)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking resume status for {resume_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/resume/upload", tags=["Resume"])
async def upload_resume(
        file: UploadFile = File(...),
        job_id: str = Form(None),
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
):
    """Upload a custom resume."""
    try:
        # Initialize resume generator
        resume_generator = ResumeGenerator(db_manager, user_id,"")

        # Read file content
        content = await file.read()

        # Upload the resume
        return await resume_generator.upload_resume(
            file_path=file.filename,
            file_content=content,
            job_id=job_id
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading resume for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Upload to Simplify endpoint (placeholder, to be implemented later)
@app.post("/api/resume/upload-to-simplify", tags=["Resume"])
async def upload_to_simplify(
        request: UploadToSimplifyRequest,
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
):
    """Upload a resume to Simplify.jobs."""
    # This is a placeholder - you'll implement the actual Simplify integration later
    try:
        db = db_manager.db

        # Check if job exists
        job = await db.get_job(request.job_id, user_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found with ID: {request.job_id} for user: {user_id}")

        # Check if resume exists
        resume_id = request.resume_id or job.resume_id
        if not resume_id:
            raise HTTPException(status_code=400, detail="No resume ID provided or associated with this job")

        resume = await db.get_resume(resume_id, user_id)
        if not resume:
            raise HTTPException(status_code=404, detail=f"Resume not found with ID: {resume_id} for user: {user_id}")

        # Placeholder response
        return {
            "message": "Simplify.jobs upload functionality will be implemented in a future update",
            "job_id": request.job_id,
            "resume_id": resume_id,
            "user_id": user_id,
            "job": job.to_dict(),
            "resume": resume.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading resume to Simplify for job {request.job_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/resume/{resume_id}/update-yaml", tags=["Resume"])
async def update_resume_yaml(
        resume_id: str,
        yaml_content: str = Form(...),
        db_manager: DBCacheManager = Depends(get_db_manager),
        user_id: str = Depends(get_user_id)
):
    """Update the YAML content of a resume."""
    try:
        # Get the existing resume
        resume = await db_manager.db.get_resume(resume_id, user_id)
        if not resume:
            raise HTTPException(status_code=404, detail=f"Resume not found with ID: {resume_id} for user: {user_id}")

        # Update the YAML content
        resume.yaml_content = yaml_content

        # Save the updated resume
        success = await db_manager.db.save_resume(resume, user_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save updated resume YAML")

        return {
            "message": "Resume YAML updated successfully",
            "resume_id": resume_id,
            "user_id": user_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating resume YAML for {resume_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Shutdown event - clean up resources
@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when the application shuts down."""
    try:
        logger.info("Successfully cleaned up resources on shutdown")
    except Exception as e:
        logger.error(f"Error cleaning up resources: {e}")

initialize()
if __name__ == "__main__":
    # Initialize application
    if not initialize():
        logger.error("Failed to initialize application. Exiting.")
        exit(1)

    # Run the FastAPI app with Uvicorn
    host = os.environ.get('API_HOST', '0.0.0.0')
    port = int(os.environ.get('API_PORT', 80))
    debug = os.environ.get('API_DEBUG', '').lower() in ('true', '1', 'yes')

    logger.info(f"Starting API server on {host}:{port} (debug={debug})")
    uvicorn.run(app, host=host, port=port, log_level="debug" if debug else "info")