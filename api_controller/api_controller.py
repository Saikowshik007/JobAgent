import hashlib
import uuid

import requests
from fastapi import FastAPI, HTTPException, Depends, Query, Header, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import logging
import os
from datetime import datetime
import uvicorn
from dotenv import load_dotenv
from starlette.responses import HTMLResponse

import config
from dataModels.api_models import JobStatusUpdateRequest, GenerateResumeRequest, \
    UploadToSimplifyRequest
from services.resume_generator import ResumeGenerator
from services.resume_improver import ResumeImprover  # Import the ResumeImprover class

# Load environment variables from .env file if it exists
load_dotenv()

from data.database import Database
from dataModels.data_models import JobStatus, Job
from data.dbcache_manager import DBCacheManager
from data.cache import JobCache, ResumeCache

# Configure logging
logging.basicConfig(
    level=config.get("api.level"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
user_sessions = {}

# FastAPI app instance
app = FastAPI(
    title="JobTrak API",
    description="API for analyzing job postings and tracking job applications",
    version="1.0.0"
)

# Get allowed origins dynamically
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

# CORS configuration - NO WILDCARDS when using credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
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

        # Additional headers that might be needed
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

# Add explicit OPTIONS handler for all routes
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle all OPTIONS requests explicitly"""
    return {"message": "OK"}

# Default user ID if none provided
DEFAULT_USER_ID = "default_user"

# Initialize function to set up application state
async def initialize(db_url=None, job_cache_size=None, search_cache_size=None):
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
        # await db.initialize_db()
        logger.info(f"Initialized database with URL {db_url}")

        # Initialize caches
        job_cache = JobCache(max_size=job_cache_size)
        resume_cache = ResumeCache()
        logger.info(f"Initialized caches with sizes - job: {job_cache_size}, search: {search_cache_size}")

        # Initialize unified cache manager (no more separate cache managers!)
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
        app.state.cache_manager = cache_manager  # Single unified cache manager

        logger.info("Application initialization complete")
        return True
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information including public IP."""
    try:
        import requests
        public_ip = requests.get('https://ipinfo.io/ip', timeout=5).text.strip()
        logger.info(f"🌐 Server starting - Public IP: {public_ip}")
        logger.info(f"🔗 API accessible at: https://{public_ip}/api/")

        # Store in app state for potential use
        app.state.public_ip = public_ip

    except Exception as e:
        logger.warning(f"Could not detect public IP: {e}")

    # Continue with existing initialization
    success = await initialize()
    if not success:
        logger.error("Failed to initialize application during startup")
        raise Exception("Application initialization failed")

# Dependency to get cache manager
def get_cache_manager():
    """Get the unified cache manager instance."""
    if not hasattr(app.state, "cache_manager"):
        # If cache_manager is not in app.state, raise an exception
        raise HTTPException(
            status_code=500,
            detail="Cache manager not initialized. Please check server configuration."
        )

    return app.state.cache_manager

async def get_user_id(x_user_id: Optional[str] = Header(None)):
    """Get user_id from header or use default if not provided."""
    if not x_user_id:
        logger.warning("No user_id provided in header, using default")
        return DEFAULT_USER_ID
    logger.info(f"Received user_id: {x_user_id}")
    return x_user_id

async def get_user_key(x_api_key: Optional[str] = Header(None)):
    """Get API key from header."""
    if not x_api_key:
        logger.warning("No api_key provided in header")
        raise HTTPException(
            status_code=401,
            detail="No API key provided. Please include X-Api-Key header."
        )
    logger.info("Received API key")
    return x_api_key

# System status endpoint
@app.get("/api/status", tags=["System"])
async def get_system_status(
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get the overall status of the job tracking system."""
    try:
        # Get health check from unified cache manager
        health_info = await cache_manager.health_check()

        # Get job statistics
        job_stats = await cache_manager.get_job_stats(user_id)

        # Get cache statistics
        cache_stats = cache_manager.get_cache_stats()

        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "health": health_info,
            "job_stats": job_stats,
            "cache_stats": cache_stats
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
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id),
        api_key: str = Depends(get_user_key)
):
    """Analyze a job posting from a URL."""
    try:
        job_exist = await cache_manager.job_exists(url=job_url, user_id=user_id)
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
            status=job_status,
            date_found=datetime.now(),
            metadata=job_details
        )

        # Save job using the unified cache manager
        await cache_manager.save_job(job, user_id)

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
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get jobs from the database with optional filtering."""
    try:
        # Convert string status to JobStatus enum if provided
        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status)
            except ValueError:
                logger.warning(f"Invalid status filter: {status}")

        # Get jobs using unified cache manager
        jobs = await cache_manager.get_all_jobs(user_id, status=status_filter, limit=limit, offset=offset)

        # Apply additional filters
        if company:
            jobs = [job for job in jobs if job.metadata and company.lower() in job.metadata.get('company', '').lower()]

        # Get total count
        total_count = len(jobs)

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
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Clear user's cache data."""
    try:
        # Clear all cache data using unified cache manager
        await cache_manager.clear_user_cache(user_id)

        return {
            "message": "Cache cleared successfully",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get specific job endpoint
@app.get("/api/jobs/{job_id}", tags=["Jobs"])
async def get_job(
        job_id: str,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get a specific job by ID."""
    try:
        job_dict = await cache_manager.get_job(job_id, user_id)

        if not job_dict:
            raise HTTPException(status_code=404, detail=f"Job not found with ID: {job_id} for user: {user_id}")

        return job_dict

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
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Update the status of a job."""
    try:
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

        success = await cache_manager.update_job_status(job_id, user_id, status_enum)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to update job status. Job may not exist: {job_id} for user: {user_id}"
            )

        # Get the updated job
        job_dict = await cache_manager.get_job(job_id, user_id)

        return {
            "message": f"Job status updated to {status}",
            "user_id": user_id,
            "job": job_dict
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job status for {job_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/resume/generate", tags=["Resume"])
async def generate_resume(
        request: GenerateResumeRequest,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id),
        api_key: str = Depends(get_user_key)
):
    """Generate a tailored resume for a specific job using the provided resume data."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager, user_id, api_key)

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
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Download a generated resume in YAML format for client-side rendering."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager, user_id, "")

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
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Check the status of a resume generation process using cache."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager, user_id, "")

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
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Upload a custom resume."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager, user_id, "")

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
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Upload a resume to Simplify.jobs."""
    # This is a placeholder - you'll implement the actual Simplify integration later
    try:
        # Check if job exists
        job_dict = await cache_manager.get_job(request.job_id, user_id)
        if not job_dict:
            raise HTTPException(status_code=404, detail=f"Job not found with ID: {request.job_id} for user: {user_id}")

        # Check if resume exists
        resume_id = request.resume_id or job_dict.get('resume_id')
        if not resume_id:
            raise HTTPException(status_code=400, detail="No resume ID provided or associated with this job")

        resume = await cache_manager.get_resume(resume_id, user_id)
        if not resume:
            raise HTTPException(status_code=404, detail=f"Resume not found with ID: {resume_id} for user: {user_id}")

        # Placeholder response
        return {
            "message": "Simplify.jobs upload functionality will be implemented in a future update",
            "job_id": request.job_id,
            "resume_id": resume_id,
            "user_id": user_id,
            "job": job_dict,
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
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Update the YAML content of a resume."""
    try:
        # Get the existing resume
        resume = await cache_manager.get_resume(resume_id, user_id)
        if not resume:
            raise HTTPException(status_code=404, detail=f"Resume not found with ID: {resume_id} for user: {user_id}")

        # Update the YAML content
        resume.yaml_content = yaml_content

        # Save the updated resume
        success = await cache_manager.save_resume(resume, user_id)

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

# Add endpoint to get active resume generations
@app.get("/api/resume/active", tags=["Resume"])
async def get_active_resume_generations(
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get all active resume generations for a user."""
    try:
        # Get cache statistics to show activity
        cache_stats = cache_manager.get_cache_stats()

        return {
            "message": "Active resume generations tracking",
            "user_id": user_id,
            "cache_stats": cache_stats.get("resume_cache", {}),
            "note": "Individual resume status should be checked using /api/resume/{resume_id}/status"
        }
    except Exception as e:
        logger.error(f"Error getting active resume generations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoint
@app.post("/api/cache/cleanup", tags=["System"])
async def cleanup_cache(
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Clean up expired cache entries."""
    try:
        await cache_manager.cleanup_expired_cache()

        return {
            "message": "Cache cleanup completed",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache statistics endpoint
@app.get("/api/cache/stats", tags=["System"])
async def get_cache_stats(
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get detailed cache statistics."""
    try:
        stats = cache_manager.get_cache_stats()
        health = await cache_manager.health_check()

        return {
            "cache_stats": stats,
            "health": health,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Shutdown event - clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down."""
    try:
        # Close database pool
        if hasattr(app.state, 'db'):
            await app.state.db.close_pool()
        logger.info("Successfully cleaned up resources on shutdown")
    except Exception as e:
        logger.error(f"Error cleaning up resources: {e}")

@app.post("/api/simplify/store-session")
async def store_simplify_session(
        session_data: dict,
        user_id: str = Depends(get_user_id)
):
    """Store Simplify session data manually provided by user"""
    try:
        # Store the session data
        user_sessions[user_id] = {
            'authorization': session_data.get('authorization'),
            'csrf_token': session_data.get('csrf_token'),
            'raw_cookies': session_data.get('raw_cookies', ''),
            'stored_at': datetime.now(),
            'user_id': user_id
        }

        logger.info(f"Stored Simplify session for user {user_id}")
        return {"message": "Session stored successfully"}

    except Exception as e:
        logger.error(f"Error storing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simplify/upload-resume")
async def upload_resume_to_simplify(
        request_data: dict,
        user_id: str = Depends(get_user_id)
):
    """Upload resume to Simplify using stored session data"""
    try:
        resume_id = request_data.get('resume_id')
        job_id = request_data.get('job_id')

        if not resume_id:
            raise HTTPException(status_code=400, detail="Resume ID is required")

        # Check if user has session data
        if user_id not in user_sessions:
            raise HTTPException(
                status_code=400,
                detail="No Simplify session found. Please complete the session capture step first."
            )

        session = user_sessions[user_id]

        # Get the resume PDF file
        resume_file_path = get_resume_file_path(resume_id)  # You'll need to implement this

        if not os.path.exists(resume_file_path):
            raise HTTPException(status_code=404, detail="Resume file not found")

        # Prepare headers exactly like your curl request
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'origin': 'https://simplify.jobs',
            'referer': 'https://simplify.jobs/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
            'x-csrf-token': session['csrf_token'],
            'dnt': '1',
            'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site'
        }

        # Prepare cookies
        cookies = {
            'authorization': session['authorization'],
            'csrf': session['csrf_token']
        }

        # If user provided raw cookies, parse and add them
        if session.get('raw_cookies'):
            try:
                raw_cookies = session['raw_cookies']
                for cookie in raw_cookies.split(';'):
                    if '=' in cookie:
                        name, value = cookie.strip().split('=', 1)
                        cookies[name] = value
            except Exception as e:
                logger.warning(f"Failed to parse raw cookies: {e}")

        # Prepare the file for upload
        with open(resume_file_path, 'rb') as resume_file:
            files = {
                'file': ('resume.pdf', resume_file, 'application/pdf')
            }

            # Make the request to Simplify's API
            response = requests.post(
                'https://api.simplify.jobs/v2/candidate/me/resume/upload',
                files=files,
                headers=headers,
                cookies=cookies,
                timeout=30
            )

        logger.info(f"Simplify API response: {response.status_code}")

        if response.status_code == 200:
            try:
                response_data = response.json()
                return {
                    "message": "Resume uploaded successfully to Simplify",
                    "data": response_data
                }
            except:
                return {"message": "Resume uploaded successfully to Simplify"}
        else:
            logger.error(f"Simplify upload failed: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Simplify upload failed: {response.text}"
            )

    except Exception as e:
        logger.error(f"Error uploading resume to Simplify: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_resume_file_path(resume_id: str) -> str:
    """Get the file path for a resume PDF"""
    # Implement this based on how you store resume files
    # Example:
    resume_dir = os.getenv('RESUME_STORAGE_PATH', './resumes')
    return os.path.join(resume_dir, f"{resume_id}.pdf")

@app.get("/api/simplify/check-session")
async def check_simplify_session(user_id: str = Depends(get_user_id)):
    """Check if user has a valid Simplify session"""
    has_session = user_id in user_sessions
    session_age = None

    if has_session:
        stored_at = user_sessions[user_id]['stored_at']
        session_age = (datetime.now() - stored_at).total_seconds() / 3600  # hours

    return {
        "has_session": has_session,
        "session_age_hours": session_age
    }

@app.post("/api/simplify/upload-resume-pdf")
async def upload_resume_pdf_to_simplify(
        resume_pdf: UploadFile = File(...),
        resume_id: str = Form(...),
        job_id: str = Form(None),
        user_id: str = Depends(get_user_id)
):
    """Upload a PDF resume to Simplify using stored session data"""
    try:
        # Check if user has session data
        if user_id not in user_sessions:
            raise HTTPException(
                status_code=400,
                detail="No Simplify session found. Please complete the session capture step first."
            )

        session = user_sessions[user_id]

        # Validate the uploaded file
        if not resume_pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read the PDF content
        pdf_content = await resume_pdf.read()

        if len(pdf_content) == 0:
            raise HTTPException(status_code=400, detail="PDF file is empty")

        logger.info(f"Received PDF file: {resume_pdf.filename}, size: {len(pdf_content)} bytes")

        # Prepare headers exactly like the working curl request
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'origin': 'https://simplify.jobs',
            'referer': 'https://simplify.jobs/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
            'x-csrf-token': session['csrf_token'],
            'dnt': '1',
            'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site'
        }

        # Prepare cookies
        cookies = {
            'authorization': session['authorization'],
            'csrf': session['csrf_token']
        }

        # If user provided raw cookies, parse and add them
        if session.get('raw_cookies'):
            try:
                raw_cookies = session['raw_cookies']
                for cookie in raw_cookies.split(';'):
                    if '=' in cookie:
                        name, value = cookie.strip().split('=', 1)
                        cookies[name] = value
            except Exception as e:
                logger.warning(f"Failed to parse raw cookies: {e}")

        # Prepare the file for upload
        files = {
            'file': (resume_pdf.filename, pdf_content, 'application/pdf')
        }

        # Make the request to Simplify's API
        response = requests.post(
            'https://api.simplify.jobs/v2/candidate/me/resume/upload',
            files=files,
            headers=headers,
            cookies=cookies,
            timeout=30
        )

        logger.info(f"Simplify API response: {response.status_code}")

        if response.status_code == 200:
            try:
                response_data = response.json()
                return {
                    "message": "Resume PDF uploaded successfully to Simplify",
                    "data": response_data,
                    "resume_id": resume_id,
                    "job_id": job_id,
                    "pdf_size": len(pdf_content)
                }
            except:
                return {
                    "message": "Resume PDF uploaded successfully to Simplify",
                    "resume_id": resume_id,
                    "job_id": job_id,
                    "pdf_size": len(pdf_content)
                }
        else:
            logger.error(f"Simplify upload failed: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Simplify upload failed: {response.text}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF resume to Simplify: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simplify/auto-capture")
async def auto_capture_tokens(
        request_data: dict,
        user_id: str = Depends(get_user_id)
):
    """Automatically capture tokens from bookmarklet or extension"""
    try:
        # Store the auto-captured session data
        user_sessions[user_id] = {
            'authorization': request_data.get('authorization'),
            'csrf_token': request_data.get('csrf'),
            'raw_cookies': request_data.get('cookies', ''),
            'stored_at': datetime.now(),
            'user_id': user_id,
            'capture_method': 'auto'
        }

        logger.info(f"Auto-captured Simplify session for user {user_id}")
        return {"message": "Tokens captured successfully via automation"}

    except Exception as e:
        logger.error(f"Error auto-capturing tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    host = os.environ.get('API_HOST', '0.0.0.0')
    port = int(os.environ.get('API_PORT', 8000))
    debug = os.environ.get('API_DEBUG', '').lower() in ('true', '1', 'yes')

    logger.info(f"Starting API server on {host}:{port} (debug={debug})")
    uvicorn.run(app, host=host, port=port, log_level="debug" if debug else "info")