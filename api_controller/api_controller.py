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
    UploadToSimplifyRequest, SimplifyLoginRequest, SimplifyAPIRequest, SubmitSimplifySessionRequest
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
simplify_sessions: Dict[str, Dict[str, Any]] = {}
user_simplify_sessions: Dict[str, Dict[str, Any]] = {}

# FastAPI app instance
app = FastAPI(
    title="JobTrak API",
    description="API for analyzing job postings and tracking job applications",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.vercel.app",                # All Vercel subdomains
        "https://job-agent-ui.vercel.app",     # Your specific Vercel domain
        "https://localhost:3000",              # Local HTTPS development
        "http://localhost:3000",               # Local HTTP development
        "https://127.0.0.1:3000",             # Local IP HTTPS
        "http://127.0.0.1:3000",              # Local IP HTTP
        # Add your public IP here (replace with your actual IP)
        # "https://YOUR_PUBLIC_IP",
        # "http://YOUR_PUBLIC_IP",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Api-Key",
        "x-api-key",
        "X-User-Id",
        "x-user-id",
        "x_user_id",
        "Cache-Control",
        "DNT",
        "If-Modified-Since",
        "Keep-Alive",
        "Origin",
        "User-Agent",
        "X-Requested-With",
        "Range"
    ],
    expose_headers=["Content-Range", "X-Content-Range"],
)

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

class SimplifyJobsIntegration:
    def __init__(self):
        self.site_key = "6LcStf4UAAAAAIVZo9JUJ3PntTfRBhvXLKBTGww8"
        self.base_url = "https://api.simplify.jobs/v2"

    def create_session_for_user(self, user_id: str, username: str) -> str:
        """Create a new Simplify login session for a user"""
        session_id = str(uuid.uuid4())

        simplify_sessions[session_id] = {
            "user_id": user_id,
            "username": username,
            "status": "pending",
            "created_at": datetime.now(),
            "cookies_received": False
        }

        return session_id

    def get_login_page_html(self, session_id: str, username: str) -> str:
        """Generate the login page HTML for manual authentication"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Connect to Simplify Jobs</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #333;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 16px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #4f46e5;
                    margin-bottom: 10px;
                    font-size: 2rem;
                }}
                .status {{
                    padding: 16px;
                    border-radius: 8px;
                    margin: 20px 0;
                    font-weight: 500;
                }}
                .pending {{ background: #fef3c7; border-left: 4px solid #f59e0b; }}
                .success {{ background: #d1fae5; border-left: 4px solid #10b981; }}
                .error {{ background: #fee2e2; border-left: 4px solid #ef4444; }}
                button {{
                    background: #4f46e5;
                    color: white;
                    border: none;
                    padding: 14px 28px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: 600;
                    margin: 10px 5px;
                    transition: all 0.2s;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                }}
                button:hover {{ background: #4338ca; transform: translateY(-1px); }}
                button:disabled {{ 
                    background: #9ca3af; 
                    cursor: not-allowed; 
                    transform: none;
                }}
                .step {{
                    margin: 25px 0;
                    padding: 20px;
                    background: #f8fafc;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                }}
                .step h3 {{
                    margin-top: 0;
                    color: #1e293b;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .step-number {{
                    background: #4f46e5;
                    color: white;
                    width: 24px;
                    height: 24px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 14px;
                    font-weight: bold;
                }}
                .info-box {{
                    background: #eff6ff;
                    border: 1px solid #bfdbfe;
                    border-radius: 6px;
                    padding: 12px;
                    margin: 15px 0;
                    font-size: 14px;
                }}
                .success-box {{
                    background: #f0fdf4;
                    border: 1px solid #bbf7d0;
                    color: #166534;
                    text-align: center;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .username-display {{
                    background: #f1f5f9;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-family: monospace;
                    display: inline-block;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔗 Connect to Simplify Jobs</h1>
                    <p>Link your Simplify Jobs account to enable automatic job applications</p>
                    <div class="info-box">
                        <strong>Account:</strong> <span class="username-display">{username}</span><br>
                        <strong>Session:</strong> <span class="username-display">{session_id[:8]}...</span>
                    </div>
                </div>
                
                <div id="status" class="status pending">
                    <strong>Status:</strong> <span id="status-text">Ready to connect</span>
                </div>
                
                <div class="step">
                    <h3><span class="step-number">1</span>Login to Simplify Jobs</h3>
                    <p>Click below to open Simplify Jobs and complete your login manually (including CAPTCHA).</p>
                    <button id="loginBtn" onclick="openLoginTab()">
                        🚀 Open Simplify Jobs Login
                    </button>
                </div>
                
                <div class="step">
                    <h3><span class="step-number">2</span>Capture Session</h3>
                    <p>After successful login, return here and click to capture your session cookies.</p>
                    <button id="captureBtn" onclick="captureSession()" disabled>
                        🍪 Capture Session Data
                    </button>
                </div>
                
                <div class="step">
                    <h3><span class="step-number">3</span>Verify Connection</h3>
                    <p>Test the connection to make sure everything works properly.</p>
                    <button id="verifyBtn" onclick="verifySession()" disabled>
                        ✅ Verify & Complete Setup
                    </button>
                </div>
                
                <div id="results"></div>
            </div>

            <script>
                const sessionId = '{session_id}';
                let loginTabReference = null;

                function updateStatus(text, type = 'pending') {{
                    const statusDiv = document.getElementById('status');
                    const statusText = document.getElementById('status-text');
                    statusDiv.className = `status ${{type}}`;
                    statusText.textContent = text;
                }}

                function openLoginTab() {{
                    updateStatus('Opening Simplify Jobs login page...', 'pending');
                    loginTabReference = window.open('https://simplify.jobs/login', '_blank');
                    
                    document.getElementById('captureBtn').disabled = false;
                    document.getElementById('loginBtn').textContent = '🔄 Reopen Login Page';
                    
                    updateStatus('Login page opened. Complete login and return here.', 'pending');
                }}

                async function captureSession() {{
                    updateStatus('Capturing session data...', 'pending');
                    
                    try {{
                        // Get all cookies for simplify.jobs domain
                        const cookies = await new Promise((resolve) => {{
                            chrome.cookies?.getAll({{domain: '.simplify.jobs'}}, resolve) || 
                            // Fallback for non-extension context
                            resolve([]);
                        }});
                        
                        // If we can't get cookies via extension, try document.cookie
                        let cookieData = {{}};
                        if (!cookies || cookies.length === 0) {{
                            // Parse document.cookie if available
                            if (document.cookie) {{
                                document.cookie.split(';').forEach(cookie => {{
                                    const [name, value] = cookie.trim().split('=');
                                    if (name && value) {{
                                        cookieData[name] = value;
                                    }}
                                }});
                            }}
                        }} else {{
                            cookies.forEach(cookie => {{
                                cookieData[cookie.name] = cookie.value;
                            }});
                        }}

                        // Submit session data to backend
                        const response = await fetch(`/api/simplify/submit-session`, {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json',
                            }},
                            body: JSON.stringify({{
                                session_id: sessionId,
                                cookies: cookieData,
                                user_agent: navigator.userAgent
                            }})
                        }});

                        const result = await response.json();
                        
                        if (response.ok) {{
                            updateStatus('Session captured successfully!', 'success');
                            document.getElementById('verifyBtn').disabled = false;
                            document.getElementById('captureBtn').textContent = '✅ Session Captured';
                        }} else {{
                            throw new Error(result.detail || 'Failed to capture session');
                        }}
                    }} catch (error) {{
                        updateStatus(`Error capturing session: ${{error.message}}`, 'error');
                        console.error('Session capture error:', error);
                    }}
                }}

                async function verifySession() {{
                    updateStatus('Verifying connection...', 'pending');
                    
                    try {{
                        const response = await fetch(`/api/simplify/verify-session/${{sessionId}}`, {{
                            method: 'POST'
                        }});

                        const result = await response.json();
                        
                        if (response.ok && result.success) {{
                            updateStatus('Connection verified! You can close this window.', 'success');
                            
                            // Show success message
                            document.getElementById('results').innerHTML = `
                                <div class="success-box">
                                    <h3>🎉 Successfully Connected!</h3>
                                    <p>Your Simplify Jobs account is now linked and ready for automated applications.</p>
                                    <p><strong>You can now close this window and return to your dashboard.</strong></p>
                                </div>
                            `;
                            
                            // Notify parent window
                            if (window.opener) {{
                                window.opener.postMessage({{
                                    type: 'SIMPLIFY_LOGIN_SUCCESS',
                                    sessionId: sessionId
                                }}, '*');
                            }}
                            
                            // Auto-close after delay
                            setTimeout(() => window.close(), 3000);
                        }} else {{
                            throw new Error(result.detail || 'Verification failed');
                        }}
                    }} catch (error) {{
                        updateStatus(`Verification failed: ${{error.message}}`, 'error');
                    }}
                }}

                // Check session status periodically
                setInterval(async () => {{
                    try {{
                        const response = await fetch(`/api/simplify/session-status/${{sessionId}}`);
                        const result = await response.json();
                        
                        if (result.status === 'completed') {{
                            updateStatus('Connection completed successfully!', 'success');
                            document.getElementById('verifyBtn').disabled = false;
                        }}
                    }} catch (error) {{
                        console.error('Status check error:', error);
                    }}
                }}, 5000);
            </script>
        </body>
        </html>
        """

    def submit_session_data(self, session_id: str, cookies: Dict[str, str], user_agent: str = None) -> bool:
        """Store the captured session data"""
        if session_id not in simplify_sessions:
            return False

        session_data = simplify_sessions[session_id]
        session_data.update({
            "cookies": cookies,
            "user_agent": user_agent,
            "status": "cookies_received",
            "updated_at": datetime.now()
        })

        return True

    def verify_session(self, session_id: str) -> Dict[str, Any]:
        """Verify the session works by making a test API call"""
        if session_id not in simplify_sessions:
            return {"success": False, "error": "Session not found"}

        session_data = simplify_sessions[session_id]

        try:
            # Test the session by making a simple API call
            cookies = session_data.get("cookies", {})
            headers = {
                "User-Agent": session_data.get("user_agent", "Mozilla/5.0"),
                "Accept": "application/json",
            }

            # Convert cookies to requests format
            cookie_str = "; ".join([f"{k}={v}" for k, v in cookies.items()])
            if cookie_str:
                headers["Cookie"] = cookie_str

            response = requests.get(
                f"{self.base_url}/profile",
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                # Session is valid, store it for the user
                user_id = session_data["user_id"]
                user_simplify_sessions[user_id] = {
                    "cookies": cookies,
                    "user_agent": session_data.get("user_agent"),
                    "created_at": datetime.now(),
                    "last_verified": datetime.now(),
                    "profile_data": response.json()
                }

                # Mark session as completed
                session_data["status"] = "completed"

                return {"success": True, "profile": response.json()}
            else:
                return {"success": False, "error": f"API call failed: {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def make_api_request(self, user_id: str, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make an API request using the user's stored session"""
        if user_id not in user_simplify_sessions:
            return {"error": "No active Simplify session found"}

        session = user_simplify_sessions[user_id]

        try:
            headers = {
                "User-Agent": session.get("user_agent", "Mozilla/5.0"),
                "Accept": "application/json",
                "Content-Type": "application/json"
            }

            # Add cookies
            cookies = session.get("cookies", {})
            cookie_str = "; ".join([f"{k}={v}" for k, v in cookies.items()])
            if cookie_str:
                headers["Cookie"] = cookie_str

            url = f"{self.base_url}{endpoint}"

            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=data, timeout=30)
            else:
                return {"error": f"Unsupported method: {method}"}

            return {
                "status_code": response.status_code,
                "data": response.json() if response.content else None,
                "headers": dict(response.headers)
            }

        except Exception as e:
            return {"error": str(e)}

# Initialize the integration
simplify_integration = SimplifyJobsIntegration()

@app.post("/api/simplify/initiate-login")
async def initiate_simplify_login(request: SimplifyLoginRequest, user_id: str = Depends(get_user_id)):
    """Initiate the Simplify Jobs login process"""
    try:
        session_id = simplify_integration.create_session_for_user(user_id, request.username)

        return {
            "session_id": session_id,
            "login_url": f"/api/simplify/login-page/{session_id}",
            "message": "Login session created. Complete authentication in the opened window."
        }
    except Exception as e:
        logger.error(f"Error initiating Simplify login: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/simplify/login-page/{session_id}", response_class=HTMLResponse)
async def get_simplify_login_page(session_id: str):
    """Serve the login page for manual authentication"""
    if session_id not in simplify_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = simplify_sessions[session_id]
    username = session_data.get("username", "Unknown")

    html_content = simplify_integration.get_login_page_html(session_id, username)
    return HTMLResponse(content=html_content)

@app.post("/api/simplify/submit-session")
async def submit_simplify_session(request: SubmitSimplifySessionRequest):
    """Submit captured session data"""
    try:
        success = simplify_integration.submit_session_data(
            request.session_id,
            request.cookies,
            request.user_agent
        )

        if success:
            return {"message": "Session data captured successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error submitting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simplify/verify-session/{session_id}")
async def verify_simplify_session(session_id: str):
    """Verify the captured session works"""
    try:
        result = simplify_integration.verify_session(session_id)

        if result["success"]:
            return {
                "success": True,
                "message": "Session verified successfully",
                "profile": result.get("profile")
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
    except Exception as e:
        logger.error(f"Error verifying session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/simplify/session-status/{session_id}")
async def get_session_status(session_id: str):
    """Get the current status of a login session"""
    if session_id not in simplify_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = simplify_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session_data.get("status", "pending"),
        "created_at": session_data.get("created_at"),
        "username": session_data.get("username")
    }

@app.get("/api/simplify/user-session/{user_id}")
async def get_user_simplify_session(user_id: str):
    """Check if user has an active Simplify session"""
    has_session = user_id in user_simplify_sessions

    session_info = None
    if has_session:
        session = user_simplify_sessions[user_id]
        session_info = {
            "created_at": session.get("created_at"),
            "last_verified": session.get("last_verified"),
            "profile": session.get("profile_data", {})
        }

    return {
        "user_id": user_id,
        "has_session": has_session,
        "session_info": session_info
    }

@app.post("/api/simplify/api-request")
async def make_simplify_api_request(request: SimplifyAPIRequest, user_id: str = Depends(get_user_id)):
    """Make an API request to Simplify Jobs using the user's session"""
    try:
        result = simplify_integration.make_api_request(
            user_id,
            request.endpoint,
            request.method,
            request.data
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        logger.error(f"Error making Simplify API request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/simplify/disconnect/{user_id}")
async def disconnect_simplify(user_id: str):
    """Disconnect user's Simplify session"""
    if user_id in user_simplify_sessions:
        del user_simplify_sessions[user_id]
        return {"message": "Simplify session disconnected"}
    else:
        raise HTTPException(status_code=404, detail="No active session found")


if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    host = os.environ.get('API_HOST', '0.0.0.0')
    port = int(os.environ.get('API_PORT', 8000))
    debug = os.environ.get('API_DEBUG', '').lower() in ('true', '1', 'yes')

    logger.info(f"Starting API server on {host}:{port} (debug={debug})")
    uvicorn.run(app, host=host, port=port, log_level="debug" if debug else "info")