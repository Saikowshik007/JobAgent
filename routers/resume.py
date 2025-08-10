"""
Resume management routes for generating, uploading, and managing resumes.
"""
from fastapi import APIRouter, Depends, HTTPException, Form, Query, File, UploadFile
from typing import Optional
import logging

from core.dependencies import get_cache_manager, get_user_id, get_user_key, get_user
from data.dbcache_manager import DBCacheManager
from dataModels.api_models import GenerateResumeRequest
from dataModels.user_model import User
from services.resume_generator import ResumeGenerator

logger = logging.getLogger(__name__)
router = APIRouter()



@router.post("/generate")
async def generate_resume(
        request: GenerateResumeRequest,
        handle_existing: str = Query("replace", regex="^(replace|keep_both|error)$",
                                     description="How to handle existing resumes: replace, keep_both, or error"),
        cache_manager: DBCacheManager = Depends(get_cache_manager),
):
    """Generate a tailored resume with orphaning prevention."""
    try:
        # Use include_objective from the request body, not from query parameter
        include_objective = request.include_objective
        if include_objective is None:
            include_objective = True  # Default to True if not specified

        logger.info(f"Resume generation request for job {request.job_id}, include_objective={include_objective}")

        resume_generator = ResumeGenerator(cache_manager, request.user)
        resume_info = await resume_generator.generate_resume(
            job_id=request.job_id,
            template=request.template or "standard",
            customize=request.customize,
            resume_data=request.resume_data,
            handle_existing=handle_existing,
            include_objective=include_objective  # Pass the value from request body
        )

        return resume_info

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating safe resume for job {request.job_id} for user {request.user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{resume_id}/download")
async def download_resume(
        resume_id: str,
        format: str = Query("yaml", regex="^(yaml)$"),
        force_refresh: bool = Query(False, description="Force refresh from database"),
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Download a generated resume in YAML format for client-side rendering."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager, user_id)

        # If force_refresh is True, bypass cache and get directly from database
        if force_refresh:
            # Get resume directly from database
            if cache_manager.db:
                resume = await cache_manager.db.get_resume(resume_id, user_id)
                if not resume:
                    raise ValueError(f"Resume not found with ID: {resume_id} for user: {user_id}")

                # Update cache with fresh data
                if cache_manager.cache:
                    cache_manager.cache.add_resume(resume, user_id)

                yaml_content = resume.yaml_content
            else:
                # Fallback to cache manager method
                yaml_content = await resume_generator.get_resume_content(resume_id)
        else:
            # Normal flow - uses cache first, then database
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

@router.get("/{resume_id}/status")
async def check_resume_status(
        resume_id: str,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Check the status of a resume generation process using cache."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager, user_id)

        # Check resume status (uses cache first, much faster)
        return await resume_generator.check_resume_status(resume_id)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking resume status for {resume_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_resume(
        file: UploadFile = File(...),
        job_id: str = Form(None),
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Upload a custom resume."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager, user_id)

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

@router.post("/{resume_id}/update-yaml")
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

        resume.yaml_content = yaml_content

        success = await cache_manager.save_resume(resume, user_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save updated resume YAML")

        await cache_manager.remove_resume_status(resume_id, user_id)

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

@router.delete("/{resume_id}")
async def delete_resume(
        resume_id: str,
        update_job: bool = Query(True, description="Update associated job to remove resume_id reference"),
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Delete a resume and optionally update the associated job."""
    try:
        # First check if resume exists and get its details
        resume = await cache_manager.get_resume(resume_id, user_id)
        if not resume:
            raise HTTPException(status_code=404, detail=f"Resume not found with ID: {resume_id} for user: {user_id}")

        job_id = resume.job_id

        # Delete the resume
        success = await cache_manager.delete_resume(resume_id, user_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete resume")

        # Update associated job if requested and job exists
        job_updated = False
        if update_job and job_id:
            try:
                # Clear the resume_id from the job
                job_update_success = await cache_manager.update_job_resume_id(job_id, user_id, None)
                if job_update_success:
                    job_updated = True
                    logger.info(f"Cleared resume_id from job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to update job {job_id} after resume deletion: {e}")

        # Also remove from resume generation cache
        await cache_manager.remove_resume_status(resume_id, user_id)

        return {
            "message": "Resume deleted successfully",
            "resume_id": resume_id,
            "user_id": user_id,
            "associated_job_id": job_id,
            "job_updated": job_updated
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting resume {resume_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def get_user_resumes(
        job_id: Optional[str] = Query(None, description="Filter by job ID"),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get all resumes for a user with optional filtering."""
    try:
        resumes = await cache_manager.get_all_resumes(user_id, job_id, limit, offset)

        # Convert to dict format
        resumes_dict = [resume.to_dict() for resume in resumes]

        return {
            "user_id": user_id,
            "job_id_filter": job_id,
            "count": len(resumes_dict),
            "limit": limit,
            "offset": offset,
            "resumes": resumes_dict
        }

    except Exception as e:
        logger.error(f"Error getting resumes for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active")
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


