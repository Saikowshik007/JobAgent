"""
Resume management routes for generating, uploading, and managing resumes.
"""
from fastapi import APIRouter, Depends, HTTPException, Form, Query, File, UploadFile
from typing import Optional
import logging
import asyncio
import uuid

from core.dependencies import get_cache_manager
from data.dbcache_manager import DBCacheManager
from dataModels.api_models import GenerateResumeRequest
from dataModels.user_model import User
from data.cache import ResumeGenerationStatus
from services.resume_generator import ResumeGenerator
from services.resume_parser import parse_source_resume

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

        resume_generator = ResumeGenerator(cache_manager)
        resume_info = await resume_generator.generate_resume(
            job_id=request.job_id,
            user= request.user,
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

@router.get("/{user_id}/{resume_id}/download")
async def download_resume(
        user_id: str,
        resume_id: str,
        format: str = Query("yaml", regex="^(yaml)$"),
        force_refresh: bool = Query(False, description="Force refresh from database"),
        cache_manager: DBCacheManager = Depends(get_cache_manager)

):
    """Download a generated resume in YAML format for client-side rendering."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager)

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
                yaml_content = await resume_generator.get_resume_content(resume_id, user_id)
        else:
            # Normal flow - uses cache first, then database
            yaml_content = await resume_generator.get_resume_content(resume_id, user_id)

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

@router.get("/{user_id}/{resume_id}/status")
async def check_resume_status(
        user_id: str,
        resume_id: str,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
):
    """Check the status of a resume generation process using cache."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager)

        # Check resume status (uses cache first, much faster)
        return await resume_generator.check_resume_status(resume_id, user_id)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking resume status for {resume_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{user_id}/upload")
async def upload_resume(
        user_id: str,
        file: UploadFile = File(...),
        job_id: str = Form(None),
        cache_manager: DBCacheManager = Depends(get_cache_manager)

):
    """Upload a custom resume."""
    try:
        # Initialize resume generator with unified cache manager
        resume_generator = ResumeGenerator(cache_manager)

        # Read file content
        content = await file.read()

        # Upload the resume
        return await resume_generator.upload_resume(
            file_path=file.filename,
            user_id=user_id,
            file_content=content,
            job_id=job_id
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading resume for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{user_id}/{resume_id}/update-yaml")
async def update_resume_yaml(
        user_id: str,
        resume_id: str,
        yaml_content: str = Form(...),
        cache_manager: DBCacheManager = Depends(get_cache_manager),
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

@router.delete("/{user_id}/{resume_id}")
async def delete_resume(
        user_id: str,
        resume_id: str,
        update_job: bool = Query(True, description="Update associated job to remove resume_id reference"),
        cache_manager: DBCacheManager = Depends(get_cache_manager),

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

@router.get("/{user_id}/")
async def get_user_resumes(
        user_id: str,
        job_id: Optional[str] = Query(None, description="Filter by job ID"),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        cache_manager: DBCacheManager = Depends(get_cache_manager)
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

@router.get("/{user_id}/active")
async def get_active_resume_generations(
        user_id: str,
        cache_manager: DBCacheManager = Depends(get_cache_manager)
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


@router.post("/{user_id}/parse-pdf")
async def parse_resume_pdf(
        user_id: str,
        file: UploadFile = File(...),
        api_key: str = Form(...),
        model: str = Form("gpt-4o"),
        cache_manager: DBCacheManager = Depends(get_cache_manager),
):
    """Start an asynchronous PDF-to-canonical-resume import."""
    if file.content_type not in {"application/pdf", "application/x-pdf"} and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported")
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="An OpenAI API key is required to parse a resume")
    try:
        import_id = str(uuid.uuid4())
        await cache_manager.set_resume_status(
            import_id,
            user_id,
            ResumeGenerationStatus.PENDING,
            data={
                "operation": "resume_import",
                "stage": "queued",
                "progress_percentage": 5,
                "message": "PDF resume import is queued",
            },
        )
        asyncio.create_task(_parse_resume_pdf_background(
            import_id,
            user_id,
            await file.read(),
            User(id=user_id, api_key=api_key.strip(), model=model),
            cache_manager,
        ))
        return {"status": "processing", "import_id": import_id}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.exception("resume_pdf_parse_failed")
        raise HTTPException(status_code=502, detail="Could not parse the resume PDF. Please try again.") from error


async def _parse_resume_pdf_background(import_id: str, user_id: str, pdf_content: bytes, user: User,
                                       cache_manager: DBCacheManager) -> None:
    """Run source-resume extraction without blocking the request loop."""
    try:
        await cache_manager.set_resume_status(
            import_id, user_id, ResumeGenerationStatus.IN_PROGRESS,
            data={"operation": "resume_import", "stage": "extracting", "progress_percentage": 25,
                  "message": "Extracting text from the PDF"},
        )
        event_loop = asyncio.get_running_loop()

        def report_progress(stage: str, progress: int, message: str) -> None:
            future = asyncio.run_coroutine_threadsafe(
                cache_manager.set_resume_status(
                    import_id, user_id, ResumeGenerationStatus.IN_PROGRESS,
                    data={"operation": "resume_import", "stage": stage,
                          "progress_percentage": progress, "message": message},
                ),
                event_loop,
            )
            future.result(timeout=5)

        resume_data = await asyncio.to_thread(parse_source_resume, pdf_content, user, report_progress)
        await cache_manager.set_resume_status(
            import_id, user_id, ResumeGenerationStatus.COMPLETED,
            data={"operation": "resume_import", "stage": "completed", "progress_percentage": 100,
                  "message": "Resume fields are ready for review", "resume_data": resume_data},
        )
        logger.info("resume_pdf_parsed", extra={"resume.source_type": "pdf"})
    except Exception as error:
        logger.exception("resume_pdf_parse_failed")
        await cache_manager.set_resume_status(
            import_id, user_id, ResumeGenerationStatus.FAILED,
            data={"operation": "resume_import", "stage": "failed", "progress_percentage": 0,
                  "message": f"PDF resume import failed: {error}"},
            error=str(error),
        )
