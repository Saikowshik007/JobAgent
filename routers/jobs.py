"""
Job management routes for analyzing, retrieving, updating, and deleting jobs.
"""
from fastapi import APIRouter, Depends, HTTPException, Form, Query
from typing import Optional, List
from datetime import datetime
import hashlib
import logging

from core.dependencies import get_cache_manager, get_user_id, get_user_key
from data.dbcache_manager import DBCacheManager
from dataModels.api_models import JobStatusUpdateRequest
from dataModels.data_models import JobStatus, Job
from services.resume_improver import ResumeImprover

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyze")
async def analyze_job(
        job_url: str = Form(...),
        status: Optional[str] = Form(None),
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

@router.get("/")
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
    
@router.get("/status")
async def get_job_stats(
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get detailed job statistics including resume information."""
    try:
        # Get basic job stats
        job_stats = await cache_manager.get_job_stats(user_id)

        # Get all jobs to analyze resume relationships
        all_jobs = await cache_manager.get_all_jobs(user_id)

        # Analyze resume linkage
        jobs_with_resumes = 0
        jobs_without_resumes = 0
        total_job_resume_links = 0

        for job in all_jobs:
            if job.resume_id:
                jobs_with_resumes += 1
                total_job_resume_links += 1
            else:
                jobs_without_resumes += 1

        # Get all resumes
        all_resumes = await cache_manager.get_all_resumes(user_id)

        # Analyze orphaned resumes
        orphaned_resumes = 0
        linked_resumes = 0

        for resume in all_resumes:
            if not resume.job_id:
                orphaned_resumes += 1
            else:
                # Verify the job still exists
                job = await cache_manager.get_job(resume.job_id, user_id)
                if job:
                    linked_resumes += 1
                else:
                    orphaned_resumes += 1  # Job was deleted but resume still references it

        return {
            "user_id": user_id,
            "job_stats": job_stats,
            "resume_stats": {
                "total_resumes": len(all_resumes),
                "linked_resumes": linked_resumes,
                "orphaned_resumes": orphaned_resumes,
                "jobs_with_resumes": jobs_with_resumes,
                "jobs_without_resumes": jobs_without_resumes,
                "total_job_resume_links": total_job_resume_links
            },
            "health_ratios": {
                "jobs_with_resumes_ratio": round(jobs_with_resumes / len(all_jobs) * 100, 2) if all_jobs else 0,
                "orphaned_resume_ratio": round(orphaned_resumes / len(all_resumes) * 100, 2) if all_resumes else 0
            }
        }

    except Exception as e:
        logger.error(f"Error getting detailed job stats for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}")
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

@router.put("/{job_id}/status")
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

@router.delete("/{job_id}")
async def delete_job(
        job_id: str,
        cascade_resumes: bool = Query(False, description="Also delete associated resumes"),
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Delete a job and optionally its associated resumes."""
    try:
        # First check if job exists
        job_dict = await cache_manager.get_job(job_id, user_id)
        if not job_dict:
            raise HTTPException(status_code=404, detail=f"Job not found with ID: {job_id} for user: {user_id}")

        # Find associated resumes if cascade is requested
        deleted_resumes = []
        if cascade_resumes:
            # Get resumes associated with this job
            associated_resumes = await cache_manager.get_resumes_for_job(job_id, user_id)
            for resume in associated_resumes:
                success = await cache_manager.delete_resume(resume.id, user_id)
                if success:
                    deleted_resumes.append(resume.id)
                    logger.info(f"Deleted associated resume {resume.id} for job {job_id}")

        # Delete the job
        success = await cache_manager.delete_job(job_id, user_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete job")

        return {
            "message": "Job deleted successfully",
            "job_id": job_id,
            "user_id": user_id,
            "deleted_resumes": deleted_resumes if cascade_resumes else [],
            "cascade_resumes": cascade_resumes
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job {job_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/batch")
async def delete_jobs_batch(
        job_ids: List[str],
        cascade_resumes: bool = Query(False, description="Also delete associated resumes"),
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Delete multiple jobs in a batch operation."""
    try:
        if not job_ids:
            raise HTTPException(status_code=400, detail="No job IDs provided")

        if len(job_ids) > 100:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Too many jobs (max 100 per batch)")

        deleted_jobs = []
        deleted_resumes = []
        failed_jobs = []

        for job_id in job_ids:
            try:
                # Check if job exists
                job_dict = await cache_manager.get_job(job_id, user_id)
                if not job_dict:
                    failed_jobs.append({"job_id": job_id, "reason": "not_found"})
                    continue

                # Handle cascading resume deletion
                if cascade_resumes:
                    associated_resumes = await cache_manager.get_resumes_for_job(job_id, user_id)
                    for resume in associated_resumes:
                        success = await cache_manager.delete_resume(resume.id, user_id)
                        if success:
                            deleted_resumes.append(resume.id)

                # Delete the job
                success = await cache_manager.delete_job(job_id, user_id)
                if success:
                    deleted_jobs.append(job_id)
                else:
                    failed_jobs.append({"job_id": job_id, "reason": "delete_failed"})

            except Exception as e:
                logger.error(f"Error deleting job {job_id}: {e}")
                failed_jobs.append({"job_id": job_id, "reason": str(e)})

        return {
            "message": f"Batch delete completed: {len(deleted_jobs)} jobs deleted",
            "deleted_jobs": deleted_jobs,
            "deleted_resumes": deleted_resumes if cascade_resumes else [],
            "failed_jobs": failed_jobs,
            "cascade_resumes": cascade_resumes,
            "user_id": user_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch job deletion for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}/resumes")
async def get_job_resumes(
        job_id: str,
        cache_manager: DBCacheManager = Depends(get_cache_manager),
        user_id: str = Depends(get_user_id)
):
    """Get all resumes associated with a specific job."""
    try:
        # Check if job exists
        job_dict = await cache_manager.get_job(job_id, user_id)
        if not job_dict:
            raise HTTPException(status_code=404, detail=f"Job not found with ID: {job_id} for user: {user_id}")

        # Get resumes for this job
        resumes = await cache_manager.get_resumes_for_job(job_id, user_id)

        # Convert to dict format
        resumes_dict = [resume.to_dict() for resume in resumes]

        return {
            "job_id": job_id,
            "user_id": user_id,
            "resume_count": len(resumes_dict),
            "resumes": resumes_dict
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting resumes for job {job_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
