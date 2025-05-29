import asyncio
import concurrent
import uuid
import yaml
from datetime import datetime
from typing import Optional, Dict, Any, List
from yaml import YAMLError

import config
from data.cache import ResumeGenerationStatus
from dataModels.data_models import JobStatus, Resume
from services.resume_improver import ResumeImprover

logger = config.getLogger("Resume Generator")

class ResumeGenerator:
    """
    Resume Generator using unified cache manager for all operations.
    Simplified - just manages async jobs, ResumeImprover does the work.
    """

    def __init__(self, cache_manager, user_id: str, api_key: str):
        """Initialize the ResumeGenerator with unified cache manager and user ID."""
        self.cache_manager = cache_manager
        self.user_id = user_id
        self.api_key = api_key
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def generate_resume(self, job_id: str, template: str = "standard",
                              customize: bool = True, resume_data: Optional[Dict[str, Any]] = None,
                              handle_existing: str = "replace") -> Dict[str, Any]:
        """
        Generate a tailored resume for a specific job with orphaning prevention.

        Args:
            job_id: Job ID to generate resume for
            template: Resume template to use
            customize: Whether to customize resume for the job
            resume_data: User's resume data
            handle_existing: How to handle existing resumes - "replace", "keep_both", "error"
        """
        # Get the job from database
        job_dict = await self.cache_manager.get_job(job_id, self.user_id)
        if not job_dict:
            raise ValueError(f"Job not found with ID: {job_id} for user: {self.user_id}")

        # Check for existing resumes linked to this job
        existing_resumes = await self.cache_manager.get_resumes_for_job(job_id, self.user_id)

        if existing_resumes and handle_existing == "error":
            raise ValueError(f"Job {job_id} already has {len(existing_resumes)} resume(s). Use handle_existing='replace' or 'keep_both' to proceed.")

        # Generate a unique ID for the new resume
        resume_id = str(uuid.uuid4())

        # Set initial status in cache
        await self.cache_manager.set_resume_status(
            resume_id, self.user_id, ResumeGenerationStatus.PENDING
        )

        # Start background generation (non-blocking)
        asyncio.create_task(self._generate_resume_background(
            job_dict, resume_id, template, customize, resume_data, existing_resumes, handle_existing
        ))

        return {
            "status": "generating",
            "message": f"Resume generation started for job {job_dict.get('job_title', 'Unknown')} at {job_dict.get('company', 'Unknown')}",
            "resume_id": resume_id,
            "job_id": job_id,
            "user_id": self.user_id,
            "template": template,
            "existing_resumes_count": len(existing_resumes),
            "handle_existing": handle_existing,
            "estimated_completion_seconds": 60
        }
    async def _generate_resume_background(self, job_dict: dict, resume_id: str, template: str,
                                          customize: bool, resume_data: Optional[Dict[str, Any]],
                                          existing_resumes: List = None, handle_existing: str = "replace"):
        """Background task to generate resume with orphaning prevention."""
        try:
            logger.info(f"Starting resume generation for job {job_dict.get('id')} for user {self.user_id}")

            # Update status to in progress
            await self.cache_manager.set_resume_status(
                resume_id, self.user_id, ResumeGenerationStatus.IN_PROGRESS
            )

            # Run the blocking resume generation in thread pool
            loop = asyncio.get_event_loop()
            yaml_content = await loop.run_in_executor(
                self.thread_pool,
                self._generate_resume_sync,
                job_dict, resume_data
            )

            # Create the resume object
            resume = Resume(
                id=resume_id,
                job_id=job_dict.get('id'),
                file_path="",
                yaml_content=yaml_content,
                date_created=datetime.now(),
                uploaded_to_simplify=False
            )

            # Save completed resume to database
            await self.cache_manager.save_resume(resume, self.user_id)

            # Handle existing resumes BEFORE updating the job
            if existing_resumes:
                if handle_existing == "replace":
                    logger.info(f"Replacing {len(existing_resumes)} existing resume(s) for job {job_dict.get('id')}")

                    # Delete old resumes (but don't update the job yet since we're about to set the new one)
                    for old_resume in existing_resumes:
                        try:
                            success = await self.cache_manager.delete_resume(old_resume.id, self.user_id)
                            if success:
                                logger.info(f"Deleted old resume {old_resume.id} for job {job_dict.get('id')}")
                                # Remove from generation cache too
                                await self.cache_manager.remove_resume_status(old_resume.id, self.user_id)
                            else:
                                logger.warning(f"Failed to delete old resume {old_resume.id}")
                        except Exception as e:
                            logger.error(f"Error deleting old resume {old_resume.id}: {e}")

                elif handle_existing == "keep_both":
                    logger.info(f"Keeping {len(existing_resumes)} existing resume(s) alongside new resume for job {job_dict.get('id')}")
                    # Don't delete anything, just add the new resume
                    # Note: Only the newest resume will be linked to the job

            # Update the job with the NEW resume_id (this is always done, regardless of handle_existing)
            await self._update_job_with_resume_id(job_dict.get('id'), resume_id)

            # Update cache with completed status
            await self.cache_manager.set_resume_status(
                resume_id, self.user_id, ResumeGenerationStatus.COMPLETED,
                data={"yaml_content": yaml_content}
            )

            # Update job status to RESUME_GENERATED after everything is complete
            await self.cache_manager.update_job_status(
                job_dict.get('id'), self.user_id, JobStatus.RESUME_GENERATED
            )

            logger.info(f"Resume generation completed for job {job_dict.get('id')} for user {self.user_id}")

        except Exception as e:
            logger.error(f"Error generating resume for job {job_dict.get('id')} for user {self.user_id}: {e}")

            # Update cache with failed status
            await self.cache_manager.set_resume_status(
                resume_id, self.user_id, ResumeGenerationStatus.FAILED,
                error=str(e)
            )
    async def _update_job_with_resume_id(self, job_id: str, resume_id: str):
        """Update the job record with the generated resume ID."""
        try:
            # Use the new cache manager method to update job's resume_id
            success = await self.cache_manager.update_job_resume_id(job_id, self.user_id, resume_id)

            if success:
                logger.info(f"Successfully updated job {job_id} with resume_id {resume_id}")
            else:
                logger.error(f"Failed to update job {job_id} with resume_id {resume_id}")

        except Exception as e:
            logger.error(f"Error updating job {job_id} with resume_id {resume_id}: {e}")

    def _generate_resume_sync(self, job_dict: dict, resume_data: Optional[Dict[str, Any]]) -> str:
        """Synchronous resume generation that runs in thread pool."""
        try:
            job_url = job_dict.get('job_url')
            parsed_job = job_dict.get("metadata")
            if not job_url:
                raise ValueError("Job URL not found in job data")

            # Initialize ResumeImprover - it does ALL the work
            resume_improver = ResumeImprover(
                url=job_url,
                parsed_job=parsed_job,
                api_key=self.api_key
            )

            # Set up resume data if provided
            if resume_data:
                logger.info(f"Using user-provided resume data for job {job_dict.get('id')}")
                self._setup_resume_data(resume_improver, resume_data)
            else:
                logger.info(f"Using default resume template for job {job_dict.get('id')}")
                # You'll need to provide default data or handle this case

            # Let ResumeImprover do all the work
            return resume_improver.create_complete_tailored_resume()

        except Exception as e:
            logger.error(f"Synchronous resume generation failed: {e}")
            raise

    def _setup_resume_data(self, resume_improver: ResumeImprover, resume_data: Dict[str, Any]):
        """Set up resume data in the improver"""
        resume_improver.resume = resume_data
        resume_improver.basic_info = self.get_dict_field("basic", resume_data)
        resume_improver.education = self.get_dict_field("education", resume_data)
        resume_improver.experiences = self.get_dict_field("experiences", resume_data)
        resume_improver.projects = self.get_dict_field("projects", resume_data)
        resume_improver.skills = self.get_dict_field("skills", resume_data)
        resume_improver.objective = self.get_dict_field("objective", resume_data)
        resume_improver.degrees = resume_improver._get_degrees(resume_data)

    async def check_resume_status(self, resume_id: str) -> Dict[str, Any]:
        """Check the status of a resume generation process."""
        # First check cache
        cache_entry = await self.cache_manager.get_resume_status(resume_id, self.user_id)

        if cache_entry:
            status = cache_entry["status"].value

            response = {
                "status": status,
                "resume_id": resume_id,
                "user_id": self.user_id,
                "updated_at": datetime.fromtimestamp(cache_entry["updated_at"]).isoformat(),
            }

            if cache_entry.get("error"):
                response["error"] = cache_entry["error"]

            if status == ResumeGenerationStatus.COMPLETED.value:
                # If completed, also get job info from database
                try:
                    resume = await self.cache_manager.get_resume(resume_id, self.user_id)
                    if resume and resume.job_id:
                        job_dict = await self.cache_manager.get_job(resume.job_id, self.user_id)
                        if job_dict:
                            response["job"] = job_dict
                            response["job_id"] = resume.job_id
                except Exception as e:
                    logger.warning(f"Could not fetch job info for completed resume: {e}")

            return response

        # If not in cache, check if it exists in database
        resume = await self.cache_manager.get_resume(resume_id, self.user_id)
        if not resume:
            raise ValueError(f"Resume not found with ID: {resume_id} for user: {self.user_id}")

        # If exists in database, it's completed
        job_dict = None
        if resume.job_id:
            job_dict = await self.cache_manager.get_job(resume.job_id, self.user_id)

        return {
            "status": "completed",
            "resume_id": resume_id,
            "job_id": resume.job_id,
            "user_id": self.user_id,
            "date_created": resume.date_created.isoformat() if resume.date_created else None,
            "job": job_dict
        }

    async def get_resume_content(self, resume_id: str) -> str:
        """Get the resume content."""
        # First check cache for recently generated resumes
        cache_entry = await self.cache_manager.get_resume_status(resume_id, self.user_id)

        if cache_entry and cache_entry["status"] == ResumeGenerationStatus.COMPLETED:
            yaml_content = cache_entry.get("data", {}).get("yaml_content")
            if yaml_content:
                return yaml_content

        # If not in cache, check database
        resume = await self.cache_manager.get_resume(resume_id, self.user_id)
        if not resume:
            raise ValueError(f"Resume not found with ID: {resume_id} for user: {self.user_id}")

        if not resume.yaml_content:
            raise ValueError("Resume generation is not complete")

        return resume.yaml_content

    async def upload_resume(self, file_path: str, file_content: bytes, job_id: str = None) -> Dict[str, Any]:
        """Upload a custom resume."""
        try:
            # Generate a unique ID for the resume
            resume_id = str(uuid.uuid4())

            # Convert file content to string (assuming it's YAML/text)
            yaml_content = file_content.decode('utf-8')

            # Create the resume object
            resume = Resume(
                id=resume_id,
                job_id=job_id,
                file_path=file_path,
                yaml_content=yaml_content,
                date_created=datetime.now(),
                uploaded_to_simplify=False
            )

            # Save resume to database
            success = await self.cache_manager.save_resume(resume, self.user_id)

            if not success:
                raise ValueError("Failed to save uploaded resume")

            # If this resume is for a specific job, update the job's resume_id
            if job_id:
                await self._update_job_with_resume_id(job_id, resume_id)

            return {
                "message": "Resume uploaded successfully",
                "resume_id": resume_id,
                "job_id": job_id,
                "user_id": self.user_id,
                "file_path": file_path
            }

        except UnicodeDecodeError:
            raise ValueError("Invalid file format. Please upload a text-based resume file.")
        except Exception as e:
            logger.error(f"Error uploading resume for user {self.user_id}: {e}")
            raise

    def get_dict_field(self, field: str, data_dict: dict) -> Optional[dict]:
        """Retrieves a field from a dictionary."""
        try:
            return data_dict[field]
        except KeyError as e:
            logger.warning(f"`{field}` is missing in raw resume.")
        return None

    def dict_to_yaml_string(self, data: dict) -> str:
        """Converts a dictionary to a YAML-formatted string."""
        yaml.allow_unicode = True
        try:
            from io import StringIO
            stream = StringIO()
            yaml.dump(data, stream=stream, default_flow_style=False, allow_unicode=True)
            return stream.getvalue()
        except YAMLError as e:
            logger.error("Failed to convert dict to YAML string.")
            raise e

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

    async def replace_job_resume(self, job_id: str, new_resume_id: str) -> Dict[str, Any]:
        """Replace the current resume for a job with a new one, handling orphaning properly."""
        try:
            # Verify the job exists
            job_dict = await self.cache_manager.get_job(job_id, self.user_id)
            if not job_dict:
                raise ValueError(f"Job not found with ID: {job_id} for user: {self.user_id}")

            # Verify the new resume exists and belongs to this user
            new_resume = await self.cache_manager.get_resume(new_resume_id, self.user_id)
            if not new_resume:
                raise ValueError(f"Resume not found with ID: {new_resume_id} for user: {self.user_id}")

            # Get current resume(s) for this job
            current_resumes = await self.cache_manager.get_resumes_for_job(job_id, self.user_id)

            # Update the job to point to the new resume
            success = await self.cache_manager.update_job_resume_id(job_id, self.user_id, new_resume_id)

            if not success:
                raise ValueError("Failed to update job with new resume ID")

            # Update the new resume to point to this job (if it wasn't already)
            if new_resume.job_id != job_id:
                new_resume.job_id = job_id
                await self.cache_manager.save_resume(new_resume, self.user_id)

            # Optionally orphan the old resumes (don't delete them, just clear their job_id)
            orphaned_resumes = []
            for old_resume in current_resumes:
                if old_resume.id != new_resume_id:  # Don't orphan the new resume we just linked
                    old_resume.job_id = None  # Orphan it
                    await self.cache_manager.save_resume(old_resume, self.user_id)
                    orphaned_resumes.append(old_resume.id)

            return {
                "message": f"Successfully replaced resume for job {job_id}",
                "job_id": job_id,
                "new_resume_id": new_resume_id,
                "orphaned_resumes": orphaned_resumes,
                "user_id": self.user_id
            }

        except Exception as e:
            logger.error(f"Error replacing job resume for job {job_id} for user {self.user_id}: {e}")
            raise

    async def cleanup_orphaned_resumes(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Find orphaned resumes (resumes not linked to any job)."""
        target_user_id = user_id or self.user_id

        try:
            # Get all resumes for the user
            all_resumes = await self.cache_manager.get_all_resumes(target_user_id)

            # Find orphaned resumes (job_id is None or points to non-existent job)
            orphaned_resumes = []
            for resume in all_resumes:
                if not resume.job_id:
                    orphaned_resumes.append({
                        "resume_id": resume.id,
                        "reason": "no_job_id",
                        "date_created": resume.date_created.isoformat() if resume.date_created else None
                    })
                else:
                    # Check if the job still exists
                    job = await self.cache_manager.get_job(resume.job_id, target_user_id)
                    if not job:
                        orphaned_resumes.append({
                            "resume_id": resume.id,
                            "reason": "job_not_found",
                            "missing_job_id": resume.job_id,
                            "date_created": resume.date_created.isoformat() if resume.date_created else None
                        })

            return {
                "user_id": target_user_id,
                "total_resumes": len(all_resumes),
                "orphaned_count": len(orphaned_resumes),
                "orphaned_resumes": orphaned_resumes
            }

        except Exception as e:
            logger.error(f"Error finding orphaned resumes for user {target_user_id}: {e}")
            raise