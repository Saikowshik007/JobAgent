import asyncio
import uuid
import yaml
from datetime import datetime
from typing import Optional, Dict, Any, List
from yaml import YAMLError

import config
from data.cache import ResumeGenerationStatus
from dataModels.data_models import JobStatus, Resume
from dataModels.user_model import User
from services.resume_improver import ResumeImprover

logger = config.getLogger("Resume Generator")

class ResumeGenerator:
    """
    Resume Generator using unified cache manager for all operations.
    Simplified - just manages async jobs, ResumeImprover does the work.
    """

    def __init__(self, cache_manager):
        """Initialize the ResumeGenerator with unified cache manager and user ID."""
        self.cache_manager = cache_manager

    async def generate_resume(self, job_id: str, user: User, template: str = "standard",
                              customize: bool = True, resume_data: Optional[Dict[str, Any]] = None,
                              handle_existing: str = "replace", include_objective: bool = True) -> Dict[str, Any]:
        """
        Generate a tailored resume for a specific job with orphaning prevention.

        Args:
            job_id: Job ID to generate resume for
            template: Resume template to use
            customize: Whether to customize resume for the job
            resume_data: User's resume data
            handle_existing: How to handle existing resumes - "replace", "keep_both", "error"
            include_objective: Whether to include an objective section in the resume
        """
        # Get the job from database
        job_dict = await self.cache_manager.get_job(job_id, user.id)
        if not job_dict:
            raise ValueError(f"Job not found with ID: {job_id} for user: {user.id}")
        if not isinstance(resume_data, dict) or not resume_data:
            raise ValueError("resume_data is required to generate a factual tailored resume")

        # Check for existing resumes linked to this job
        existing_resumes = await self.cache_manager.get_resumes_for_job(job_id, user.id)

        if existing_resumes and handle_existing == "error":
            raise ValueError(f"Job {job_id} already has {len(existing_resumes)} resume(s). Use handle_existing='replace' or 'keep_both' to proceed.")

        # Generate a unique ID for the new resume
        resume_id = str(uuid.uuid4())

        # Set initial status in cache
        await self.cache_manager.set_resume_status(
            resume_id, user.id, ResumeGenerationStatus.PENDING
        )

        # Start background generation (non-blocking)
        asyncio.create_task(self._generate_resume_background(
            job_dict, resume_id, user, template, customize, resume_data, existing_resumes, handle_existing, include_objective
        ))

        return {
            "status": "generating",
            "message": f"Resume generation started for job {job_dict.get('job_title', 'Unknown')} at {job_dict.get('company', 'Unknown')}",
            "resume_id": resume_id,
            "job_id": job_id,
            "user_id": user.id,
            "template": template,
            "existing_resumes_count": len(existing_resumes),
            "handle_existing": handle_existing,
            "include_objective": include_objective,
            "estimated_completion_seconds": 60
        }
    async def _generate_resume_background(self, job_dict: dict, resume_id: str, user: User, template: str,
                                          customize: bool, resume_data: Optional[Dict[str, Any]],
                                          existing_resumes: List = None, handle_existing: str = "replace",
                                          include_objective: bool = True):
        """Background task to generate resume with orphaning prevention."""
        try:
            logger.info(f"Starting resume generation for job {job_dict.get('id')} for user {user.id} with model : {user.model}")

            # Update status to in progress
            await self.cache_manager.set_resume_status(
                resume_id, user.id, ResumeGenerationStatus.IN_PROGRESS
            )

            # Run the blocking LLM work outside the request event loop.  Using the
            # shared asyncio executor avoids creating one thread pool per request.
            yaml_content = await asyncio.to_thread(
                self._generate_resume_sync,
                job_dict, user, resume_data, customize, include_objective,
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
            await self.cache_manager.save_resume(resume, user.id)

            # Handle existing resumes BEFORE updating the job
            if existing_resumes:
                if handle_existing == "replace":
                    logger.info(f"Replacing {len(existing_resumes)} existing resume(s) for job {job_dict.get('id')}")

                    # Delete old resumes (but don't update the job yet since we're about to set the new one)
                    for old_resume in existing_resumes:
                        try:
                            success = await self.cache_manager.delete_resume(old_resume.id, user.id)
                            if success:
                                logger.info(f"Deleted old resume {old_resume.id} for job {job_dict.get('id')}")
                                # Remove from generation cache too
                                await self.cache_manager.remove_resume_status(old_resume.id, user.id)
                            else:
                                logger.warning(f"Failed to delete old resume {old_resume.id}")
                        except Exception as e:
                            logger.error(f"Error deleting old resume {old_resume.id}: {e}")

                elif handle_existing == "keep_both":
                    logger.info(f"Keeping {len(existing_resumes)} existing resume(s) alongside new resume for job {job_dict.get('id')}")
                    # Don't delete anything, just add the new resume
                    # Note: Only the newest resume will be linked to the job

            # Update the job with the NEW resume_id (this is always done, regardless of handle_existing)
            await self._update_job_with_resume_id(job_dict.get('id'), user.id, resume_id)

            # Update cache with completed status
            await self.cache_manager.set_resume_status(
                resume_id, user.id, ResumeGenerationStatus.COMPLETED,
                data={"yaml_content": yaml_content}
            )

            # Update job status to RESUME_GENERATED after everything is complete
            await self.cache_manager.update_job_status(
                job_dict.get('id'), user.id, JobStatus.RESUME_GENERATED
            )

            logger.info(f"Resume generation completed for job {job_dict.get('id')} for user {user.id}")

        except Exception as e:
            logger.error(f"Error generating resume for job {job_dict.get('id')} for user {user.id}: {e}")

            # Update cache with failed status
            await self.cache_manager.set_resume_status(
                resume_id, user.id, ResumeGenerationStatus.FAILED,
                error=str(e)
            )
    async def _update_job_with_resume_id(self, job_id: str, user_id:str, resume_id: str):
        """Update the job record with the generated resume ID."""
        try:
            # Use the new cache manager method to update job's resume_id
            success = await self.cache_manager.update_job_resume_id(job_id, user_id, resume_id)

            if success:
                logger.info(f"Successfully updated job {job_id} with resume_id {resume_id}")
            else:
                logger.error(f"Failed to update job {job_id} with resume_id {resume_id}")

        except Exception as e:
            logger.error(f"Error updating job {job_id} with resume_id {resume_id}: {e}")

    def _generate_resume_sync(self, job_dict: dict, user: User, resume_data: Dict[str, Any],
                              customize: bool, include_objective: bool = True) -> str:
        """Synchronous resume generation that runs in thread pool."""
        try:
            job_url = job_dict.get('job_url')
            parsed_job = job_dict.get("metadata")
            if not job_url:
                raise ValueError("Job URL not found in job data")

            if not customize:
                logger.info("Customization disabled; preserving the supplied resume content")
                return self.dict_to_yaml_string(resume_data)

            # Initialize ResumeImprover - it does ALL the work
            resume_improver = ResumeImprover(
                url=job_url,
                parsed_job=parsed_job,
                user = user
            )
            try:
                logger.info(f"Using user-provided resume data for job {job_dict.get('id')}")
                self._setup_resume_data(resume_improver, resume_data)

                # Let ResumeImprover do all the work with include_objective flag
                return resume_improver.create_complete_tailored_resume(include_objective)
            finally:
                resume_improver.close()

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

    async def check_resume_status(self, resume_id: str, user_id:str) -> Dict[str, Any]:
        """Check the status of a resume generation process."""
        # First check cache
        cache_entry = await self.cache_manager.get_resume_status(resume_id, user_id)

        if cache_entry:
            status = cache_entry["status"].value

            response = {
                "status": status,
                "resume_id": resume_id,
                "user_id": user_id,
                "updated_at": datetime.fromtimestamp(cache_entry["updated_at"]).isoformat(),
            }

            if cache_entry.get("error"):
                response["error"] = cache_entry["error"]

            if status == ResumeGenerationStatus.COMPLETED.value:
                # If completed, also get job info from database
                try:
                    resume = await self.cache_manager.get_resume(resume_id, user_id)
                    if resume and resume.job_id:
                        job_dict = await self.cache_manager.get_job(resume.job_id, user_id)
                        if job_dict:
                            response["job"] = job_dict
                            response["job_id"] = resume.job_id
                except Exception as e:
                    logger.warning(f"Could not fetch job info for completed resume: {e}")

            return response

        # If not in cache, check if it exists in database
        resume = await self.cache_manager.get_resume(resume_id, user_id)
        if not resume:
            raise ValueError(f"Resume not found with ID: {resume_id} for user: {user_id}")

        # If exists in database, it's completed
        job_dict = None
        if resume.job_id:
            job_dict = await self.cache_manager.get_job(resume.job_id, user_id)

        return {
            "status": "completed",
            "resume_id": resume_id,
            "job_id": resume.job_id,
            "user_id": user_id,
            "date_created": resume.date_created.isoformat() if resume.date_created else None,
            "job": job_dict
        }

    async def get_resume_content(self, resume_id: str, user_id:str, force_refresh: bool = False) -> str:
        """Get the resume content."""
        # If force_refresh, skip cache entirely
        if force_refresh:
            resume = await self.cache_manager.get_resume(resume_id, user_id)
            if not resume:
                raise ValueError(f"Resume not found with ID: {resume_id} for user: {user_id}")

            if not resume.yaml_content:
                raise ValueError("Resume generation is not complete")

            return resume.yaml_content

        # Normal flow - check cache first
        cache_entry = await self.cache_manager.get_resume_status(resume_id, user_id)

        if cache_entry and cache_entry["status"] == ResumeGenerationStatus.COMPLETED:
            yaml_content = cache_entry.get("data", {}).get("yaml_content")
            if yaml_content:
                return yaml_content

        # If not in cache, check database
        resume = await self.cache_manager.get_resume(resume_id, user_id)
        if not resume:
            raise ValueError(f"Resume not found with ID: {resume_id} for user: {user_id}")

        if not resume.yaml_content:
            raise ValueError("Resume generation is not complete")

        return resume.yaml_content

    async def upload_resume(self, file_path: str, user_id:str, file_content: bytes, job_id: str = None, ) -> Dict[str, Any]:
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
            success = await self.cache_manager.save_resume(resume, user_id)

            if not success:
                raise ValueError("Failed to save uploaded resume")

            # If this resume is for a specific job, update the job's resume_id
            if job_id:
                await self._update_job_with_resume_id(job_id, user_id, resume_id)

            return {
                "message": "Resume uploaded successfully",
                "resume_id": resume_id,
                "job_id": job_id,
                "user_id":user_id,
                "file_path": file_path
            }

        except UnicodeDecodeError:
            raise ValueError("Invalid file format. Please upload a text-based resume file.")
        except Exception as e:
            logger.error(f"Error uploading resume for user {user_id}: {e}")
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
