import asyncio
import concurrent
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
from yaml import YAMLError

import config
from data.cache import ResumeGenerationStatus
from dataModels.data_models import JobStatus, Resume
from services.resume_improver import ResumeImprover

logger = config.getLogger("Resume Generator")

class ResumeGenerator:
    """
    Resume Generator using unified cache manager for all operations.
    """

    def __init__(self, cache_manager, user_id: str, api_key: str):
        """Initialize the ResumeGenerator with unified cache manager and user ID."""
        self.cache_manager = cache_manager  # Single unified cache manager
        self.user_id = user_id
        self.api_key = api_key
        # Thread pool for blocking operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def generate_resume(self, job_id: str, template: str = "standard",
                              customize: bool = True, resume_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a tailored resume for a specific job."""
        # Get the job from database using the unified cache manager
        job_dict = await self.cache_manager.get_job(job_id, self.user_id)
        if not job_dict:
            raise ValueError(f"Job not found with ID: {job_id} for user: {self.user_id}")

        # Convert job dict back to Job object for processing
        from dataModels.data_models import Job
        job = Job.from_dict(job_dict)

        # Generate a unique ID for the resume
        resume_id = str(uuid.uuid4())

        # Set initial status in cache using unified cache manager
        await self.cache_manager.set_resume_status(
            resume_id, self.user_id, ResumeGenerationStatus.PENDING
        )

        # Update job status to indicate resume generation is in progress
        await self.cache_manager.update_job_status(job.id, self.user_id, JobStatus.RESUME_GENERATED)

        # Also update resume_id in job record
        job.resume_id = resume_id
        await self.cache_manager.save_job(job, self.user_id)

        # Start background generation (non-blocking)
        asyncio.create_task(self._generate_resume_background(
            job, resume_id, template, customize, resume_data
        ))

        return {
            "status": "generating",
            "message": f"Resume generation started for job {job.metadata.get('job_title', 'Unknown')} at {job.metadata.get('company', 'Unknown')}",
            "resume_id": resume_id,
            "job_id": job.id,
            "user_id": self.user_id,
            "template": template,
            "estimated_completion_seconds": 60
        }

    async def _generate_resume_background(self, job, resume_id: str, template: str,
                                          customize: bool, resume_data: Optional[Dict[str, Any]]):
        """Background task to generate resume using thread pool."""
        try:
            logger.info(f"Starting resume generation for job {job.id} for user {self.user_id}")

            # Update status to in progress using unified cache manager
            await self.cache_manager.set_resume_status(
                resume_id, self.user_id, ResumeGenerationStatus.IN_PROGRESS
            )

            # Run the blocking resume generation in thread pool
            loop = asyncio.get_event_loop()
            yaml_content = await loop.run_in_executor(
                self.thread_pool,
                self._generate_resume_sync,
                job, resume_data
            )

            # Create the resume object
            resume = Resume(
                id=resume_id,
                job_id=job.id,
                file_path="",  # Not used anymore
                yaml_content=yaml_content,
                date_created=datetime.now(),
                uploaded_to_simplify=False
            )

            # Save completed resume to database using unified cache manager
            await self.cache_manager.save_resume(resume, self.user_id)

            # Update cache with completed status using unified cache manager
            await self.cache_manager.set_resume_status(
                resume_id, self.user_id, ResumeGenerationStatus.COMPLETED,
                data={"yaml_content": yaml_content}
            )

            logger.info(f"Resume generation completed for job {job.id} for user {self.user_id}")

        except Exception as e:
            logger.error(f"Error generating resume for job {job.id} for user {self.user_id}: {e}")

            # Update cache with failed status using unified cache manager
            await self.cache_manager.set_resume_status(
                resume_id, self.user_id, ResumeGenerationStatus.FAILED,
                error=str(e)
            )

    def _generate_resume_sync(self, job, resume_data: Optional[Dict[str, Any]]) -> str:
        """Synchronous resume generation that runs in thread pool."""
        try:
            # Initialize ResumeImprover
            resume_improver = None

            if resume_data:
                logger.info(f"Using user-provided resume data for job {job.id}")
                resume_improver = ResumeImprover(
                    url=job.job_url,
                    api_key=self.api_key,
                    resume_location=None
                )
                resume_improver.resume = resume_data
                resume_improver.basic_info = self.get_dict_field(field="basic", data_dict=resume_data)
                resume_improver.education = self.get_dict_field(field="education", data_dict=resume_data)
                resume_improver.experiences = self.get_dict_field(field="experiences", data_dict=resume_data)
                resume_improver.projects = self.get_dict_field(field="projects", data_dict=resume_data)
                resume_improver.skills = self.get_dict_field(field="skills", data_dict=resume_data)
                resume_improver.objective = self.get_dict_field(field="objective", data_dict=resume_data)
                resume_improver.degrees = resume_improver._get_degrees(resume_data)
            else:
                logger.info(f"Using default resume template for job {job.id}")
                resume_improver = ResumeImprover(
                    url=job.job_url,
                    resume_location=None,  # Will need to handle default resume
                    api_key=self.api_key
                )

            # Initialize job parsing - check if job details are already in metadata
            if hasattr(job, 'metadata') and job.metadata:
                # If metadata contains parsed job details, use them
                if 'job_title' in job.metadata and 'company' in job.metadata:
                    resume_improver.parsed_job = job.metadata
                    logger.info(f"Using job details from metadata for job {job.id}")
                else:
                    # Parse the job from URL
                    resume_improver.download_and_parse_job_post()
            else:
                # Parse the job from URL
                resume_improver.download_and_parse_job_post()

            # Generate resume content
            logger.info("Extracting matched skills...")
            skills = resume_improver.extract_matched_skills(verbose=False)

            logger.info("Updating bullet points...")
            experiences = resume_improver.rewrite_unedited_experiences(verbose=False)

            logger.info("Updating projects...")
            projects = resume_improver.rewrite_unedited_projects(verbose=False)

            objective = resume_improver.write_objective()

            # Create resume content dictionary
            yaml_content_dict = {
                'editing': False,
                'basic': resume_improver.basic_info,
                'objective': objective,
                'education': resume_improver.education,
                'experiences': experiences,
                'projects': projects,
                'skills': skills,
                'generated_info': {
                    'job_id': job.id,
                    'job_title': job.metadata.get('job_title') if hasattr(job, 'metadata') and job.metadata else None,
                    'company': job.metadata.get('company') if hasattr(job, 'metadata') and job.metadata else None,
                    'generated_at': datetime.now().isoformat(),
                    'user_id': self.user_id,
                    'used_user_resume': resume_data is not None
                }
            }

            # Convert to YAML string
            return self.dict_to_yaml_string(yaml_content_dict)

        except Exception as e:
            logger.error(f"Synchronous resume generation failed: {e}")
            raise

    async def check_resume_status(self, resume_id: str) -> Dict[str, Any]:
        """Check the status of a resume generation process using unified cache manager."""
        # First check cache using unified cache manager
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

        # If not in cache, check if it exists in database (for old resumes)
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
        """Get the resume content using unified cache manager."""
        # First check cache for recently generated resumes
        cache_entry = await self.cache_manager.get_resume_status(resume_id, self.user_id)

        if cache_entry and cache_entry["status"] == ResumeGenerationStatus.COMPLETED:
            yaml_content = cache_entry.get("data", {}).get("yaml_content")
            if yaml_content:
                return yaml_content

        # If not in cache or cache doesn't have content, check database
        resume = await self.cache_manager.get_resume(resume_id, self.user_id)
        if not resume:
            raise ValueError(f"Resume not found with ID: {resume_id} for user: {self.user_id}")

        if not resume.yaml_content:
            raise ValueError("Resume generation is not complete")

        return resume.yaml_content

    async def upload_resume(self, file_path: str, file_content: bytes, job_id: str = None) -> Dict[str, Any]:
        """Upload a custom resume using unified cache manager."""
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

            # Save resume to database using unified cache manager
            success = await self.cache_manager.save_resume(resume, self.user_id)

            if not success:
                raise ValueError("Failed to save uploaded resume")

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