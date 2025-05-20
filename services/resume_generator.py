from io import StringIO
from typing import Optional, Dict, Any
import os
import logging
import uuid
from datetime import datetime

import yaml
from yaml import YAMLError

from services.resume_improver import ResumeImprover
from utils import  resume_format_checker
import config
from dataModels.data_models import Resume, JobStatus


logger = logging.getLogger(__name__)

class ResumeGenerator:
    """
    Class to handle resume generation for jobs.
    Acts as a bridge between the API controller and the ResumeImprover class.
    """

    def __init__(self, db_manager, user_id: str, api_key:str):
        """Initialize the ResumeGenerator with database manager and user ID."""
        self.db_manager = db_manager
        self.user_id = user_id
        self.api_key = api_key
        self.default_resume_path = config.get("files.default_resume")

    async def generate_resume(self, job_id: str, template: str = "standard", customize: bool = True, resume_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a tailored resume for a specific job."""
        # Get the job from database
        job = await self.db_manager.db.get_job(job_id, self.user_id)
        if not job:
            raise ValueError(f"Job not found with ID: {job_id} for user: {self.user_id}")

        # Generate a unique ID for the resume
        resume_id = str(uuid.uuid4())

        # Create initial resume record in database
        resume = Resume(
            id=resume_id,
            job_id=job.id,
            file_path="",  # We don't need file paths anymore
            yaml_content="",  # Will be populated by background task
            date_created=datetime.now(),
            uploaded_to_simplify=False
        )

        # Store whether we have user-provided resume data
        if resume_data:
            logger.info(f"User-provided resume data received for job {job_id}")
            if not hasattr(resume, 'metadata') or resume.metadata is None:
                resume.metadata = {}
            resume.metadata['has_user_resume_data'] = True
        else:
            logger.warning(f"No user-provided resume data for job {job_id}")
            if not hasattr(resume, 'metadata') or resume.metadata is None:
                resume.metadata = {}
            resume.metadata['has_user_resume_data'] = False

        await self.db_manager.db.save_resume(resume, self.user_id)

        # Update job status to indicate resume generation is in progress
        await self.db_manager.db.update_job_status(job.id, self.user_id, JobStatus.RESUME_GENERATED)

        # Also update resume_id in job record
        job.resume_id = resume_id
        await self.db_manager.db.save_job(job, self.user_id)

        return {
            "status": "generating",
            "message": f"Resume generation started for job {job.metadata.get('job_title')} at {job.metadata.get('company')}",
            "resume_id": resume_id,
            "job_id": job.id,
            "user_id": self.user_id,
            "template": template,
            "estimated_completion_seconds": 60  # Approximate time for generation
        }

    async def generate_resume_background(
            self,
            job,
            resume_id: str,
            resume_path: str = None,  # Not used
            pdf_path: str = None,     # Not used
            template: str = "standard",
            customize: bool = True,
            resume_data: Optional[Dict[str, Any]] = None
    ):
        """Background task to generate resume."""
        try:
            logger.info(f"Starting resume generation for job {job.id} for user {self.user_id}")

            # Initialize ResumeImprover using in-memory data approach
            resume_improver = None

            # Create a virtual ResumeImprover that uses the provided resume_data
            if resume_data:
                logger.info(f"Using user-provided resume data for job {job.id}")

                # Initialize ResumeImprover with default resume path (we'll override its internal state later)
                resume_improver = ResumeImprover(
                    url=job.job_url,
                    api_key=self.api_key,
                    resume_location=self.default_resume_path  # This will be overridden
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
                    resume_location=self.default_resume_path,
                    api_key=self.api_key
                )

            # Check if we have parsed job details in metadata
            parsed_job_details = None
            if hasattr(job, 'metadata') and job.metadata and 'parsed_job_details' in job.metadata:
                parsed_job_details = job.metadata['parsed_job_details']
                logger.info(f"Using parsed job details from metadata for job {job.id}")

            # Initialize job parsing based on available data
            if parsed_job_details:
                # Set the parsed job directly instead of re-parsing
                resume_improver.parsed_job = parsed_job_details
            elif hasattr(job, 'description') and job.description:
                resume_improver.parse_raw_job_post(job.description)
            else:
                # If no description either, use URL to fetch job details
                resume_improver.download_and_parse_job_post()

            # Extract skills and update content without generating files
            logger.info("Extracting matched skills...")
            skills = resume_improver.extract_matched_skills(verbose=False)

            logger.info("Updating bullet points...")
            experiences = resume_improver.rewrite_unedited_experiences(verbose=False)

            logger.info("Updating projects...")
            projects = resume_improver.rewrite_unedited_projects(verbose=False)

            # Create resume content dictionary
            yaml_content = {
                'editing': False,
                'basic': resume_improver.basic_info,
                'objective': resume_improver.objective,
                'education': resume_improver.education,
                'experiences': experiences,
                'projects': projects,
                'skills': skills,
                'generated_info': {
                    'job_id': job.id,
                    'job_title': job.metadata.get('job_title') if hasattr(job, 'metadata') else None,
                    'company': job.metadata.get('company') if hasattr(job, 'metadata') else None,
                    'generated_at': datetime.now().isoformat(),
                    'user_id': self.user_id,
                    'used_user_resume': resume_data is not None
                }
            }

            # Convert to YAML string for database storage
            yaml_string = self.dict_to_yaml_string(yaml_content)

            # Update the resume record with content
            resume = await self.db_manager.db.get_resume(resume_id, self.user_id)
            if resume:
                resume.yaml_content = yaml_string
                await self.db_manager.db.save_resume(resume, self.user_id)

            logger.info(f"Resume generation completed for job {job.id} for user {self.user_id}")

        except Exception as e:
            logger.error(f"Error generating resume for job {job.id} for user {self.user_id}: {e}")

    async def check_resume_status(self, resume_id: str) -> Dict[str, Any]:
        """Check the status of a resume generation process."""
        # Get the resume from database
        resume = await self.db_manager.db.get_resume(resume_id, self.user_id)
        if not resume:
            raise ValueError(f"Resume not found with ID: {resume_id} for user: {self.user_id}")

        # Determine status
        status = "unknown"

        if resume.yaml_content:
            status = "completed"
        elif resume.date_created:
            # If created but no content yet, it's still generating
            status = "generating"

        # Get associated job
        job = None
        if resume.job_id:
            job = await self.db_manager.db.get_job(resume.job_id, self.user_id)

        return {
            "status": status,
            "resume_id": resume_id,
            "job_id": resume.job_id,
            "user_id": self.user_id,
            "date_created": resume.date_created.isoformat() if resume.date_created else None,
            "has_pdf": False,  # We don't generate PDFs server-side anymore
            "job": job.to_dict() if job else None
        }

    async def get_resume_file_path(self, resume_id: str, format: str = "pdf") -> str:
        """Get the resume content (not a file path anymore)."""
        # Get the resume from database
        resume = await self.db_manager.db.get_resume(resume_id, self.user_id)
        if not resume:
            raise ValueError(f"Resume not found with ID: {resume_id} for user: {self.user_id}")

        # Check if resume generation is complete
        if not resume.yaml_content:
            raise ValueError("Resume generation is not complete")

        # For PDF format, we respond that it's not supported on the server
        if format.lower() == "pdf":
            raise ValueError("PDF generation is not supported on the server. Please generate PDF in the UI.")

        # Return the YAML content itself
        return resume.yaml_content


    def get_dict_field(self,field: str, data_dict: dict) -> Optional[dict]:
        """
        Retrieves a field from a dictionary.

        Args:
            field (str): The field to retrieve.
            data_dict (dict): The dictionary to retrieve the field from.

        Returns:
            Optional[dict]: The value of the field, or None if the field is missing.
        """
        try:
            return data_dict[field]
        except KeyError as e:
            message = f"`{field}` is missing in raw resume."
            config.getLogger("file_handler").warning(message)
        return None

    def dict_to_yaml_string(self,data: dict) -> str:
        """
        Converts a dictionary to a YAML-formatted string.

        Args:
            data (dict): Data to be converted to YAML string.

        Returns:
            str: YAML-formatted string.
        """
        yaml.allow_unicode = True
        try:
            yaml.allow_unicode = True
            stream = StringIO()
            yaml.dump(data, stream=stream)
            return stream.getvalue()
        except YAMLError as e:
            logger.error("Failed to convert dict to YAML string.")
            raise e