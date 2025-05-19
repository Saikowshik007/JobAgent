from typing import Optional, Dict, Any
import os
import logging
import uuid
from datetime import datetime
from services.resume_improver import ResumeImprover
from utils import file_handler, yaml_handler, resume_format_checker
import config
from dataModels.data_models import Resume, JobStatus


logger = logging.getLogger(__name__)

class ResumeGenerator:
    """
    Class to handle resume generation for jobs.
    Acts as a bridge between the API controller and the ResumeImprover class.
    """

    def __init__(self, db_manager, user_id: str, api_key:str):
        """Initialize the ResumeGenerator with database manager and user ID.

        Args:
            db_manager: Database cache manager
            user_id: ID of the user generating the resume
        """
        self.db_manager = db_manager
        self.user_id = user_id
        self.api_key = api_key
        self.default_resume_path = config.get("files.default_resume")

    async def generate_resume(self, job_id: str, template: str = "standard", customize: bool = True) -> Dict[str, Any]:
        """Generate a tailored resume for a specific job.

        Args:
            job_id: ID of the job to generate resume for
            template: Resume template to use
            customize: Whether to customize resume for the job

        Returns:
            Dict containing resume details and status
        """
        # Get the job from database
        job = await self.db_manager.db.get_job(job_id, self.user_id)
        if not job:
            raise ValueError(f"Job not found with ID: {job_id} for user: {self.user_id}")

        # Generate a unique ID for the resume
        resume_id = str(uuid.uuid4())

        # Create job directory if it doesn't exist
        job_dir = os.path.join(config.get("paths.data"), self.user_id, job.id)
        os.makedirs(job_dir, exist_ok=True)

        # Initialize resume object
        resume_path = os.path.join(job_dir, f"resume_{resume_id}.yaml")

        # Create initial resume record in database
        resume = Resume(
            id=resume_id,
            job_id=job.id,
            file_path=resume_path,
            yaml_content="",  # Will be populated by background task
            date_created=datetime.now(),
            uploaded_to_simplify=False
        )

        await self.db_manager.db.save_resume(resume, self.user_id)

        # Update job status to indicate resume generation is in progress
        await self.db_manager.db.update_job_status(job.id, self.user_id, JobStatus.RESUME_GENERATED)

        # Also update resume_id in job record
        job.resume_id = resume_id
        await self.db_manager.db.save_job(job, self.user_id)

        return {
            "status": "generating",
            "message": f"Resume generation started for job {job.title} at {job.company}",
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
            resume_path: str,
            pdf_path: str,
            template: str = "standard",
            customize: bool = True
    ):
        """Background task to generate resume.

        Args:
            job: Job object
            resume_id: ID of the resume
            resume_path: Path to save the resume YAML
            pdf_path: Path to save the PDF
            template: Resume template to use
            customize: Whether to customize resume for the job
        """
        try:
            logger.info(f"Starting resume generation for job {job.id} for user {self.user_id}")
            # Initialize ResumeImprover with job URL and default resume
            resume_improver = ResumeImprover(
                url=job.linkedin_url,
                resume_location=self.default_resume_path,
                api_key=self.api_key
            )

            # Parse job description if we got it from the database
            if hasattr(job, 'description') and job.description:
                resume_improver.parse_raw_job_post(job.description)

            # Create tailored resume
            resume_improver.create_draft_tailored_resume(
                auto_open=False,
                manual_review=False,  # Skip manual review for API usage
                skip_pdf_create=False  # Generate PDF
            )

            # Read the generated YAML content
            yaml_content = yaml_handler.read_yaml(filename=resume_improver.yaml_loc)

            # Set editing to False
            yaml_content['editing'] = False

            # Write the YAML back
            yaml_handler.write_yaml(yaml_content, filename=resume_path)

            # Get the PDF path
            pdf_file = resume_improver.create_pdf(auto_open=False)

            # Update the resume record with content
            resume = await self.db_manager.db.get_resume(resume_id, self.user_id)
            if resume:
                resume.yaml_content = yaml_handler.dict_to_yaml_string(yaml_content)
                resume.file_path = pdf_file  # Update with actual PDF path
                await self.db_manager.db.save_resume(resume, self.user_id)

            logger.info(f"Resume generation completed for job {job.id} for user {self.user_id}")

        except Exception as e:
            logger.error(f"Error generating resume for job {job.id} for user {self.user_id}: {e}")

    async def check_resume_status(self, resume_id: str) -> Dict[str, Any]:
        """Check the status of a resume generation process.

        Args:
            resume_id: ID of the resume to check

        Returns:
            Dict containing status information
        """
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
            "has_pdf": os.path.exists(resume.file_path) if resume.file_path else False,
            "job": job.to_dict() if job else None
        }

    async def get_resume_file_path(self, resume_id: str, format: str = "pdf") -> str:
        """Get the file path for a resume.

        Args:
            resume_id: ID of the resume
            format: File format (pdf or yaml)

        Returns:
            File path for the resume
        """
        # Get the resume from database
        resume = await self.db_manager.db.get_resume(resume_id, self.user_id)
        if not resume:
            raise ValueError(f"Resume not found with ID: {resume_id} for user: {self.user_id}")

        # Check if resume generation is complete
        if not resume.yaml_content:
            raise ValueError("Resume generation is not complete")

        if format.lower() == "pdf":
            # Check if PDF exists
            if not resume.file_path or not os.path.exists(resume.file_path):
                raise ValueError("Resume PDF not found")

            return resume.file_path
        else:  # yaml format
            # Return the YAML content itself
            return resume.yaml_content

    async def upload_resume(self, file_path: str, file_content: bytes, job_id: str = None) -> Dict[str, Any]:
        """Upload a custom resume.

        Args:
            file_path: Original file path
            file_content: Binary content of the file
            job_id: Optional job ID to associate with the resume

        Returns:
            Dict containing upload information
        """
        # Generate a unique ID for the resume
        resume_id = str(uuid.uuid4())

        # Create user directory if it doesn't exist
        user_dir = os.path.join(config.get("paths.data"), self.user_id)
        os.makedirs(user_dir, exist_ok=True)

        # Determine file paths
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in [".yaml", ".yml", ".pdf"]:
            raise ValueError("Unsupported file format. Only YAML or PDF allowed.")

        is_yaml = file_ext in [".yaml", ".yml"]

        new_file_path = os.path.join(user_dir, f"resume_{resume_id}{file_ext}")

        # Save the uploaded file
        with open(new_file_path, "wb") as f:
            f.write(file_content)

        # If it's a YAML file, read its content
        yaml_content = ""
        if is_yaml:
            try:
                yaml_content = yaml_handler.dict_to_yaml_string(yaml_handler.read_yaml(filename=new_file_path))
            except Exception as e:
                os.remove(new_file_path)  # Clean up file if parsing fails
                raise ValueError(f"Invalid YAML file: {str(e)}")

        # Create resume record
        resume = Resume(
            id=resume_id,
            job_id=job_id,
            file_path=new_file_path,
            yaml_content=yaml_content,
            date_created=datetime.now(),
            uploaded_to_simplify=False
        )

        await self.db_manager.db.save_resume(resume, self.user_id)

        # If job_id is provided, update the job record
        if job_id:
            job = await self.db_manager.db.get_job(job_id, self.user_id)
            if job:
                job.resume_id = resume_id
                await self.db_manager.db.save_job(job, self.user_id)

        return {
            "status": "success",
            "message": "Resume uploaded successfully",
            "resume_id": resume_id,
            "file_type": "yaml" if is_yaml else "pdf",
            "job_id": job_id,
            "user_id": self.user_id
        }