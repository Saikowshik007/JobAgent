from enum import Enum
from datetime import datetime
from typing import Dict, Optional, Any, Union


class JobStatus(Enum):
    """Enumeration representing the status of a job application."""
    NEW = "NEW"
    INTERESTED = "INTERESTED"
    RESUME_GENERATED = "RESUME_GENERATED"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    INTERVIEW = "INTERVIEW"
    OFFER = "OFFER"
    DECLINED = "DECLINED"
    
    def __str__(self):
        return self.value


class Job:
    """Class representing a job posting and application status."""
    
    def __init__(
        self,
        id: str,
        job_url:Optional[str]= None,
        status: Union[JobStatus, str] = JobStatus.NEW,
        date_found: Optional[datetime] = None,
        applied_date: Optional[datetime] = None,
        rejected_date: Optional[datetime] = None,
        resume_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Job instance.
        
        Args:
            id: Unique identifier for the job
            job_url: URL for website
            status: Current status of the job application
            date_found: Date the job was found
            applied_date: Date applied to the job
            rejected_date: Date rejected from the job
            resume_id: ID of the resume used for this job
            metadata: Additional metadata about the job
        """
        self.id = id
        self.job_url = job_url
        
        # Convert string status to enum if needed
        if isinstance(status, str):
            try:
                self.status = JobStatus(status)
            except ValueError:
                self.status = JobStatus.NEW
        else:
            self.status = status
            
        self.date_found = date_found or datetime.now()
        self.applied_date = applied_date
        self.rejected_date = rejected_date
        self.resume_id = resume_id
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Job instance to a dictionary for database storage."""
        return {
            "id": self.id,
            "job_url": self.job_url,
            "status": str(self.status),
            "date_found": self.date_found.isoformat() if self.date_found else None,
            "applied_date": self.applied_date.isoformat() if self.applied_date else None,
            "rejected_date": self.rejected_date.isoformat() if self.rejected_date else None,
            "resume_id": self.resume_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create a Job instance from a dictionary."""
        # Convert datetime strings back to datetime objects
        date_found = datetime.fromisoformat(data["date_found"]) if data.get("date_found") else None
        applied_date = datetime.fromisoformat(data["applied_date"]) if data.get("applied_date") else None
        rejected_date = datetime.fromisoformat(data["rejected_date"]) if data.get("rejected_date") else None
        
        return cls(
            id=data["id"],
            job_url=data["job_url"],
            status=data["status"],
            date_found=date_found,
            applied_date=applied_date,
            rejected_date=rejected_date,
            resume_id=data.get("resume_id"),
            metadata=data.get("metadata", {})
        )


class Resume:
    """Class representing a resume."""
    
    def __init__(
        self,
        id: str,
        job_id: Optional[str],
        file_path: str,
        yaml_content: str,
        date_created: Optional[datetime] = None,
        uploaded_to_simplify: bool = False
    ):
        """
        Initialize a Resume instance.
        
        Args:
            id: Unique identifier for the resume
            job_id: Job ID this resume is for (if job-specific)
            file_path: Path to the resume file
            yaml_content: YAML content of the resume
            date_created: Date the resume was created
            uploaded_to_simplify: Whether the resume was uploaded to Simplify.jobs
        """
        self.id = id
        self.job_id = job_id
        self.file_path = file_path
        self.yaml_content = yaml_content
        self.date_created = date_created or datetime.now()
        self.uploaded_to_simplify = uploaded_to_simplify
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Resume instance to a dictionary for database storage."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "file_path": self.file_path,
            "yaml_content": self.yaml_content,
            "date_created": self.date_created.isoformat() if self.date_created else None,
            "uploaded_to_simplify": self.uploaded_to_simplify
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resume':
        """Create a Resume instance from a dictionary."""
        # Convert datetime strings back to datetime objects
        date_created = datetime.fromisoformat(data["date_created"]) if data.get("date_created") else None
        
        return cls(
            id=data["id"],
            job_id=data.get("job_id"),
            file_path=data["file_path"],
            yaml_content=data["yaml_content"],
            date_created=date_created,
            uploaded_to_simplify=data.get("uploaded_to_simplify", False)
        )
