# Pydantic models for API validation
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class JobStatusEnum(str, Enum):
    """Enum for job status values matching the JobStatus class."""
    NEW = "NEW"
    INTERESTED = "INTERESTED"
    RESUME_GENERATED = "RESUME_GENERATED"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    INTERVIEW = "INTERVIEW"
    OFFER = "OFFER"
    DECLINED = "DECLINED"

class FilterOptions(BaseModel):
    """Search filter options."""
    experience_level: Optional[List[str]] = None
    job_type: Optional[List[str]] = None
    date_posted: Optional[str] = None
    workplace_type: Optional[List[str]] = None
    easy_apply: Optional[bool] = None

class JobSearchRequest(BaseModel):
    """Job search request model."""
    keywords: str = Field(..., description="Job title or keywords")
    location: str = Field(..., description="Job location")
    filters: Optional[FilterOptions] = Field(default_factory=FilterOptions, description="Search filters")
    max_jobs: Optional[int] = Field(default=20, description="Maximum number of jobs to return")
    headless: Optional[bool] = Field(default=True, description="Run browser in headless mode")

class JobStatusUpdateRequest(BaseModel):
    """Job status update request model."""
    status: JobStatusEnum = Field(..., description="New job status")

class GenerateResumeRequest(BaseModel):
    """Resume generation request model."""
    job_id: str = Field(..., description="ID of the job to generate resume for")
    template: Optional[str] = Field("standard", description="Resume template to use")
    customize: Optional[bool] = Field(True, description="Whether to customize resume for the job")
    resume_data: Optional[Dict[str, Any]] = Field(None, description="User's resume data in YAML format")

class UploadToSimplifyRequest(BaseModel):
    """Resume upload to Simplify request model."""
    job_id: str = Field(..., description="ID of the job")
    resume_id: Optional[str] = None