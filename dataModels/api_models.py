# Pydantic models for API validation
from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from dataModels.user_model import User


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

class JobStatusUpdateRequest(BaseModel):
    """Job status update request model."""
    status: JobStatusEnum = Field(..., description="New job status")

class GenerateResumeRequest(BaseModel):
    """Resume generation request model."""
    job_id: str = Field(..., description="ID of the job to generate resume for")
    template: Optional[str] = Field("standard", description="Resume template to use")
    customize: Optional[bool] = Field(True, description="Whether to customize resume for the job")
    resume_data: Optional[Dict[str, Any]] = Field(None, description="User's resume data in YAML format")
    include_objective: Optional[bool] = Field(True, description="Whether to include an objective section")
    user: User = Field(..., description="User object")
