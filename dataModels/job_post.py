from pydantic import BaseModel, Field
from typing import List, Optional
from prompts.prompts import Prompts
from services.langchain_helpers import invoke_structured

Prompts.initialize()


class JobDescription(BaseModel):
    """Description of a job posting."""

    company: Optional[str] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["company"]
    )
    job_title: Optional[str] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["job_title"]
    )
    job_type: Optional[str] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["job_type"]
    )
    location: Optional[str] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["location"]
    )
    team: Optional[str] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["team"]
    )
    job_summary: Optional[str] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["job_summary"]
    )
    salary: Optional[str] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["salary"]
    )
    duties: Optional[List[str]] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["duties"]
    )
    qualifications: Optional[List[str]] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["qualifications"]
    )
    ats_keywords: Optional[List[str]] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["ats_keywords"]
    )
    is_fully_remote: Optional[bool] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["is_fully_remote"]
    )
    technical_skills: Optional[List[str]] = Field(
        None, description=Prompts.descriptions["JOB_DESCRIPTION"]["technical_skills"]
    )
    non_technical_skills: Optional[List[str]] = Field(
        None,
        description=Prompts.descriptions["JOB_DESCRIPTION"]["non_technical_skills"],
    )


class JobPost:
    def __init__(self, posting: str, user):
        """Initialize JobPost with the job posting string."""
        self.posting = posting
        self.user = user
        self.parsed_job = None

    def parse_job_post(self) -> dict:
        """Parse the job posting to extract job description and skills."""
        self.parsed_job = invoke_structured(
            self.user,
            "JOB_DESCRIPTION",
            JobDescription,
            posting=self.posting,
        ).model_dump()
        return self.parsed_job
