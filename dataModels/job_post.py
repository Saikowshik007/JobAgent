from pydantic import BaseModel, Field
from typing import List, Optional
from prompts.prompts import Prompts
import services
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
        self.extractor_llm = services.langchain_helpers.create_llm(
            user= user,
            chat_model=ChatOpenAI,
            model_name=user.model,
            temperature=user.preferences.get("temperature"),
            cache=True
        )
        self.parsed_job = None

    def parse_job_post(self, use_enhanced_prompts=True, **chain_kwargs) -> dict:
        """
        Parse the job posting to extract job description and skills.

        Args:
            use_enhanced_prompts: If True, uses JOB_EXTRACTOR prompts for better extraction.
                                 If False, uses simple extraction (backward compatible).
        """
        if use_enhanced_prompts and "JOB_EXTRACTOR" in Prompts.lookup:
            # Use enhanced extraction with JOB_EXTRACTOR prompts
            prompt = ChatPromptTemplate(messages=Prompts.lookup["JOB_EXTRACTOR"])
            chain = prompt | self.extractor_llm.with_structured_output(JobDescription)

            # Prepare inputs for the prompt template
            inputs = {"raw_job_text": self.posting}

            self.parsed_job = chain.invoke(inputs).dict()
        else:
            # Fallback to simple extraction (original method)
            model = self.extractor_llm.with_structured_output(JobDescription)
            self.parsed_job = model.invoke(self.posting).dict()

        return self.parsed_job