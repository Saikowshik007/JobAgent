from pydantic import BaseModel, ConfigDict, Field
from typing import List
from prompts.prompts import Prompts

Prompts.initialize()


class _ClosedSchema(BaseModel):
    """OpenAI structured outputs require a closed JSON object at every level."""

    model_config = ConfigDict(extra="forbid")


class SourceResumeBasic(_ClosedSchema):
    name: str
    address: str
    email: str
    phone: str
    websites: List[str]


class SourceResumeDegree(_ClosedSchema):
    names: List[str]
    gpa: str
    dates: str


class SourceResumeEducation(_ClosedSchema):
    school: str
    degrees: List[SourceResumeDegree]


class SourceResumeTitle(_ClosedSchema):
    name: str
    startdate: str
    enddate: str


class SourceResumeExperience(_ClosedSchema):
    company: str
    skip_name: bool
    location: str
    titles: List[SourceResumeTitle]
    highlights: List[str]


class SourceResumeProject(_ClosedSchema):
    name: str
    technologies: str
    link: str
    hyperlink: bool
    show_link: bool
    highlights: List[str]


class SourceResumeSkillGroup(_ClosedSchema):
    category: str
    skills: List[str]


class SourceResumeData(_ClosedSchema):
    basic: SourceResumeBasic
    objective: str
    education: List[SourceResumeEducation]
    experiences: List[SourceResumeExperience]
    projects: List[SourceResumeProject]
    skills: List[SourceResumeSkillGroup]


class SourceResumeExtractionOutput(_ClosedSchema):
    """Canonical resume data extracted from a user-uploaded source document."""

    final_answer: SourceResumeData


class TailoredResumeData(_ClosedSchema):
    """Primary tailored resume content returned by the full writer."""

    objective: str
    experiences: List[SourceResumeExperience]
    projects: List[SourceResumeProject]
    skills: List[SourceResumeSkillGroup]


class TailoredResumeWriterOutput(_ClosedSchema):
    """Structured output for the full tailored resume writer."""

    final_answer: TailoredResumeData = Field(
        ...,
        description=Prompts.descriptions["RESUME_TAILORED_RESUME_OUTPUT"]["final_answer"],
    )


class ResumeRepairWriterOutput(_ClosedSchema):
    """Structured output for the targeted repair writer."""

    final_answer: TailoredResumeData = Field(
        ...,
        description=Prompts.descriptions["RESUME_REPAIR_OUTPUT"]["final_answer"],
    )


class EvidenceMatch(BaseModel):
    """A candidate-supported match between one job requirement and resume evidence."""

    requirement: str
    source_ids: List[str] = Field(default_factory=list)
    safe_keywords: List[str] = Field(default_factory=list)
    match_strength: int = Field(ge=0, le=5)
    gap: bool


class ResumeEvidencePlanOutput(BaseModel):
    """Structured job-to-resume evidence map used to ground all later edits."""

    final_answer: List[EvidenceMatch] = Field(
        description=Prompts.descriptions["RESUME_EVIDENCE_PLAN_OUTPUT"]["final_answer"]
    )


class RejectedSectionFeedback(_ClosedSchema):
    """Validator feedback for one rejected tailored section."""

    section_id: str = Field(
        ...,
        description=Prompts.descriptions["RESUME_REJECTED_SECTION_FEEDBACK"]["section_id"],
    )
    reason: str = Field(
        ...,
        description=Prompts.descriptions["RESUME_REJECTED_SECTION_FEEDBACK"]["reason"],
    )


class ResumeValidationOutput(BaseModel):
    """Section-level factual-grounding verdicts for a tailored resume."""

    approved_section_ids: List[str] = Field(default_factory=list)
    rejected_section_ids: List[str] = Field(default_factory=list)
    rejected_sections: List["RejectedSectionFeedback"] = Field(default_factory=list)
