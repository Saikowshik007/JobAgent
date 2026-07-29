from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from prompts.prompts import Prompts

Prompts.initialize()


class ResumeSectionHighlight(BaseModel):
    """Pydantic class that defines each highlight to be returned by the LLM."""

    source_index: int = Field(
        ..., ge=1,
        description="One-based index of the source highlight this rewrite replaces.",
    )
    highlight: str = Field(
        ..., description=Prompts.descriptions["RESUME_SECTION_HIGHLIGHT"]["highlight"]
    )
    relevance: int = Field(
        ...,
        description=Prompts.descriptions["RESUME_SECTION_HIGHLIGHT"]["relevance"],
        ge=1,
        le=5,
    )


class ResumeSectionHighlighterOutput(BaseModel):
    """Pydantic class that defines a list of highlights to be returned by the LLM."""

    final_answer: List[ResumeSectionHighlight] = Field(
        ...,
        description=Prompts.descriptions["RESUME_SECTION_HIGHLIGHTER_OUTPUT"]["final_answer"],
    )


class ResumeSkills(BaseModel):
    """Pydantic model that defines grouped skills with dynamic subcategories for technical skills and simple list for non-technical."""

    technical_skills: Optional[Dict[str, List[str]]] = Field(
        default_factory=dict,
        description=Prompts.descriptions["RESUME_SKILLS"]["technical_skills"]
    )
    non_technical_skills: Optional[List[str]] = Field(
        default_factory=list,
        description=Prompts.descriptions["RESUME_SKILLS"]["non_technical_skills"]
    )


class ResumeSkillsMatcherOutput(BaseModel):
    """Pydantic class that defines a list of skills to be returned by the LLM."""

    final_answer: ResumeSkills = Field(
        description=Prompts.descriptions["RESUME_SKILLS_MATCHER_OUTPUT"]["final_answer"],
    )


class ResumeSummarizerOutput(BaseModel):
    """Pydantic class that defines a list of skills to be returned by the LLM."""

    final_answer: str = Field(
        ...,
        description=Prompts.descriptions["RESUME_OBJECTIVE_OUTPUT"]["final_answer"],
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


class ResumeValidationOutput(BaseModel):
    """Section-level factual-grounding verdicts for a tailored resume."""

    approved_section_ids: List[str] = Field(default_factory=list)
    rejected_section_ids: List[str] = Field(default_factory=list)
