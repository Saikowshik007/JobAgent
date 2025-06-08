from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from prompts.prompts import Prompts

Prompts.initialize()


class ResumeSectionHighlight(BaseModel):
    """Pydantic class that defines each highlight to be returned by the LLM."""

    highlight: str = Field(
        ..., description=Prompts.descriptions["RESUME_SECTION_HIGHLIGHT"]["highlight"]
    )
    relevance: int = Field(
        ...,
        description=Prompts.descriptions["RESUME_SECTION_HIGHLIGHT"]["relevance"],
        enum=[1, 2, 3, 4, 5],
    )


class ResumeSectionHighlighterOutput(BaseModel):
    """Pydantic class that defines a list of highlights to be returned by the LLM."""

    plan: List[str] = Field(
        ...,
        description=Prompts.descriptions["RESUME_SECTION_HIGHLIGHTER_OUTPUT"]["plan"],
    )
    additional_steps: List[str] = Field(
        ...,
        description=Prompts.descriptions["RESUME_SECTION_HIGHLIGHTER_OUTPUT"]["additional_steps"],
    )
    work: List[str] = Field(
        ...,
        description=Prompts.descriptions["RESUME_SECTION_HIGHLIGHTER_OUTPUT"]["work"],
    )
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

    plan: List[str] = Field(
        description=Prompts.descriptions["RESUME_SKILLS_MATCHER_OUTPUT"]["plan"]
    )
    additional_steps: List[str] = Field(
        description=Prompts.descriptions["RESUME_SKILLS_MATCHER_OUTPUT"]["additional_steps"],
    )
    work: List[str] = Field(
        description=Prompts.descriptions["RESUME_SKILLS_MATCHER_OUTPUT"]["work"]
    )
    final_answer: ResumeSkills = Field(
        description=Prompts.descriptions["RESUME_SKILLS_MATCHER_OUTPUT"]["final_answer"],
    )


class ResumeSummarizerOutput(BaseModel):
    """Pydantic class that defines a list of skills to be returned by the LLM."""

    plan: List[str] = Field(
        ..., description=Prompts.descriptions["RESUME_OBJECTIVE_OUTPUT"]["plan"]
    )
    additional_steps: List[str] = Field(
        ...,
        description=Prompts.descriptions["RESUME_OBJECTIVE_OUTPUT"]["additional_steps"],
    )
    work: List[str] = Field(
        ..., description=Prompts.descriptions["RESUME_OBJECTIVE_OUTPUT"]["work"]
    )
    final_answer: str = Field(
        ...,
        description=Prompts.descriptions["RESUME_OBJECTIVE_OUTPUT"]["final_answer"],
    )


class ResumeImprovements(BaseModel):
    """Pydantic class that defines a list of improvements to be returned by the LLM."""

    section: str = Field(
        ...,
        enum=[
            "objective",
            "education",
            "experiences",
            "projects",
            "skills",
            "spelling and grammar",
            "other",
        ],
    )
    improvements: List[str] = Field(
        ..., description=Prompts.descriptions["RESUME_IMPROVEMENTS"]["improvements"]
    )


class ResumeImproverOutput(BaseModel):
    """Pydantic class that defines a list of improvements to be returned by the LLM."""

    plan: List[str] = Field(
        ..., description=Prompts.descriptions["RESUME_IMPROVER_OUTPUT"]["plan"]
    )
    additional_steps: List[str] = Field(
        ...,
        description=Prompts.descriptions["RESUME_IMPROVER_OUTPUT"]["additional_steps"],
    )
    work: List[str] = Field(
        ..., description=Prompts.descriptions["RESUME_IMPROVER_OUTPUT"]["work"]
    )
    final_answer: List[ResumeImprovements] = Field(
        ..., description=Prompts.descriptions["RESUME_IMPROVER_OUTPUT"]["final_answer"]
    )

class ComprehensiveSuggestions(BaseModel):
    """Pydantic model that combines content improvements and optimization suggestions."""

    content_improvements: List[ResumeImprovements] = Field(
        ..., description="List of content improvement suggestions organized by section"
    )
    optimization_suggestions: List[str] = Field(
        ..., description="List of specific one-page optimization suggestions in priority order"
    )


class ComprehensiveSuggestionsOutput(BaseModel):
    """Pydantic class that defines the output for comprehensive suggestions analysis."""

    plan: List[str] = Field(
        ..., description="Itemized plan for analyzing both content and length improvements"
    )
    additional_steps: List[str] = Field(
        ..., description="Additional steps needed for comprehensive analysis"
    )
    work: List[str] = Field(
        ..., description="Detailed work performed during analysis"
    )
    final_answer: ComprehensiveSuggestions = Field(
        ..., description="Both content improvements and optimization suggestions"
    )


class AllSuggestionsApplierOutput(BaseModel):
    """Pydantic class that defines the output for applying all suggestions."""

    plan: List[str] = Field(
        ..., description="Itemized plan for applying all suggestions"
    )
    additional_steps: List[str] = Field(
        ..., description="Additional steps needed for comprehensive application"
    )
    work: List[str] = Field(
        ..., description="Detailed work performed during application"
    )
    final_answer: str = Field(
        ..., description="The complete improved and optimized resume in valid YAML format"
    )
