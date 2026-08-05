import asyncio
import time
from copy import deepcopy
from datetime import datetime
from typing import Optional

import httpx
import yaml
from bs4 import BeautifulSoup
from yaml import YAMLError

from config import config
from dataModels.job_post import JobPost

logger = config.getLogger("ResumeImprover")


class ResumeImprover:
    """Resume tailoring pipeline using planner -> full writer -> validator -> optional repair."""

    def __init__(self, url, user, parsed_job=None, llm_kwargs: dict = None,
                 timeout: int = 500, progress_callback=None):
        self.job_post_html_data = None
        self.job_post_raw = None
        self.resume = None
        self.job_post = None
        self.parsed_job = parsed_job
        self.llm_kwargs = llm_kwargs or {}
        self.user = user
        self.url = url
        self.timeout = timeout
        self.progress_callback = progress_callback
        self.basic_info = None
        self.education = None
        self.experiences = None
        self.projects = None
        self.skills = None
        self.objective = None
        self.degrees = None
        self.evidence_inventory = []
        self.evidence_plan = []

    def create_complete_tailored_resume(self, include_objective) -> str:
        """Create a complete tailored resume with one primary writer call."""
        logger.info("resume_tailoring_started", extra={"event.action": "resume_tailoring"})
        if not self.evidence_inventory:
            self._report_progress("evidence_planning", 20, "Matching job requirements to resume evidence")
            self.prepare_tailoring_plan()

        self._report_progress("tailored_resume_writer", 55, "Writing the full tailored resume")
        writer_started_at = time.time()
        tailored_content = self._write_tailored_resume(include_objective)
        logger.info(
            "resume_tailoring_writer_completed",
            extra={
                "event.duration_seconds": round(time.time() - writer_started_at, 2),
                "resume.objective_included": bool(tailored_content.get("objective")),
                "resume.skill_group_count": len(tailored_content.get("skills", [])),
                "resume.experience_count": len(tailored_content.get("experiences", [])),
                "resume.project_count": len(tailored_content.get("projects", [])),
            },
        )

        final_resume = {
            "editing": False,
            "basic": self.basic_info or {},
            "objective": tailored_content.get("objective", ""),
            "education": self.education or [],
            "experiences": tailored_content.get("experiences", []),
            "projects": tailored_content.get("projects", []),
            "skills": tailored_content.get("skills", []),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "job_url": self.url,
                "tailored": True,
                "match_report": self._match_report(),
            },
        }

        self._report_progress("grounding_validation", 85, "Validating the tailored resume")
        final_resume = self._validate_tailored_resume(final_resume, include_objective)

        self._report_progress("saving_resume", 95, "Preparing the tailored resume")
        yaml_content = self.dict_to_yaml_string(final_resume)
        logger.info("resume_tailoring_completed")
        return yaml_content

    def _report_progress(self, stage: str, progress_percentage: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(stage, progress_percentage, message)

    def prepare_tailoring_plan(self) -> dict:
        self.evidence_inventory = self._build_evidence_inventory()
        self.evidence_plan = self._build_evidence_plan()
        return self.export_tailoring_plan()

    def export_tailoring_plan(self) -> dict:
        return {
            "evidence_inventory": self.evidence_inventory or [],
            "evidence_plan": self.evidence_plan or [],
            "plan_summary": self._match_report(),
            "tailoring_brief": self._tailoring_brief(),
        }

    def load_tailoring_plan(self, plan_data: Optional[dict]) -> bool:
        if not isinstance(plan_data, dict):
            return False
        inventory = plan_data.get("evidence_inventory")
        plan = plan_data.get("evidence_plan")
        if not isinstance(inventory, list) or not isinstance(plan, list):
            return False
        if len(plan) < self._expected_requirement_count():
            return False
        self.evidence_inventory = inventory
        self.evidence_plan = plan
        return True

    def _build_evidence_inventory(self) -> list[dict]:
        inventory = []
        for index, education in enumerate(self.education or []):
            inventory.append({"section_id": f"education:{index}", "content": education})
        for index, experience in enumerate(self.experiences or []):
            context = {key: value for key, value in experience.items() if key != "highlights"}
            inventory.append({"section_id": f"experience:{index}:context", "content": context})
            for bullet_index, highlight in enumerate(experience.get("highlights", [])):
                inventory.append({
                    "section_id": f"experience:{index}:highlight:{bullet_index}",
                    "content": {"context": context, "highlight": highlight},
                })
        for index, project in enumerate(self.projects or []):
            context = {key: value for key, value in project.items() if key != "highlights"}
            inventory.append({"section_id": f"project:{index}:context", "content": context})
            for bullet_index, highlight in enumerate(project.get("highlights", [])):
                inventory.append({
                    "section_id": f"project:{index}:highlight:{bullet_index}",
                    "content": {"context": context, "highlight": highlight},
                })
        for index, skill_group in enumerate(self.skills or []):
            inventory.append({"section_id": f"skills:{index}", "content": skill_group})
        inventory.append({"section_id": "objective", "content": self.objective or ""})
        return inventory

    def _build_evidence_plan(self) -> list[dict]:
        if not self.evidence_inventory or not self.parsed_job:
            return []
        try:
            from dataModels.resume import ResumeEvidencePlanOutput
            from services.langchain_helpers import invoke_structured

            result = invoke_structured(
                self.user,
                "RESUME_EVIDENCE_PLANNER",
                ResumeEvidencePlanOutput,
                timeout_seconds=150.0,
                max_retries=1,
                **self._get_prompt_inputs(),
            )
            valid_ids = {item["section_id"] for item in self.evidence_inventory}
            plan = []
            for match in result.final_answer:
                item = match.model_dump()
                item["source_ids"] = [source_id for source_id in item["source_ids"] if source_id in valid_ids]
                if not item["source_ids"]:
                    item["gap"] = True
                plan.append(item)
            if len(plan) < self._expected_requirement_count():
                raise ValueError("Evidence planner did not evaluate every extracted job requirement")
            return plan
        except Exception as error:
            logger.error("Evidence planning failed: %s", error)
            raise RuntimeError(f"Evidence planning failed: {error}") from error

    def _match_report(self) -> dict:
        matches = [match for match in self.evidence_plan if not match.get("gap")]
        gaps = [match["requirement"] for match in self.evidence_plan if match.get("gap")]
        requirements_evaluated = len(matches) + len(gaps)
        return {
            "matched_requirements": len(matches),
            "requirements_evaluated": requirements_evaluated,
            "evidence_coverage_percentage": round((len(matches) / requirements_evaluated) * 100) if requirements_evaluated else 0,
            "gaps": gaps,
            "strong_matches": [match["requirement"] for match in matches if match.get("match_strength", 0) >= 4],
        }

    def _expected_requirement_count(self) -> int:
        if not isinstance(self.parsed_job, dict):
            return 0
        requirements = []
        for key in ("duties", "qualifications", "ats_keywords", "technical_skills", "non_technical_skills"):
            values = self.parsed_job.get(key) or []
            if isinstance(values, list):
                requirements.extend(str(value).strip().lower() for value in values if str(value).strip())
        return len(set(requirements))

    def _tailoring_brief(self) -> dict:
        matched = [match for match in self.evidence_plan if not match.get("gap")]
        ranked_matches = sorted(
            matched,
            key=lambda item: (
                -int(item.get("match_strength", 0)),
                len(item.get("source_ids", [])),
                item.get("requirement", ""),
            ),
        )

        overall_priorities = []
        for match in ranked_matches[:12]:
            overall_priorities.append(
                {
                    "requirement": match.get("requirement", ""),
                    "match_strength": match.get("match_strength", 0),
                    "source_ids": match.get("source_ids", []),
                    "safe_keywords": match.get("safe_keywords", []),
                }
            )

        section_priorities = {}
        for item in self.evidence_inventory or []:
            section_id = item.get("section_id")
            if not section_id or section_id == "objective":
                continue
            section_matches = [
                {
                    "requirement": match.get("requirement", ""),
                    "match_strength": match.get("match_strength", 0),
                    "safe_keywords": match.get("safe_keywords", []),
                }
                for match in ranked_matches
                if any(source_id == section_id or source_id.startswith(section_id + ":")
                       for source_id in match.get("source_ids", []))
            ][:6]
            if section_matches:
                section_priorities[section_id] = section_matches

        return {
            "overall_priorities": overall_priorities,
            "section_priorities": section_priorities,
            "gaps": [match.get("requirement", "") for match in self.evidence_plan if match.get("gap")][:8],
        }

    def _write_tailored_resume(self, include_objective: bool) -> dict:
        try:
            from dataModels.resume import TailoredResumeWriterOutput
            from services.langchain_helpers import invoke_structured

            result = invoke_structured(
                self.user,
                "RESUME_TAILORED_RESUME_WRITER",
                TailoredResumeWriterOutput,
                timeout_seconds=90.0,
                max_retries=1,
                **self._get_prompt_inputs(include_objective=include_objective),
            )
            tailored = result.final_answer.model_dump() if hasattr(result.final_answer, "model_dump") else result.final_answer
            if not include_objective:
                tailored["objective"] = ""
            return self._preserve_section_identity(tailored)
        except Exception as error:
            logger.error("Tailored resume writer failed: %s", error)
            raise RuntimeError(f"Tailored resume writing failed: {error}") from error

    def _validate_tailored_resume(self, tailored_resume: dict, include_objective: bool) -> dict:
        try:
            from dataModels.resume import ResumeValidationOutput

            result = self._run_grounding_validation(tailored_resume, ResumeValidationOutput)
            expected_ids = set(self._tailored_sections(tailored_resume))
            rejected_ids = self._validated_rejected_ids(result, expected_ids)
            if rejected_ids:
                rejection_feedback = self._validation_feedback_by_section(result, expected_ids)
                if set(rejection_feedback) != rejected_ids:
                    raise ValueError(
                        "Grounding review rejected sections without actionable feedback: "
                        + ", ".join(sorted(rejected_ids))
                    )
                logger.warning("resume_grounding_sections_repair_requested", extra={"rejected.section_ids": sorted(rejected_ids)})
                pre_repair_resume = deepcopy(tailored_resume)
                repaired_resume = self._repair_tailored_resume(tailored_resume, rejection_feedback, include_objective)
                self._restore_approved_sections(pre_repair_resume, repaired_resume, rejected_ids)
                repaired_result = self._run_grounding_validation(repaired_resume, ResumeValidationOutput)
                still_rejected_ids = self._validated_rejected_ids(
                    repaired_result, set(self._tailored_sections(repaired_resume))
                )
                if still_rejected_ids:
                    logger.warning("resume_grounding_fallback_to_source", extra={"rejected.section_ids": sorted(still_rejected_ids)})
                    self._restore_rejected_sections_from_source(repaired_resume, still_rejected_ids)
                    repaired_resume.setdefault("metadata", {})["grounding_fallback_sections"] = sorted(still_rejected_ids)
                tailored_resume = repaired_resume
        except Exception as error:
            logger.error("Grounding validation failed: %s", error)
            raise RuntimeError(f"Final grounding validation failed: {error}") from error
        return tailored_resume

    def _run_grounding_validation(self, tailored_resume: dict, schema):
        from services.langchain_helpers import invoke_structured

        tailored_sections = self._tailored_sections(tailored_resume)
        return invoke_structured(
            self.user,
            "RESUME_GROUNDING_VALIDATOR",
            schema,
            timeout_seconds=75.0,
            max_retries=1,
            **self._get_prompt_inputs(tailored_sections=tailored_sections),
        )

    def _tailored_sections(self, tailored_resume: dict) -> dict:
        experience_indexes = self._source_indexes(
            tailored_resume["experiences"], self.experiences or [],
            ("company", "skip_name", "location", "titles"), "experiences",
        )
        project_indexes = self._source_indexes(
            tailored_resume["projects"], self.projects or [],
            ("name", "technologies", "link", "hyperlink", "show_link"), "projects",
        )
        return {
            "objective": tailored_resume["objective"],
            "skills": tailored_resume["skills"],
            **{f"experience:{source_index}": item for item, source_index in zip(tailored_resume["experiences"], experience_indexes)},
            **{f"project:{source_index}": item for item, source_index in zip(tailored_resume["projects"], project_indexes)},
        }

    @staticmethod
    def _validated_rejected_ids(validation_result, expected_ids: set[str]) -> set[str]:
        approved_ids = set(validation_result.approved_section_ids)
        rejected_ids = set(validation_result.rejected_section_ids)
        if approved_ids & rejected_ids:
            raise ValueError("Grounding review returned overlapping approved and rejected section IDs")
        if approved_ids | rejected_ids != expected_ids:
            raise ValueError("Grounding review did not return one verdict for every tailored section")
        return rejected_ids

    def _preserve_section_identity(self, tailored: dict) -> dict:
        """Keep source roles/projects stable while allowing relevance-based ordering."""
        for section_name, source_items, identity_fields in (
            ("experiences", self.experiences or [], ("company", "skip_name", "location", "titles")),
            ("projects", self.projects or [], ("name", "technologies", "link", "hyperlink", "show_link")),
        ):
            generated_items = tailored.get(section_name, [])
            self._source_indexes(generated_items, source_items, identity_fields, section_name)
        return tailored

    @staticmethod
    def _source_indexes(generated_items: list, source_items: list, identity_fields: tuple, section_name: str) -> list[int]:
        if len(generated_items) != len(source_items):
            raise ValueError(f"Tailored {section_name} must preserve every source section")
        remaining_indexes = set(range(len(source_items)))
        source_indexes = []
        for generated in generated_items:
            candidates = [
                index for index in remaining_indexes
                if all(generated.get(field) == source_items[index].get(field) for field in identity_fields)
            ]
            if len(candidates) != 1:
                raise ValueError(f"Tailored {section_name} changed, duplicated, or omitted a source section")
            source_index = candidates[0]
            remaining_indexes.remove(source_index)
            source_indexes.append(source_index)
        return source_indexes

    def _restore_approved_sections(self, before: dict, after: dict, rejected_ids: set[str]) -> None:
        before_sections = self._tailored_sections(before)
        after_sections = self._tailored_sections(after)
        if set(before_sections) != set(after_sections):
            raise ValueError("Repair changed the tailored resume structure")
        for section_id, original_content in before_sections.items():
            if section_id not in rejected_ids:
                self._replace_section(after, section_id, deepcopy(original_content))

    def _restore_rejected_sections_from_source(self, tailored_resume: dict, rejected_ids: set[str]) -> None:
        for section_id in rejected_ids:
            if section_id == "objective":
                tailored_resume["objective"] = self.objective or ""
            elif section_id == "skills":
                tailored_resume["skills"] = deepcopy(self.skills or [])
            elif section_id.startswith("experience:"):
                source_index = int(section_id.split(":", 1)[1])
                self._replace_section(tailored_resume, section_id, deepcopy((self.experiences or [])[source_index]))
            elif section_id.startswith("project:"):
                source_index = int(section_id.split(":", 1)[1])
                self._replace_section(tailored_resume, section_id, deepcopy((self.projects or [])[source_index]))

    def _replace_section(self, tailored_resume: dict, section_id: str, content) -> None:
        if section_id == "objective":
            tailored_resume["objective"] = content
            return
        if section_id == "skills":
            tailored_resume["skills"] = content
            return
        section_type, source_index_text = section_id.split(":", 1)
        source_index = int(source_index_text)
        section_name = "experiences" if section_type == "experience" else "projects"
        source_items = self.experiences or [] if section_type == "experience" else self.projects or []
        identity_fields = ("company", "skip_name", "location", "titles") if section_type == "experience" else ("name", "technologies", "link", "hyperlink", "show_link")
        output_index = self._source_indexes(tailored_resume[section_name], source_items, identity_fields, section_name).index(source_index)
        tailored_resume[section_name][output_index] = content

    def _validation_feedback_by_section(self, validation_result, valid_ids: set[str]) -> dict[str, str]:
        feedback = {}
        for item in getattr(validation_result, "rejected_sections", []) or []:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            if not isinstance(item, dict):
                continue
            section_id = item.get("section_id")
            reason = item.get("reason")
            if section_id in valid_ids and isinstance(reason, str) and reason.strip():
                feedback[section_id] = " ".join(reason.split())
        return feedback

    def _repair_tailored_resume(self, tailored_resume: dict, rejection_feedback: dict[str, str], include_objective: bool) -> dict:
        try:
            from dataModels.resume import ResumeRepairWriterOutput
            from services.langchain_helpers import invoke_structured

            result = invoke_structured(
                self.user,
                "RESUME_REPAIR_WRITER",
                ResumeRepairWriterOutput,
                timeout_seconds=90.0,
                max_retries=1,
                **self._get_prompt_inputs(
                    include_objective=include_objective,
                    tailored_resume_draft=tailored_resume,
                    validation_feedback=rejection_feedback,
                ),
            )
            repaired = result.final_answer.model_dump() if hasattr(result.final_answer, "model_dump") else result.final_answer
            if not include_objective:
                repaired["objective"] = ""
            repaired = self._preserve_section_identity(repaired)
            tailored_resume["objective"] = repaired["objective"]
            tailored_resume["experiences"] = repaired["experiences"]
            tailored_resume["projects"] = repaired["projects"]
            tailored_resume["skills"] = repaired["skills"]
            tailored_resume.setdefault("metadata", {})["grounding_repair_attempted_sections"] = sorted(rejection_feedback)
            return tailored_resume
        except Exception as error:
            logger.error("Resume repair failed: %s", error)
            raise RuntimeError(f"Resume repair failed: {error}") from error

    async def download_and_parse_job_post(self, url=None):
        if url:
            self.url = url
        downloaded = await self._download_url()
        if not downloaded or not self.job_post_html_data:
            raise ValueError(f"Unable to download job posting from {self.url}")
        self._extract_html_data()
        self.job_post = JobPost(self.job_post_raw, self.user)
        self.parsed_job = await asyncio.to_thread(self.job_post.parse_job_post)

    def _extract_html_data(self):
        try:
            soup = BeautifulSoup(self.job_post_html_data, "html.parser")
            self.job_post_raw = soup.get_text(separator=" ", strip=True)
        except Exception as error:
            logger.error("job_page_text_extraction_failed", extra={"error.reason": str(error)})
            raise

    async def _download_url(self, url=None):
        if url:
            self.url = url

        max_retries = config.get("settings.max_retries", 3)
        backoff_factor = config.get("settings.backoff_factor", 2)
        timeout = httpx.Timeout(20.0, connect=10.0)

        async with httpx.AsyncClient(
            headers=config.get_enhanced_headers(self.url),
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(self.url)
                    response.raise_for_status()
                    self.job_post_html_data = response.text
                    return True
                except httpx.HTTPStatusError as error:
                    status_code = error.response.status_code
                    if status_code not in (429, 999) or attempt == max_retries - 1:
                        logger.error("job_page_download_failed", extra={"error.reason": str(error)})
                        break
                    delay = backoff_factor * 2 ** attempt
                    logger.warning("job_page_rate_limited", extra={"retry.delay_seconds": delay})
                    await asyncio.sleep(delay)
                except httpx.HTTPError as error:
                    logger.error("job_page_download_failed", extra={"error.reason": str(error)})
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff_factor * 2 ** attempt)

        logger.error("job_page_download_retries_exhausted")
        return False

    def _get_prompt_inputs(self, **extra_values):
        from services.langchain_helpers import chain_formatter

        output_dict = {}
        raw_self_data = self.__dict__.copy()
        raw_self_data.update(extra_values)
        raw_self_data["evidence_inventory"] = self.evidence_inventory
        raw_self_data["evidence_map"] = self.evidence_plan
        raw_self_data["tailoring_brief"] = self._tailoring_brief()

        keys = {
            "basic", "objective", "education", "experiences", "projects", "skills",
            "company", "job_title", "job_summary", "original_description", "duties", "qualifications", "ats_keywords",
            "technical_skills", "non_technical_skills", "evidence_inventory", "evidence_map",
            "tailored_sections", "tailored_resume_draft", "validation_feedback", "include_objective",
            "tailoring_brief",
        }
        for key in keys:
            value = raw_self_data.get(key)
            if value is None and self.parsed_job:
                value = self.parsed_job.get(key)
            output_dict[key] = chain_formatter(key, value)
        return output_dict

    def _get_degrees(self, resume: dict):
        result = []
        education = resume.get("education", [])
        for edu in education:
            for degree in edu.get("degrees", []):
                names = degree.get("names", [])
                if isinstance(names, list):
                    result.extend(names)
                elif isinstance(names, str):
                    result.append(names)
        return result

    def dict_to_yaml_string(self, data: dict) -> str:
        try:
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        except YAMLError as error:
            logger.error("resume_yaml_serialization_failed")
            raise ValueError("Could not serialize the generated resume") from error

    def __del__(self):
        self.close()

    def close(self):
        return None
