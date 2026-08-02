import httpx
from bs4 import BeautifulSoup
from dataModels.job_post import JobPost
from config import config
import asyncio
import concurrent.futures
from datetime import datetime
import json
import re
import time
from typing import Dict, List, Optional
import yaml
from yaml import YAMLError

logger = config.getLogger("ResumeImprover")


class ResumeImprover:
    """
    Parallel ResumeImprover using asyncio.gather with run_in_executor for true HTTP parallelism.
    """

    def __init__(self, url, user, parsed_job=None, llm_kwargs: dict = None,
                 timeout: int = 500, progress_callback=None):
        """Initialize ResumeImprover with the job post URL and optional resume location."""
        super().__init__()
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

        # Resume data fields
        self.basic_info = None
        self.education = None
        self.experiences = None
        self.projects = None
        self.skills = None
        self.objective = None
        self.degrees = None
        self.evidence_inventory = []
        self.evidence_plan = []

        # Thread pool for running sync LLM calls.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def create_complete_tailored_resume(self, include_objective) -> str:
        """
        NEW main method: Create complete tailored resume with parallel processing.
        This is what ResumeGenerator calls - does everything.
        """
        try:
            logger.info("resume_tailoring_started", extra={"event.action": "resume_tailoring"})
            if not self.evidence_inventory:
                self._report_progress("evidence_planning", 20, "Matching job requirements to resume evidence")
                self.prepare_tailoring_plan()

            # Try parallel execution first
            try:
                self._report_progress("tailoring_sections", 45, "Tailoring summary, skills, and experience bullets")
                start_time = time.time()

                # Check if we're in an async context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, use thread-based approach
                        logger.info("resume_tailoring_parallel_mode", extra={"execution.mode": "threads"})
                        results = self._generate_content_parallel_threads(include_objective)
                    else:
                        # No active loop, safe to use asyncio.run
                        logger.info("resume_tailoring_parallel_mode", extra={"execution.mode": "asyncio"})
                        results = asyncio.run(self._generate_content_async_parallel(include_objective))
                except RuntimeError:
                    # No event loop, safe to use asyncio.run
                    logger.info("resume_tailoring_parallel_mode", extra={"execution.mode": "asyncio"})
                    results = asyncio.run(self._generate_content_async_parallel(include_objective))

                end_time = time.time()
                logger.info("resume_tailoring_sections_completed", extra={"event.duration_seconds": round(end_time - start_time, 2)})

            except Exception as parallel_error:
                logger.error("resume_tailoring_sections_failed", extra={"error.reason": str(parallel_error)})
                raise RuntimeError(f"Required resume tailoring did not complete: {parallel_error}") from parallel_error

            # Extract results with detailed logging
            objective = results.get('objective', "")
            skills = results.get('skills', [])
            experiences = results.get('experiences', [])
            projects = results.get('projects', [])

            logger.info(
                "resume_tailoring_results_ready",
                extra={
                    "resume.objective_included": bool(objective),
                    "resume.skill_group_count": len(skills),
                    "resume.experience_count": len(experiences),
                    "resume.project_count": len(projects),
                },
            )

            # Step 2: Create final resume
            logger.info("resume_tailoring_assembling")
            final_resume = {
                'editing': False,
                'basic': self.basic_info or {},
                'objective': objective,
                'education': self.education or [],
                'experiences': experiences or [],
                'projects': projects or [],
                'skills': skills or [],
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'job_url': self.url,
                    'tailored': True,
                    'match_report': self._match_report(),
                }
            }

            self._report_progress("grounding_validation", 85, "Validating every tailored claim against the source resume")
            final_resume = self._validate_tailored_resume(final_resume)

            # Step 3: Convert to YAML
            self._report_progress("saving_resume", 95, "Preparing the tailored resume")
            yaml_content = self.dict_to_yaml_string(final_resume)
            logger.info("resume_tailoring_completed")
            return yaml_content

        except Exception:
            raise

    def _report_progress(self, stage: str, progress_percentage: int, message: str) -> None:
        """Publish durable generation progress without coupling tailoring to storage."""
        if self.progress_callback:
            self.progress_callback(stage, progress_percentage, message)

    def prepare_tailoring_plan(self) -> dict:
        """Build the reusable planning state that grounds all later tailoring."""
        self.evidence_inventory = self._build_evidence_inventory()
        self.evidence_plan = self._build_evidence_plan()
        return self.export_tailoring_plan()

    def export_tailoring_plan(self) -> dict:
        """Serialize the current planning state for reuse across generations."""
        return {
            "evidence_inventory": self.evidence_inventory or [],
            "evidence_plan": self.evidence_plan or [],
            "plan_summary": self._match_report(),
        }

    def load_tailoring_plan(self, plan_data: Optional[dict]) -> bool:
        """Hydrate a previously prepared planning state if it is well formed."""
        if not isinstance(plan_data, dict):
            return False
        inventory = plan_data.get("evidence_inventory")
        plan = plan_data.get("evidence_plan")
        if not isinstance(inventory, list) or not isinstance(plan, list):
            return False
        self.evidence_inventory = inventory
        self.evidence_plan = plan
        return True

    def _build_evidence_inventory(self) -> list[dict]:
        """Assign stable IDs to factual source sections before LLM tailoring."""
        inventory = []
        for index, experience in enumerate(self.experiences or []):
            inventory.append({"section_id": f"experience:{index}", "content": experience})
        for index, project in enumerate(self.projects or []):
            inventory.append({"section_id": f"project:{index}", "content": project})
        if self.skills:
            inventory.append({"section_id": "skills", "content": self.skills})
        if self.objective:
            inventory.append({"section_id": "objective", "content": self.objective})
        return inventory

    def _build_evidence_plan(self) -> list[dict]:
        """Match job requirements to source sections; invalid model IDs are discarded."""
        if not self.evidence_inventory or not self.parsed_job:
            return []
        try:
            from dataModels.resume import ResumeEvidencePlanOutput
            from services.langchain_helpers import invoke_structured

            result = invoke_structured(
                self.user,
                "RESUME_EVIDENCE_PLANNER",
                ResumeEvidencePlanOutput,
                # Evidence planning reads the full source inventory. Give it one
                # longer attempt instead of several 60-second retries.
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
            return plan
        except Exception as error:
            logger.error("Evidence planning failed: %s", error)
            raise RuntimeError(f"Evidence planning failed: {error}") from error

    def _match_report(self) -> dict:
        """Persist an honest, UI-friendly summary of match strength and real gaps."""
        matches = [match for match in self.evidence_plan if not match.get("gap")]
        gaps = [match["requirement"] for match in self.evidence_plan if match.get("gap")]
        requirements_evaluated = len(matches) + len(gaps)
        return {
            "matched_requirements": len(matches),
            "requirements_evaluated": requirements_evaluated,
            "evidence_coverage_percentage": round(
                (len(matches) / requirements_evaluated) * 100
            ) if requirements_evaluated else 0,
            "gaps": gaps,
            "strong_matches": [match["requirement"] for match in matches if match.get("match_strength", 0) >= 4],
        }

    def _validate_tailored_resume(self, tailored_resume: dict) -> dict:
        """Attempt targeted correction for rejected sections, then revert only if still invalid."""
        if not self.evidence_inventory:
            return tailored_resume
        try:
            from dataModels.resume import ResumeValidationOutput
            from services.langchain_helpers import invoke_structured

            result = self._run_grounding_validation(tailored_resume, ResumeValidationOutput)
            valid_ids = {item["section_id"] for item in self.evidence_inventory}
            rejected_ids = set(result.rejected_section_ids) & valid_ids
            if rejected_ids:
                rejection_feedback = self._validation_feedback_by_section(result, valid_ids)
                logger.warning("resume_grounding_sections_repair_requested", extra={"rejected.section_ids": sorted(rejected_ids)})
                repaired_resume = self._repair_rejected_sections(tailored_resume, rejection_feedback)
                repair_result = self._run_grounding_validation(repaired_resume, ResumeValidationOutput)
                still_rejected_ids = set(repair_result.rejected_section_ids) & valid_ids
                if still_rejected_ids:
                    logger.warning(
                        "resume_grounding_sections_reverted",
                        extra={"rejected.section_ids": sorted(still_rejected_ids)},
                    )
                    repaired_resume = self._revert_rejected_sections(repaired_resume, still_rejected_ids)
                tailored_resume = repaired_resume
        except Exception as error:
            logger.error("Grounding validation failed: %s", error)
            raise RuntimeError(f"Final grounding validation failed: {error}") from error
        return tailored_resume

    def _run_grounding_validation(self, tailored_resume: dict, schema):
        """Run the grounding validator over the current tailored resume."""
        from services.langchain_helpers import invoke_structured

        tailored_sections = {
            "objective": tailored_resume["objective"],
            "skills": tailored_resume["skills"],
            **{f"experience:{index}": item for index, item in enumerate(tailored_resume["experiences"])},
            **{f"project:{index}": item for index, item in enumerate(tailored_resume["projects"])},
        }
        return invoke_structured(
            self.user,
            "RESUME_GROUNDING_VALIDATOR",
            schema,
            timeout_seconds=75.0,
            max_retries=1,
            **self._get_prompt_inputs(tailored_sections=tailored_sections),
        )

    def _validation_feedback_by_section(self, validation_result, valid_ids: set[str]) -> dict[str, str]:
        """Extract validator repair feedback keyed by section ID."""
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

    def _repair_rejected_sections(self, tailored_resume: dict, rejection_feedback: dict[str, str]) -> dict:
        """Retry only the rejected sections using validator feedback."""
        evidence_by_id = {
            item["section_id"]: item["content"]
            for item in self.evidence_inventory
            if isinstance(item, dict) and item.get("section_id")
        }

        for section_id, reason in rejection_feedback.items():
            source_content = evidence_by_id.get(section_id)
            if source_content is None:
                continue

            if section_id == "objective":
                repaired_objective = self.write_objective(validation_feedback_override=reason)
                tailored_resume["objective"] = repaired_objective or source_content
                continue
            if section_id == "skills":
                repaired_skills = self.extract_matched_skills(validation_feedback_override=reason)
                tailored_resume["skills"] = repaired_skills or source_content
                continue

            try:
                section_kind, raw_index = section_id.split(":", 1)
                section_index = int(raw_index)
            except (ValueError, AttributeError):
                continue

            if section_kind not in {"experience", "project"}:
                continue

            repaired_highlights = self.rewrite_section(
                section=source_content,
                section_id=section_id,
                validation_feedback_override=reason,
            )
            container_key = "experiences" if section_kind == "experience" else "projects"
            if 0 <= section_index < len(tailored_resume.get(container_key, [])):
                updated_section = dict(source_content)
                updated_section["highlights"] = repaired_highlights
                tailored_resume[container_key][section_index] = updated_section

        metadata = tailored_resume.setdefault("metadata", {})
        metadata["grounding_repair_attempted_sections"] = sorted(rejection_feedback)
        return tailored_resume

    def _revert_rejected_sections(self, tailored_resume: dict, rejected_ids: set[str]) -> dict:
        """Restore validator-rejected sections from the original source evidence."""
        evidence_by_id = {
            item["section_id"]: item["content"]
            for item in self.evidence_inventory
            if isinstance(item, dict) and item.get("section_id")
        }

        for section_id in rejected_ids:
            source_content = evidence_by_id.get(section_id)
            if source_content is None:
                continue

            if section_id == "objective":
                tailored_resume["objective"] = source_content
                continue
            if section_id == "skills":
                tailored_resume["skills"] = source_content
                continue

            try:
                section_kind, raw_index = section_id.split(":", 1)
                section_index = int(raw_index)
            except (ValueError, AttributeError):
                continue

            if section_kind == "experience" and 0 <= section_index < len(tailored_resume.get("experiences", [])):
                tailored_resume["experiences"][section_index] = source_content
            elif section_kind == "project" and 0 <= section_index < len(tailored_resume.get("projects", [])):
                tailored_resume["projects"][section_index] = source_content

        metadata = tailored_resume.setdefault("metadata", {})
        metadata["grounding_reverted_sections"] = sorted(rejected_ids)
        return tailored_resume

    async def _generate_content_async_parallel(self, include_objective: bool = True) -> Dict:
        """Generate all resume content in parallel using asyncio.gather."""
        # Create async tasks that run in thread pool (this gives true HTTP parallelism)
        if not hasattr(self, 'executor'):
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        loop = asyncio.get_event_loop()

        logger.info("resume_tailoring_tasks_started", extra={"task.count": 4 if include_objective else 3})

        tasks = []
        task_names = []

        # Conditionally add objective task
        if include_objective:
            tasks.append(loop.run_in_executor(self.executor, self._safe_write_objective))
            task_names.append('objective')

        # Always add these tasks
        tasks.extend([
            loop.run_in_executor(self.executor, self._safe_extract_matched_skills),
            loop.run_in_executor(self.executor, self._safe_rewrite_experiences),
            loop.run_in_executor(self.executor, self._safe_rewrite_projects)
        ])
        task_names.extend(['skills', 'experiences', 'projects'])

        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout
            )
            logger.info("resume_tailoring_tasks_finished", extra={"task.count": len(results)})
        except asyncio.TimeoutError as error:
            logger.error("resume_tailoring_timed_out", extra={"timeout.seconds": self.timeout})
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise RuntimeError("Required resume tailoring timed out") from error

        # Process results with detailed logging
        processed_results = {}

        for i, (result, task_name) in enumerate(zip(results, task_names)):
            if isinstance(result, Exception):
                logger.error("resume_tailoring_task_failed", extra={"task.name": task_name, "error.reason": str(result)})
                raise RuntimeError(f"Required {task_name} tailoring failed: {result}") from result
            else:
                logger.debug("resume_tailoring_task_completed", extra={"task.name": task_name})
                processed_results[task_name] = result

        # If objective was not included, set it to None/empty
        if not include_objective:
            processed_results['objective'] = None

        return processed_results

    def _safe_write_objective(self) -> Optional[str]:
        """Thread-safe wrapper for write_objective."""
        try:
            result = self.write_objective()
            return result
        except Exception as error:
            logger.warning("resume_objective_generation_failed", extra={"error.reason": str(error)})
            return None

    def _safe_extract_matched_skills(self) -> List:
        """Thread-safe wrapper for extract_matched_skills."""
        try:
            result = self.extract_matched_skills()
            return result
        except Exception as error:
            logger.warning("resume_skills_generation_failed", extra={"error.reason": str(error)})
            return self.skills or []

    def _safe_rewrite_experiences(self) -> List:
        """Thread-safe wrapper for rewrite_unedited_experiences."""
        try:
            result = self.rewrite_unedited_experiences()
            return result
        except Exception as error:
            logger.error("resume_experience_rewrite_failed", extra={"error.reason": str(error)})
            raise

    def _safe_rewrite_projects(self) -> List:
        """Thread-safe wrapper for rewrite_unedited_projects."""
        try:
            result = self.rewrite_unedited_projects()
            return result
        except Exception as error:
            logger.error("resume_project_rewrite_failed", extra={"error.reason": str(error)})
            raise

    def _generate_content_parallel_threads(self, include_objective: bool = True) -> Dict:
        """Generate content using ThreadPoolExecutor for cases where we're already in async context."""
        logger.info("resume_tailoring_parallel_mode", extra={"execution.mode": "threads"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks conditionally
            future_to_task = {}

            if include_objective:
                future_to_task[executor.submit(self._safe_write_objective)] = 'objective'

            future_to_task.update({
                executor.submit(self._safe_extract_matched_skills): 'skills',
                executor.submit(self._safe_rewrite_experiences): 'experiences',
                executor.submit(self._safe_rewrite_projects): 'projects'
            })

            results = {}
            completed_tasks = 0
            total_tasks = len(future_to_task)

            # Wait for completion with timeout
            try:
                for future in concurrent.futures.as_completed(future_to_task, timeout=self.timeout):
                    task_name = future_to_task[future]
                    completed_tasks += 1

                    try:
                        result = future.result()
                        results[task_name] = result
                        logger.debug("resume_tailoring_task_completed", extra={"task.name": task_name})

                    except Exception as error:
                        logger.error("resume_tailoring_task_failed", extra={"task.name": task_name, "error.reason": str(error)})
                        raise RuntimeError(f"Required {task_name} tailoring failed: {error}") from error

            except concurrent.futures.TimeoutError:
                logger.error("resume_tailoring_timed_out", extra={"timeout.seconds": self.timeout, "task.completed": completed_tasks, "task.total": total_tasks})

                # Cancel remaining futures
                for future in future_to_task:
                    if not future.done():
                        future.cancel()
                        task_name = future_to_task[future]
                        logger.warning("resume_tailoring_task_cancelled", extra={"task.name": task_name})
                raise RuntimeError("Required resume tailoring timed out")

            # If objective was not included, set it to None
            if not include_objective:
                results['objective'] = None

            logger.info("resume_tailoring_tasks_finished", extra={"task.completed": len(results), "task.total": total_tasks})
            return results

    async def download_and_parse_job_post(self, url=None):
        """Download and parse the job post from the provided URL."""
        if url:
            self.url = url
        downloaded = await self._download_url()
        if not downloaded or not self.job_post_html_data:
            raise ValueError(f"Unable to download job posting from {self.url}")
        self._extract_html_data()
        self.job_post = JobPost(self.job_post_raw, self.user)
        self.parsed_job = await asyncio.to_thread(self.job_post.parse_job_post)

    def _extract_html_data(self):
        """Extract text content from HTML, removing all HTML tags.

        Raises:
            Exception: If HTML data extraction fails.
        """
        try:
            soup = BeautifulSoup(self.job_post_html_data, "html.parser")
            self.job_post_raw = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.error("job_page_text_extraction_failed", extra={"error.reason": str(e)})
            raise

    async def _download_url(self, url=None):
        """Download the content of the URL and return it as a string.

        Args:
            url (str, optional): The URL to download. Defaults to None.

        Returns:
            bool: True if download was successful, False otherwise.
        """
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
                response = None
                try:
                    response = await client.get(self.url)
                    response.raise_for_status()
                    self.job_post_html_data = response.text
                    return True

                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    if status_code not in (429, 999) or attempt == max_retries - 1:
                        logger.error("job_page_download_failed", extra={"error.reason": str(e)})
                        break
                    delay = backoff_factor * 2 ** attempt
                    logger.warning("job_page_rate_limited", extra={"retry.delay_seconds": delay})
                    await asyncio.sleep(delay)
                except httpx.HTTPError as e:
                    logger.error("job_page_download_failed", extra={"error.reason": str(e)})
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff_factor * 2 ** attempt)

        logger.error("job_page_download_retries_exhausted")
        return False

    def write_objective(self, **chain_kwargs) -> str:
        """Write an objective for the resume."""
        try:
            from dataModels.resume import ResumeSummarizerOutput
            from services.langchain_helpers import invoke_structured

            prompt_inputs = self._get_prompt_inputs()
            if chain_kwargs.get("validation_feedback_override"):
                prompt_inputs["validation_feedback"] = chain_kwargs["validation_feedback_override"]
            result = invoke_structured(
                self.user,
                "OBJECTIVE_WRITER",
                ResumeSummarizerOutput,
                timeout_seconds=45.0,
                max_retries=1,
                **prompt_inputs,
            )
            if result:
                # Handle both Pydantic model and dictionary responses
                if hasattr(result, 'final_answer'):
                    # Pydantic model
                    objective = result.final_answer
                    logger.debug("resume_objective_response_parsed", extra={"response.type": "pydantic"})
                elif isinstance(result, dict):
                    # Dictionary response
                    objective = result.get('final_answer')
                    logger.debug("resume_objective_response_parsed", extra={"response.type": "dictionary"})
                else:
                    # Direct string response
                    objective = result
                    logger.debug("resume_objective_response_parsed", extra={"response.type": "string"})

                objective = self._validated_summary(objective)
                logger.debug("resume_objective_generated", extra={"resume.objective_included": bool(objective)})
                return objective

            logger.warning("resume_objective_generation_empty")
            return None

        except Exception as error:
            logger.warning("resume_objective_generation_failed", extra={"error.reason": str(error)})
            return None

    def extract_matched_skills(self, **chain_kwargs) -> list:
        """Extract matched skills from the resume and job post with LLM handling deduplication."""
        try:
            from dataModels.resume import ResumeSkillsMatcherOutput
            from services.langchain_helpers import invoke_structured

            prompt_inputs = self._get_prompt_inputs()
            if chain_kwargs.get("validation_feedback_override"):
                prompt_inputs["validation_feedback"] = chain_kwargs["validation_feedback_override"]

            extracted_skills = invoke_structured(
                self.user,
                "SKILLS_MATCHER",
                ResumeSkillsMatcherOutput,
                timeout_seconds=45.0,
                max_retries=1,
                **prompt_inputs,
            )

            if not extracted_skills:
                logger.warning("resume_skills_generation_empty")
                return self.skills or []

            # Handle both Pydantic model and dictionary responses
            if hasattr(extracted_skills, 'final_answer'):
                # Pydantic model
                extracted_skills_dict = extracted_skills.final_answer
            elif isinstance(extracted_skills, dict):
                # Dictionary response
                extracted_skills_dict = extracted_skills.get("final_answer", {})
            else:
                logger.error("resume_skills_response_invalid", extra={"response.type": type(extracted_skills).__name__})
                return self.skills or []

            logger.debug("resume_skills_response_received")

            # Build the final skills structure - LLM has already handled deduplication
            result = []

            # Handle technical skills - support both Pydantic model and dict
            if hasattr(extracted_skills_dict, 'technical_skills'):
                # Pydantic model
                technical_skills = extracted_skills_dict.technical_skills or {}
            elif isinstance(extracted_skills_dict, dict):
                # Dictionary
                technical_skills = extracted_skills_dict.get("technical_skills", {})
            else:
                technical_skills = {}

            if technical_skills and isinstance(technical_skills, dict):
                # Convert to subcategories format
                subcategories = []
                for category_name, skills_list in technical_skills.items():
                    if skills_list:  # Only add non-empty categories
                        subcategories.append({
                            "name": category_name,
                            "skills": skills_list
                        })

                if subcategories:
                    result.append({
                        "category": "Technical",
                        "subcategories": subcategories
                    })

            # Handle non-technical skills - support both Pydantic model and dict
            if hasattr(extracted_skills_dict, 'non_technical_skills'):
                # Pydantic model
                non_technical_skills = extracted_skills_dict.non_technical_skills or []
            elif isinstance(extracted_skills_dict, dict):
                # Dictionary
                non_technical_skills = extracted_skills_dict.get("non_technical_skills", [])
            else:
                non_technical_skills = []

            if non_technical_skills and isinstance(non_technical_skills, list):
                result.append({
                    "category": "Non-technical",
                    "skills": non_technical_skills
                })

            logger.info("resume_skills_generated", extra={"resume.skill_group_count": len(result)})

            return self._normalized_skill_groups(result)

        except Exception as error:
            logger.warning("resume_skills_generation_failed", extra={"error.reason": str(error)})
            return self.skills or []

    def rewrite_unedited_experiences(self, **chain_kwargs) -> list:
        """Rewrite unedited experiences in the resume."""
        try:
            if not self.experiences:
                logger.info("resume_experience_rewrite_skipped")
                return []

            logger.info("resume_experience_rewrite_started", extra={"resume.experience_count": len(self.experiences)})
            result = self._rewrite_sections_batch(self.experiences, "experience", **chain_kwargs)

            logger.info("resume_experience_rewrite_completed", extra={"resume.experience_count": len(result)})
            return result
        except Exception as error:
            logger.error("resume_experience_rewrite_failed", extra={"error.reason": str(error)})
            raise RuntimeError(f"Experience bullet rewriting failed: {error}") from error

    def rewrite_unedited_projects(self, **chain_kwargs) -> list:
        """Rewrite unedited projects in the resume."""
        try:
            if not self.projects:
                logger.info("resume_project_rewrite_skipped")
                return []

            logger.info("resume_project_rewrite_started", extra={"resume.project_count": len(self.projects)})
            result = self._rewrite_sections_batch(self.projects, "project", **chain_kwargs)

            logger.info("resume_project_rewrite_completed", extra={"resume.project_count": len(result)})
            return result
        except Exception as error:
            logger.error("resume_project_rewrite_failed", extra={"error.reason": str(error)})
            raise RuntimeError(f"Project bullet rewriting failed: {error}") from error

    def _rewrite_sections_batch(self, sections: list, section_kind: str, **chain_kwargs) -> list:
        """Rewrite same-type sections in one LLM call, with per-section fallback on failure."""
        prepared_sections = []
        rewritten_sections = [None] * len(sections)

        for index, raw_section in enumerate(sections):
            section = dict(raw_section)
            original_highlights = section.get("highlights", [])
            section_id = f"{section_kind}:{index}"
            logger.debug(
                "resume_section_rewrite_started",
                extra={"section.id": section_id, "source.highlight_count": len(original_highlights)},
            )
            if not original_highlights:
                rewritten_sections[index] = section
                continue
            prepared_sections.append((index, section, section_id, original_highlights))

        if not prepared_sections:
            return rewritten_sections

        try:
            rewritten_map = self.rewrite_sections_batch(
                [
                    {
                        "section_id": section_id,
                        "section": section,
                        "original_highlights": original_highlights,
                    }
                    for _, section, section_id, original_highlights in prepared_sections
                ],
                **chain_kwargs,
            )
            for index, section, section_id, _ in prepared_sections:
                updated_section = dict(section)
                updated_section["highlights"] = rewritten_map[section_id]
                rewritten_sections[index] = updated_section
            return rewritten_sections
        except Exception as error:
            logger.warning("resume_section_batch_rewrite_failed", extra={"section.kind": section_kind, "error.reason": str(error)})
            return self._rewrite_sections_parallel(sections, section_kind, **chain_kwargs)

    def _rewrite_sections_parallel(self, sections: list, section_kind: str, **chain_kwargs) -> list:
        """Rewrite same-type resume sections concurrently while preserving order."""
        rewritten_sections = [None] * len(sections)
        sections_to_rewrite = []

        for index, raw_section in enumerate(sections):
            section = dict(raw_section)
            original_highlights = section.get("highlights", [])
            section_id = f"{section_kind}:{index}"
            logger.debug(
                "resume_section_rewrite_started",
                extra={"section.id": section_id, "source.highlight_count": len(original_highlights)},
            )
            if not original_highlights:
                rewritten_sections[index] = section
                continue
            sections_to_rewrite.append((index, section, section_id))

        if not sections_to_rewrite:
            return rewritten_sections

        max_workers = min(4, len(sections_to_rewrite))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_section = {
                executor.submit(self.rewrite_section, section=section, section_id=section_id, **chain_kwargs): (index, section, section_id)
                for index, section, section_id in sections_to_rewrite
            }
            for future in concurrent.futures.as_completed(future_to_section):
                index, section, section_id = future_to_section[future]
                try:
                    updated_section = dict(section)
                    updated_section["highlights"] = future.result()
                    rewritten_sections[index] = updated_section
                except Exception as error:
                    logger.error("resume_section_parallel_rewrite_failed", extra={"section.id": section_id, "error.reason": str(error)})
                    raise

        return rewritten_sections

    def rewrite_sections_batch(self, section_payloads: list[dict], **chain_kwargs) -> dict[str, list[str]]:
        """Rewrite multiple sections in one structured call and validate each mapping."""
        try:
            from dataModels.resume import ResumeSectionBatchHighlighterOutput
            from services.langchain_helpers import invoke_structured

            prompt_inputs = self._get_prompt_inputs(
                section_batch=self._format_section_batch(section_payloads),
            )
            last_error = None
            for rewrite_attempt in range(1, 3):
                prompt_inputs["rewrite_attempt"] = rewrite_attempt
                prompt_inputs["validation_feedback"] = str(last_error or "None; produce the complete mapping for every section.")
                batch_revised = invoke_structured(
                    self.user,
                    "SECTION_BATCH_HIGHLIGHTER",
                    ResumeSectionBatchHighlighterOutput,
                    timeout_seconds=75.0,
                    max_retries=1,
                    **prompt_inputs,
                )

                if hasattr(batch_revised, "final_answer"):
                    section_items = batch_revised.final_answer or []
                    raw_result = [item.model_dump() for item in section_items]
                elif isinstance(batch_revised, dict):
                    raw_result = batch_revised.get("final_answer", [])
                else:
                    last_error = ValueError("The model returned no structured batched highlights")
                    continue

                try:
                    return self._validated_batched_highlights(raw_result, section_payloads)
                except ValueError as error:
                    last_error = error
                    logger.warning(
                        "resume_section_batch_rewrite_validation_failed",
                        extra={"rewrite.attempt": rewrite_attempt, "error.reason": str(error)},
                    )

            raise last_error or ValueError("The model returned no rewritten batched highlights")
        except Exception as error:
            raise RuntimeError(f"Batched section rewriting failed: {error}") from error

    def rewrite_section(self, section, section_id: str = "", **chain_kwargs) -> list:
        """Rewrite a section of the resume."""
        original_highlights = section.get("highlights", [])
        try:
            from dataModels.resume import ResumeSectionHighlighterOutput
            from services.langchain_helpers import invoke_structured

            logger.debug("resume_section_rewrite_requested", extra={"section.id": section_id})

            prompt_inputs = self._get_prompt_inputs(
                section=section,
                section_id=section_id,
                required_highlight_count=len(original_highlights),
                source_highlights="\n".join(
                    f"{index}. {highlight}"
                    for index, highlight in enumerate(original_highlights, start=1)
                ),
                highlight_word_limits="\n".join(
                    f"{index}: 3-{self._highlight_word_limit(highlight)} words"
                    for index, highlight in enumerate(original_highlights, start=1)
                ),
            )
            feedback_override = chain_kwargs.get("validation_feedback_override")
            last_error = None
            for rewrite_attempt in range(1, 3):
                prompt_inputs["rewrite_attempt"] = rewrite_attempt
                prompt_inputs["validation_feedback"] = str(last_error or feedback_override or "None; produce the complete mapping.")
                logger.debug("resume_section_rewrite_attempt", extra={"section.id": section_id, "rewrite.attempt": rewrite_attempt})
                section_revised = invoke_structured(
                    self.user,
                    "SECTION_HIGHLIGHTER",
                    ResumeSectionHighlighterOutput,
                    timeout_seconds=60.0,
                    max_retries=1,
                    **prompt_inputs,
                )

                if hasattr(section_revised, "final_answer"):
                    highlights = section_revised.final_answer or []
                    result = [item.model_dump() for item in highlights]
                elif isinstance(section_revised, dict):
                    highlights = section_revised.get("final_answer", [])
                    result = highlights
                else:
                    last_error = ValueError("The model returned no structured highlights")
                    continue

                try:
                    validated_highlights = self._validated_highlights(result, original_highlights)
                    ranked_highlights = sorted(
                        validated_highlights,
                        key=lambda item: (-item["relevance"], item["source_index"]),
                    )
                    strongest_score = ranked_highlights[0]["relevance"]
                    relevance_threshold = max(1, strongest_score - 1)
                    selected_highlights = [
                        item for item in ranked_highlights
                        if item["relevance"] >= relevance_threshold
                    ]
                    logger.info(
                        "resume_section_rewrite_completed",
                        extra={
                            "section.id": section_id,
                            "source.highlight_count": len(original_highlights),
                            "resume.highlight_count": len(selected_highlights),
                            "relevance.threshold": relevance_threshold,
                        },
                    )
                    return [item["highlight"] for item in selected_highlights]
                except ValueError as error:
                    last_error = error
                    logger.warning(
                        "resume_section_rewrite_validation_failed",
                        extra={
                            "section.id": section_id,
                            "rewrite.attempt": rewrite_attempt,
                            "error.reason": str(error),
                        },
                    )

            raise last_error or ValueError("The model returned no rewritten highlights")

        except Exception as error:
            logger.error("resume_section_rewrite_failed", extra={"section.id": section_id, "error.reason": str(error)})
            raise RuntimeError(f"Section bullet rewriting failed: {error}") from error

    def _validated_summary(self, summary) -> Optional[str]:
        """Keep summaries concise and avoid replacing a usable original with bad output."""
        if not isinstance(summary, str):
            return self.objective or None
        summary = " ".join(summary.split())
        if not summary or len(re.findall(r"\b\w+\b", summary)) > 55:
            logger.warning("resume_objective_generation_invalid")
            return self.objective or None
        return summary

    def _validated_batched_highlights(self, batch_candidates, section_payloads) -> dict[str, list[str]]:
        """Require complete, valid batched rewrites for every supplied section."""
        payload_by_id = {payload["section_id"]: payload for payload in section_payloads}
        accepted_sections = {}
        errors = []

        for item in batch_candidates or []:
            if not isinstance(item, dict):
                errors.append("batch item is not an object")
                continue
            section_id = item.get("section_id")
            if section_id not in payload_by_id:
                errors.append(f"unexpected section_id {section_id}")
                continue
            if section_id in accepted_sections:
                errors.append(f"section_id {section_id} is duplicated")
                continue
            highlights = item.get("highlights")
            validated = self._validated_highlights(
                highlights,
                payload_by_id[section_id]["original_highlights"],
            )
            ranked_highlights = sorted(
                validated,
                key=lambda candidate: (-candidate["relevance"], candidate["source_index"]),
            )
            strongest_score = ranked_highlights[0]["relevance"]
            relevance_threshold = max(1, strongest_score - 1)
            selected_highlights = [
                candidate["highlight"]
                for candidate in ranked_highlights
                if candidate["relevance"] >= relevance_threshold
            ]
            accepted_sections[section_id] = selected_highlights

        missing_section_ids = sorted(set(payload_by_id) - set(accepted_sections))
        if missing_section_ids:
            errors.append(f"missing section_ids {missing_section_ids}")
        if errors:
            raise ValueError("; ".join(errors))
        return accepted_sections

    def _format_section_batch(self, section_payloads: list[dict]) -> str:
        """Render multiple source sections into one prompt payload for batched rewriting."""
        rendered_sections = []
        for payload in section_payloads:
            original_highlights = payload["original_highlights"]
            section_id = payload["section_id"]
            section_evidence = [
                match for match in self.evidence_plan
                if section_id in match.get("source_ids", [])
            ]
            rendered_sections.append(
                "\n".join(
                    [
                        f"<Section ID>\n{section_id}",
                        f"<Candidate source section>\n{yaml.safe_dump(payload['section'], sort_keys=False, allow_unicode=True)}".strip(),
                        "<Numbered source highlights>",
                        "\n".join(
                            f"{index}. {highlight}"
                            for index, highlight in enumerate(original_highlights, start=1)
                        ),
                        "<Per-source bullet word limits>",
                        "\n".join(
                            f"{index}: 3-{self._highlight_word_limit(highlight)} words"
                            for index, highlight in enumerate(original_highlights, start=1)
                        ),
                        f"<Required highlight count>\n{len(original_highlights)}",
                        f"<Supported matches for this section>\n{json.dumps(section_evidence, ensure_ascii=False)}",
                    ]
                )
            )
        return "\n\n".join(rendered_sections)

    def _validated_highlights(self, candidates, originals) -> list:
        """Require a complete, materially changed set of grounded bullet rewrites."""
        expected_count = len(originals or [])
        expected_indexes = set(range(1, expected_count + 1))
        accepted, seen, received_indexes = {}, set(), set()
        rejection_reasons = []
        for response_position, candidate in enumerate(candidates or [], start=1):
            if not isinstance(candidate, dict):
                rejection_reasons.append(f"response item {response_position} is not an object")
                continue
            source_index = candidate.get("source_index")
            highlight = candidate.get("highlight")
            relevance = candidate.get("relevance")
            if not isinstance(source_index, int) or source_index not in expected_indexes:
                rejection_reasons.append(f"response item {response_position} has an invalid source_index")
                continue
            if source_index in received_indexes:
                rejection_reasons.append(f"source_index {source_index} is duplicated")
                continue
            if not isinstance(highlight, str):
                rejection_reasons.append(f"source_index {source_index} has no text highlight")
                continue
            if not isinstance(relevance, int) or not 1 <= relevance <= 5:
                rejection_reasons.append(f"source_index {source_index} has an invalid relevance score")
                continue
            candidate = " ".join(highlight.split())
            word_count = len(re.findall(r"\b\w+\b", candidate))
            normalized = candidate.casefold()
            original_normalized = " ".join(originals[source_index - 1].split()).casefold()
            max_words = self._highlight_word_limit(originals[source_index - 1])
            if not 3 <= word_count <= max_words:
                rejection_reasons.append(
                    f"source_index {source_index} has {word_count} words; it must have 3-{max_words}"
                )
                continue
            if normalized in seen:
                rejection_reasons.append(f"source_index {source_index} duplicates another rewritten bullet")
                continue
            if normalized == original_normalized:
                rejection_reasons.append(f"source_index {source_index} repeats the source bullet verbatim")
                continue
            accepted[source_index] = {
                "source_index": source_index,
                "highlight": candidate,
                "relevance": relevance,
            }
            seen.add(normalized)
            received_indexes.add(source_index)

        if set(accepted) != expected_indexes:
            raise ValueError(
                "The model must return one distinct, valid rewrite mapped to every source highlight "
                f"(expected indexes {sorted(expected_indexes)}, received {len(candidates or [])}, "
                f"accepted indexes {sorted(accepted)}; rejections: {'; '.join(rejection_reasons) or 'missing source indexes'})"
            )
        return [accepted[index] for index in range(1, expected_count + 1)]

    @staticmethod
    def _highlight_word_limit(source_highlight: str) -> int:
        """Allow enough room to preserve a long factual source bullet without unbounded output."""
        source_words = len(re.findall(r"\b\w+\b", source_highlight or ""))
        return min(64, max(40, source_words + 8))

    def _normalized_skill_groups(self, groups: list) -> list:
        """Defensively deduplicate and bound LLM-generated skill output."""
        normalized_groups, seen = [], set()
        technical_count = 0
        for group in groups:
            if group.get("category") == "Technical":
                subcategories = []
                for subcategory in group.get("subcategories", [])[:4]:
                    skills = []
                    for skill in subcategory.get("skills", []):
                        if not isinstance(skill, str):
                            continue
                        skill = " ".join(skill.split())
                        normalized = skill.casefold()
                        if skill and normalized not in seen and technical_count < 25:
                            skills.append(skill)
                            seen.add(normalized)
                            technical_count += 1
                    if skills:
                        subcategories.append({"name": subcategory.get("name", "Other"), "skills": skills})
                if subcategories:
                    normalized_groups.append({"category": "Technical", "subcategories": subcategories})
            elif group.get("category") == "Non-technical":
                skills = []
                for skill in group.get("skills", []):
                    if isinstance(skill, str) and skill.strip() and skill.casefold() not in seen:
                        skills.append(" ".join(skill.split()))
                        seen.add(skill.casefold())
                if skills:
                    normalized_groups.append({"category": "Non-technical", "skills": skills})
        return normalized_groups

    def _get_prompt_inputs(self, section=None, section_id: str = "", **extra_values):
        """Format the full, provider-neutral prompt context for resume generation."""
        from services.langchain_helpers import chain_formatter

        output_dict = {}
        raw_self_data = self.__dict__
        if section is not None:
            raw_self_data = raw_self_data.copy()
            raw_self_data["section"] = section

        raw_self_data.update(extra_values)
        if section_id:
            raw_self_data["section_evidence"] = [
                match for match in self.evidence_plan if section_id in match.get("source_ids", [])
            ]
        raw_self_data["evidence_inventory"] = self.evidence_inventory
        raw_self_data["evidence_map"] = self.evidence_plan

        keys = {
            "section", "objective", "experiences", "projects", "skills",
            "company", "job_summary", "duties", "qualifications", "ats_keywords",
            "technical_skills", "non_technical_skills", "evidence_inventory", "evidence_map",
            "section_evidence", "section_batch", "tailored_sections", "required_highlight_count", "rewrite_attempt",
            "source_highlights", "highlight_word_limits", "validation_feedback",
        }
        for key in keys:
            value = raw_self_data.get(key)
            if value is None and self.parsed_job:
                value = self.parsed_job.get(key)
            output_dict[key] = chain_formatter(key, value)
        return output_dict

    def _get_degrees(self, resume: dict):
        """Extract degrees from the resume."""
        result = []
        education = resume.get('education', [])
        for edu in education:
            degrees = edu.get('degrees', [])
            for degree in degrees:
                names = degree.get('names', [])
                if isinstance(names, list):
                    result.extend(names)
                elif isinstance(names, str):
                    result.append(names)
        return result

    def dict_to_yaml_string(self, data: dict) -> str:
        """Serialize the completed resume for the existing persistence layer."""
        try:
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        except YAMLError as error:
            logger.error("resume_yaml_serialization_failed")
            raise ValueError("Could not serialize the generated resume") from error

    def __del__(self):
        self.close()

    def close(self):
        """Release the LLM worker pool deterministically after a generation."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False, cancel_futures=True)
            del self.executor
