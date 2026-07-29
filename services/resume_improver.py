import httpx
from bs4 import BeautifulSoup
from dataModels.job_post import JobPost
from config import config
import asyncio
import concurrent.futures
from datetime import datetime
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

    def __init__(self, url, user, parsed_job=None, llm_kwargs: dict = None, timeout: int = 500):
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

        # Thread pool for running sync LLM calls
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def create_complete_tailored_resume(self, include_objective) -> str:
        """
        NEW main method: Create complete tailored resume with parallel processing.
        This is what ResumeGenerator calls - does everything.
        """
        try:
            logger.info("=== Creating Complete Tailored Resume (Parallel) ===")

            # Establish the factual boundary before any section is rewritten.
            self.evidence_inventory = self._build_evidence_inventory()
            self.evidence_plan = self._build_evidence_plan()

            # Try parallel execution first
            try:
                start_time = time.time()

                # Check if we're in an async context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, use thread-based approach
                        logger.info(f'Using thread-based parallel approach (async context detected) using model: {self.user.model}')
                        results = self._generate_content_parallel_threads(include_objective)
                    else:
                        # No active loop, safe to use asyncio.run
                        logger.info("Using asyncio-based parallel approach")
                        results = asyncio.run(self._generate_content_async_parallel(include_objective))
                except RuntimeError:
                    # No event loop, safe to use asyncio.run
                    logger.info("Using asyncio-based parallel approach (no existing loop)")
                    results = asyncio.run(self._generate_content_async_parallel(include_objective))

                end_time = time.time()
                logger.info(f"Parallel generation completed in {end_time - start_time:.2f} seconds")

            except Exception as parallel_error:
                logger.warning(f"Parallel execution failed: {parallel_error}, falling back to sequential")
                # Fallback to sequential execution
                results = self._generate_content_sequential()

            # Extract results with detailed logging
            objective = results.get('objective', "")
            skills = results.get('skills', [])
            experiences = results.get('experiences', [])
            projects = results.get('projects', [])

            logger.info(f"Results summary:")
            logger.info(f"  - Objective: {'✓' if objective else '✗'}")
            logger.info(f"  - Skills: {len(skills)} categories")
            logger.info(f"  - Experiences: {len(experiences)} items")
            logger.info(f"  - Projects: {len(projects)} items")

            # Step 2: Create final resume
            logger.info("Assembling final resume...")
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

            final_resume = self._validate_tailored_resume(final_resume)

            # Step 3: Convert to YAML
            yaml_content = self.dict_to_yaml_string(final_resume)
            logger.info("=== Resume Creation Complete ===")
            return yaml_content

        except Exception as e:
            logger.error(f"Complete resume creation failed: {e}")
            raise

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
                timeout_seconds=120.0,
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
            logger.exception("Evidence planning failed; refusing to generate an unplanned resume")
            raise RuntimeError(
                "Evidence planning failed; resume generation was not started. Please retry."
            ) from error

    def _match_report(self) -> dict:
        """Persist an honest, UI-friendly summary of match strength and real gaps."""
        matches = [match for match in self.evidence_plan if not match.get("gap")]
        gaps = [match["requirement"] for match in self.evidence_plan if match.get("gap")]
        return {
            "matched_requirements": len(matches),
            "gaps": gaps,
            "strong_matches": [match["requirement"] for match in matches if match.get("match_strength", 0) >= 4],
        }

    def _validate_tailored_resume(self, tailored_resume: dict) -> dict:
        """Revert whole sections when a final factual-grounding review rejects them."""
        if not self.evidence_inventory:
            return tailored_resume
        try:
            from dataModels.resume import ResumeValidationOutput
            from services.langchain_helpers import invoke_structured

            tailored_sections = {
                "objective": tailored_resume["objective"],
                "skills": tailored_resume["skills"],
                **{f"experience:{index}": item for index, item in enumerate(tailored_resume["experiences"])},
                **{f"project:{index}": item for index, item in enumerate(tailored_resume["projects"])},
            }
            result = invoke_structured(
                self.user,
                "RESUME_GROUNDING_VALIDATOR",
                ResumeValidationOutput,
                **self._get_prompt_inputs(tailored_sections=tailored_sections),
            )
            originals = {item["section_id"]: item["content"] for item in self.evidence_inventory}
            rejected_ids = set(result.rejected_section_ids) & set(originals)
            for section_id in rejected_ids:
                if section_id == "objective":
                    tailored_resume["objective"] = originals[section_id]
                elif section_id == "skills":
                    tailored_resume["skills"] = originals[section_id]
                else:
                    section_type, index = section_id.split(":", 1)
                    section_key = f"{section_type}s"
                    tailored_resume[section_key][int(index)] = originals[section_id]
            tailored_resume["metadata"]["grounding_rejections"] = sorted(rejected_ids)
        except Exception as error:
            logger.warning(f"Grounding validation failed; retaining generated sections with source safeguards: {error}")
        return tailored_resume

    async def _generate_content_async_parallel(self, include_objective: bool = True) -> Dict:
        """Generate all resume content in parallel using asyncio.gather."""
        # Create async tasks that run in thread pool (this gives true HTTP parallelism)
        if not hasattr(self, 'executor'):
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        loop = asyncio.get_event_loop()

        logger.info("Starting parallel task execution...")

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
            logger.info(f"All {len(results)} tasks completed")
        except asyncio.TimeoutError:
            logger.error(f"Parallel generation timed out after {self.timeout} seconds")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Use default values
            default_results = []
            if include_objective:
                default_results.append(None)
            default_results.extend([[], [], []])
            results = default_results

        # Process results with detailed logging
        processed_results = {}

        for i, (result, task_name) in enumerate(zip(results, task_names)):
            if isinstance(result, Exception):
                logger.error(f"Task '{task_name}' failed with exception: {result}")
                processed_results[task_name] = self._get_default_value(task_name)
            else:
                logger.info(f"Task '{task_name}' completed successfully")
                processed_results[task_name] = result

        # If objective was not included, set it to None/empty
        if not include_objective:
            processed_results['objective'] = None

        return processed_results

    def _safe_write_objective(self) -> Optional[str]:
        """Thread-safe wrapper for write_objective."""
        try:
            logger.debug("Starting objective generation...")
            result = self.write_objective()
            logger.debug(f"Objective generation result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in parallel objective generation: {e}")
            import traceback
            logger.error(f"Objective traceback: {traceback.format_exc()}")
            return None

    def _safe_extract_matched_skills(self) -> List:
        """Thread-safe wrapper for extract_matched_skills."""
        try:
            logger.debug("Starting skills extraction...")
            result = self.extract_matched_skills()
            logger.debug(f"Skills extraction result: {len(result) if result else 0} categories")
            return result
        except Exception as e:
            logger.error(f"Error in parallel skills extraction: {e}")
            import traceback
            logger.error(f"Skills traceback: {traceback.format_exc()}")
            return self.skills or []

    def _safe_rewrite_experiences(self) -> List:
        """Thread-safe wrapper for rewrite_unedited_experiences."""
        try:
            logger.debug("Starting experience rewriting...")
            result = self.rewrite_unedited_experiences()
            logger.debug(f"Experience rewriting result: {len(result) if result else 0} experiences")
            return result
        except Exception as e:
            logger.error(f"Error in parallel experience rewriting: {e}")
            import traceback
            logger.error(f"Experience traceback: {traceback.format_exc()}")
            return self.experiences or []

    def _safe_rewrite_projects(self) -> List:
        """Thread-safe wrapper for rewrite_unedited_projects."""
        try:
            logger.debug("Starting project rewriting...")
            result = self.rewrite_unedited_projects()
            logger.debug(f"Project rewriting result: {len(result) if result else 0} projects")
            return result
        except Exception as e:
            logger.error(f"Error in parallel project rewriting: {e}")
            import traceback
            logger.error(f"Project traceback: {traceback.format_exc()}")
            return self.projects or []

    def _generate_content_parallel_threads(self, include_objective: bool = True) -> Dict:
        """Generate content using ThreadPoolExecutor for cases where we're already in async context."""
        logger.info("Using ThreadPoolExecutor for parallel generation...")

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
                        logger.info(f"✓ {task_name} completed successfully ({completed_tasks}/{total_tasks})")

                        # Debug log the result
                        if task_name == 'experiences':
                            logger.debug(f"Experiences result: {len(result) if result else 0} items")
                        elif task_name == 'projects':
                            logger.debug(f"Projects result: {len(result) if result else 0} items")

                    except Exception as e:
                        logger.error(f"✗ {task_name} failed with exception: {e}")
                        import traceback
                        logger.error(f"{task_name} traceback: {traceback.format_exc()}")
                        results[task_name] = self._get_default_value(task_name)

            except concurrent.futures.TimeoutError:
                logger.error(f"Parallel generation timed out after {self.timeout} seconds")
                logger.error(f"Completed {completed_tasks}/{total_tasks} tasks before timeout")

                # Cancel remaining futures
                for future in future_to_task:
                    if not future.done():
                        future.cancel()
                        task_name = future_to_task[future]
                        logger.warning(f"Cancelled task: {task_name}")

                # Fill in defaults for missing results
                expected_tasks = ['skills', 'experiences', 'projects']
                if include_objective:
                    expected_tasks.insert(0, 'objective')

                for task_name in expected_tasks:
                    if task_name not in results:
                        results[task_name] = self._get_default_value(task_name)
                        logger.warning(f"Using default value for {task_name} due to timeout")

            # If objective was not included, set it to None
            if not include_objective:
                results['objective'] = None

            logger.info(f"Thread-based parallel execution completed: {len(results)}/{total_tasks} tasks")
            return results

    def _generate_content_sequential(self, include_objective: bool = True) -> Dict:
        """Fallback sequential content generation."""
        logger.info("Running sequential content generation...")

        results = {}

        if include_objective:
            logger.info("Sequential: Generating objective...")
            results['objective'] = self._safe_write_objective()
        else:
            results['objective'] = None

        logger.info("Sequential: Extracting skills...")
        results['skills'] = self._safe_extract_matched_skills()

        logger.info("Sequential: Rewriting experiences...")
        results['experiences'] = self._safe_rewrite_experiences()

        logger.info("Sequential: Rewriting projects...")
        results['projects'] = self._safe_rewrite_projects()

        logger.info("Sequential content generation completed")
        return results

    def _get_default_value(self, task_name: str):
        """Get default value for a task that failed or timed out."""
        defaults = {
            'objective': None,
            'skills': self.skills or [],
            'experiences': self.experiences or [],
            'projects': self.projects or []
        }
        default_value = defaults.get(task_name)
        logger.debug(
            f"Using default for {task_name}: {type(default_value)} with {len(default_value) if isinstance(default_value, list) else 'N/A'} items")
        return default_value

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
            logger.error(f"Failed to extract HTML data: {e}")
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
                        logger.error(f"Failed to download URL {self.url}: {e}")
                        break
                    delay = backoff_factor * 2 ** attempt
                    logger.warning(f"Job site rate-limited the request; retrying in {delay} seconds")
                    await asyncio.sleep(delay)
                except httpx.HTTPError as e:
                    logger.error(f"Failed to download URL {self.url}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff_factor * 2 ** attempt)

        logger.error(f"Exceeded maximum retries for URL {self.url}")
        return False

    def write_objective(self, **chain_kwargs) -> str:
        """Write an objective for the resume."""
        try:
            from dataModels.resume import ResumeSummarizerOutput
            from services.langchain_helpers import invoke_structured

            prompt_inputs = self._get_prompt_inputs()
            result = invoke_structured(
                self.user, "OBJECTIVE_WRITER", ResumeSummarizerOutput, **prompt_inputs
            )
            if result:
                # Handle both Pydantic model and dictionary responses
                if hasattr(result, 'final_answer'):
                    # Pydantic model
                    objective = result.final_answer
                    logger.info("Using Pydantic model access")
                elif isinstance(result, dict):
                    # Dictionary response
                    objective = result.get('final_answer')
                    logger.info("Using dictionary access")
                else:
                    # Direct string response
                    objective = result
                    logger.info("Using direct response")

                objective = self._validated_summary(objective)
                logger.debug(f"Objective result: {objective}")
                return objective

            logger.warning("Objective generation returned None")
            return None

        except Exception as e:
            logger.error(f"Error in write_objective: {e}")
            return None

    def extract_matched_skills(self, **chain_kwargs) -> list:
        """Extract matched skills from the resume and job post with LLM handling deduplication."""
        try:
            from dataModels.resume import ResumeSkillsMatcherOutput
            from services.langchain_helpers import invoke_structured

            extracted_skills = invoke_structured(
                self.user,
                "SKILLS_MATCHER",
                ResumeSkillsMatcherOutput,
                **self._get_prompt_inputs(),
            )

            if not extracted_skills:
                logger.warning("No extracted_skills returned from LLM")
                return self.skills or []

            # Handle both Pydantic model and dictionary responses
            if hasattr(extracted_skills, 'final_answer'):
                # Pydantic model
                extracted_skills_dict = extracted_skills.final_answer
            elif isinstance(extracted_skills, dict):
                # Dictionary response
                extracted_skills_dict = extracted_skills.get("final_answer", {})
            else:
                logger.error(f"Unexpected response type: {type(extracted_skills)}")
                return self.skills or []

            logger.info(f"LLM returned skills: {extracted_skills_dict}")

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

            logger.info(f"Final skills structure: {len(result)} categories")
            for category in result:
                if "subcategories" in category:
                    logger.info(f"  {category['category']}: {len(category['subcategories'])} subcategories")
                    for subcat in category['subcategories']:
                        logger.info(f"    - {subcat['name']}: {len(subcat['skills'])} skills")
                else:
                    logger.info(f"  {category['category']}: {len(category['skills'])} skills")

            return self._normalized_skill_groups(result)

        except Exception as e:
            logger.error(f"Error in extract_matched_skills: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self.skills or []

    def rewrite_unedited_experiences(self, **chain_kwargs) -> list:
        """Rewrite unedited experiences in the resume."""
        try:
            if not self.experiences:
                logger.info("No experiences to rewrite")
                return []

            logger.info(f"Rewriting {len(self.experiences)} experiences...")
            result = []
            for i, exp in enumerate(self.experiences):
                logger.info(f"Processing experience {i + 1}: {exp.get('title', 'Unknown')}")
                exp = dict(exp)

                # Log original highlights
                original_highlights = exp.get("highlights", [])
                logger.info(f"  Original highlights: {len(original_highlights)} items")
                for j, highlight in enumerate(original_highlights):
                    logger.debug(f"    {j + 1}: {highlight}")

                # Rewrite section
                new_highlights = self.rewrite_section(section=exp, section_id=f"experience:{i}", **chain_kwargs)
                logger.info(f"  New highlights: {len(new_highlights) if new_highlights else 0} items")

                if new_highlights:
                    for j, highlight in enumerate(new_highlights):
                        logger.debug(f"    NEW {j + 1}: {highlight}")
                    exp["highlights"] = new_highlights
                else:
                    logger.warning(f"  No new highlights generated, keeping original")
                    exp["highlights"] = original_highlights

                result.append(exp)

            logger.info(f"Completed rewriting {len(result)} experiences")
            return result
        except Exception as e:
            logger.error(f"Error in rewrite_unedited_experiences: {e}")
            import traceback
            logger.error(f"Experience rewrite traceback: {traceback.format_exc()}")
            return self.experiences or []

    def rewrite_unedited_projects(self, **chain_kwargs) -> list:
        """Rewrite unedited projects in the resume."""
        try:
            if not self.projects:
                logger.info("No projects to rewrite")
                return []

            logger.info(f"Rewriting {len(self.projects)} projects...")
            result = []
            for i, proj in enumerate(self.projects):
                logger.info(f"Processing project {i + 1}: {proj.get('name', 'Unknown')}")
                proj = dict(proj)

                # Log original highlights
                original_highlights = proj.get("highlights", [])
                logger.info(f"  Original highlights: {len(original_highlights)} items")
                for j, highlight in enumerate(original_highlights):
                    logger.debug(f"    {j + 1}: {highlight}")

                # Rewrite section
                new_highlights = self.rewrite_section(section=proj, section_id=f"project:{i}", **chain_kwargs)
                logger.info(f"  New highlights: {len(new_highlights) if new_highlights else 0} items")

                if new_highlights:
                    for j, highlight in enumerate(new_highlights):
                        logger.debug(f"    NEW {j + 1}: {highlight}")
                    proj["highlights"] = new_highlights
                else:
                    logger.warning(f"  No new highlights generated, keeping original")
                    proj["highlights"] = original_highlights

                result.append(proj)

            logger.info(f"Completed rewriting {len(result)} projects")
            return result
        except Exception as e:
            logger.error(f"Error in rewrite_unedited_projects: {e}")
            import traceback
            logger.error(f"Project rewrite traceback: {traceback.format_exc()}")
            return self.projects or []

    def rewrite_section(self, section, section_id: str = "", **chain_kwargs) -> list:
        """Rewrite a section of the resume."""
        original_highlights = section.get("highlights", [])
        try:
            from dataModels.resume import ResumeSectionHighlighterOutput
            from services.langchain_helpers import invoke_structured

            logger.debug(f"Starting rewrite_section for: {section.get('title') or section.get('name', 'Unknown')}")

            prompt_inputs = self._get_prompt_inputs(section=section, section_id=section_id)
            logger.debug("Invoking structured highlight generation...")
            section_revised = invoke_structured(
                self.user,
                "SECTION_HIGHLIGHTER",
                ResumeSectionHighlighterOutput,
                **prompt_inputs,
            )
            logger.debug(f"LLM response type: {type(section_revised)}")
            logger.debug(f"LLM response: {section_revised}")

            if section_revised:
                # Handle both Pydantic model and dictionary responses
                if hasattr(section_revised, 'final_answer'):
                    # Pydantic model
                    highlights = section_revised.final_answer or []
                    logger.info(f"Pydantic model: Got {len(highlights)} highlights")
                    if highlights:
                        sorted_highlights = sorted(highlights, key=lambda d: d.relevance * -1)

                        # Determine limit based on section type
                        section_type = self._determine_section_type(section)
                        limit = 4 if section_type == 'experience' else 2 if section_type == 'project' else len(
                            sorted_highlights)

                        # Apply limit
                        limited_highlights = sorted_highlights[:limit]
                        result = [s.highlight for s in limited_highlights]

                        logger.info(f"Limited to top {limit} highlights for {section_type} section")
                        logger.debug(f"Final highlights: {result}")
                        return self._validated_highlights(result, original_highlights, limit)

                elif isinstance(section_revised, dict):
                    # Dictionary response
                    highlights = section_revised.get("final_answer", [])
                    logger.info(f"Dictionary: Got {len(highlights)} highlights")
                    if highlights:
                        sorted_highlights = sorted(highlights, key=lambda d: d.get("relevance", 0) * -1)

                        # Determine limit based on section type
                        section_type = self._determine_section_type(section)
                        limit = 4 if section_type == 'experience' else 2 if section_type == 'project' else len(
                            sorted_highlights)

                        # Apply limit
                        limited_highlights = sorted_highlights[:limit]
                        result = [s.get("highlight", "") for s in limited_highlights]

                        logger.info(f"Limited to top {limit} highlights for {section_type} section")
                        logger.debug(f"Final highlights: {result}")
                        return self._validated_highlights(result, original_highlights, limit)
                else:
                    logger.error(f"Unexpected response type: {type(section_revised)}")
                    logger.error(f"Response content: {section_revised}")

            logger.warning("No valid highlights generated by LLM, returning original")
            logger.debug(f"Returning original highlights: {original_highlights}")
            return original_highlights

        except Exception as e:
            logger.error(f"Error in rewrite_section: {e}")
            import traceback
            logger.error(f"Rewrite section traceback: {traceback.format_exc()}")
            return section.get("highlights", [])

    def _validated_summary(self, summary) -> Optional[str]:
        """Keep summaries concise and avoid replacing a usable original with bad output."""
        if not isinstance(summary, str):
            return self.objective or None
        summary = " ".join(summary.split())
        if not summary or len(re.findall(r"\b\w+\b", summary)) > 55:
            logger.warning("Discarding empty or oversized generated summary")
            return self.objective or None
        return summary

    def _validated_highlights(self, candidates, originals, limit: int) -> list:
        """Apply deterministic readability checks after structured LLM output."""
        accepted, seen = [], set()
        for candidate in candidates or []:
            if not isinstance(candidate, str):
                continue
            candidate = " ".join(candidate.split())
            word_count = len(re.findall(r"\b\w+\b", candidate))
            normalized = candidate.casefold()
            if 3 <= word_count <= 30 and normalized not in seen:
                accepted.append(candidate)
                seen.add(normalized)

        if accepted:
            return accepted[:limit]
        logger.warning("Discarding invalid generated highlights and preserving source highlights")
        return list(originals or [])

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

    def _determine_section_type(self, section) -> str:
        """Determine if section is experience or project based on its structure."""
        # Check for experience indicators
        if 'titles' in section or 'title' in section or 'company' in section:
            return 'experience'
        # Check for project indicators
        elif 'name' in section:
            return 'project'
        else:
            # Default fallback
            return 'unknown'

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
            "section_evidence", "tailored_sections",
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
            logger.error("Failed to serialize the tailored resume as YAML")
            raise ValueError("Could not serialize the generated resume") from error

    def __del__(self):
        self.close()

    def close(self):
        """Release the LLM worker pool deterministically after a generation."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False, cancel_futures=True)
            del self.executor
