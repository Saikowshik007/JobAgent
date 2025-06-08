import yaml
import requests
from bs4 import BeautifulSoup
from fp.fp import FreeProxy
from yaml import YAMLError
from services.langchain_helpers import *
from dataModels.job_post import JobPost
from config import config
import asyncio
import concurrent.futures
from datetime import datetime
import time
from typing import Dict, List, Optional

logger = config.getLogger("ResumeImprover")

class ResumeImprover:
    """
    Parallel ResumeImprover using asyncio.gather with run_in_executor for true HTTP parallelism.
    """

    def __init__(self, url, api_key, parsed_job=None, llm_kwargs: dict = None, timeout: int = 500):
        """Initialize ResumeImprover with the job post URL and optional resume location."""
        super().__init__()
        self.job_post_html_data = None
        self.job_post_raw = None
        self.resume = None
        self.job_post = None
        self.parsed_job = parsed_job
        self.llm_kwargs = llm_kwargs or {}
        self.api_key = api_key
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

        # Thread pool for running sync LLM calls
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def create_complete_tailored_resume(self) -> str:
        """
        NEW main method: Create complete tailored resume with parallel processing.
        This is what ResumeGenerator calls - does everything.
        """
        try:
            logger.info("=== Creating Complete Tailored Resume (Parallel) ===")

            # Try parallel execution first
            try:
                start_time = time.time()

                # Check if we're in an async context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, use thread-based approach
                        logger.info("Using thread-based parallel approach (async context detected)")
                        results = self._generate_content_parallel_threads()
                    else:
                        # No active loop, safe to use asyncio.run
                        logger.info("Using asyncio-based parallel approach")
                        results = asyncio.run(self._generate_content_async_parallel())
                except RuntimeError:
                    # No event loop, safe to use asyncio.run
                    logger.info("Using asyncio-based parallel approach (no existing loop)")
                    results = asyncio.run(self._generate_content_async_parallel())

                end_time = time.time()
                logger.info(f"Parallel generation completed in {end_time - start_time:.2f} seconds")

            except Exception as parallel_error:
                logger.warning(f"Parallel execution failed: {parallel_error}, falling back to sequential")
                # Fallback to sequential execution
                results = self._generate_content_sequential()

            # Extract results with detailed logging
            objective = results.get('objective')
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
                    'tailored': True
                }
            }

            # Step 3: Convert to YAML
            yaml_content = self.dict_to_yaml_string(final_resume)
            logger.info("=== Resume Creation Complete ===")

            optimized_yaml = self.optimize_resume_for_length(yaml_content)
            return optimized_yaml

        except Exception as e:
            logger.error(f"Complete resume creation failed: {e}")
            raise

    async def _generate_content_async_parallel(self) -> Dict:
        """Generate all resume content in parallel using asyncio.gather."""
        # Create async tasks that run in thread pool (this gives true HTTP parallelism)
        if not hasattr(self, 'executor'):
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        loop = asyncio.get_event_loop()

        logger.info("Starting parallel task execution...")

        tasks = [
            loop.run_in_executor(self.executor, self._safe_write_objective),
            loop.run_in_executor(self.executor, self._safe_extract_matched_skills),
            loop.run_in_executor(self.executor, self._safe_rewrite_experiences),
            loop.run_in_executor(self.executor, self._safe_rewrite_projects)
        ]

        task_names = ['objective', 'skills', 'experiences', 'projects']

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
            results = [None, [], [], []]

        # Process results with detailed logging
        processed_results = {}

        for i, (result, task_name) in enumerate(zip(results, task_names)):
            if isinstance(result, Exception):
                logger.error(f"Task '{task_name}' failed with exception: {result}")
                processed_results[task_name] = self._get_default_value(task_name)
            else:
                logger.info(f"Task '{task_name}' completed successfully")
                processed_results[task_name] = result

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

    def _generate_content_parallel_threads(self) -> Dict:
        """Generate content using ThreadPoolExecutor for cases where we're already in async context."""
        logger.info("Using ThreadPoolExecutor for parallel generation...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._safe_write_objective): 'objective',
                executor.submit(self._safe_extract_matched_skills): 'skills',
                executor.submit(self._safe_rewrite_experiences): 'experiences',
                executor.submit(self._safe_rewrite_projects): 'projects'
            }

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
                for task_name in ['objective', 'skills', 'experiences', 'projects']:
                    if task_name not in results:
                        results[task_name] = self._get_default_value(task_name)
                        logger.warning(f"Using default value for {task_name} due to timeout")

            logger.info(f"Thread-based parallel execution completed: {len(results)}/4 tasks")
            return results

    def _generate_content_sequential(self) -> Dict:
        """Fallback sequential content generation."""
        logger.info("Running sequential content generation...")

        results = {}

        logger.info("Sequential: Generating objective...")
        results['objective'] = self._safe_write_objective()

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

    def download_and_parse_job_post(self, url=None):
        """Download and parse the job post from the provided URL."""
        if url:
            self.url = url
        self._download_url()
        self._extract_html_data()
        self.job_post = JobPost(self.job_post_raw, self.api_key)
        self.parsed_job = self.job_post.parse_job_post(verbose=False)

    def _extract_html_data(self):
        """Extract text content from HTML, removing all HTML tags.

        Raises:
            Exception: If HTML data extraction fails.
        """
        try:
            soup = BeautifulSoup(self.job_post_html_data, "html.parser")
            self.job_post_raw = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            config.logger.error(f"Failed to extract HTML data: {e}")
            raise

    def _download_url(self, url=None):
        """Download the content of the URL and return it as a string.

        Args:
            url (str, optional): The URL to download. Defaults to None.

        Returns:
            bool: True if download was successful, False otherwise.
        """
        if url:
            self.url = url

        max_retries = config.get("settings.max_retries")
        backoff_factor = config.get("settings.backoff_factor")
        use_proxy = False

        for attempt in range(max_retries):
            try:
                proxies = None
                if use_proxy:
                    proxy = FreeProxy(rand=True).get()
                    proxies = {"http": proxy, "https": proxy}

                response = requests.get(
                    self.url, headers=config.get_enhanced_headers(self.url), proxies=proxies
                )
                response.raise_for_status()
                self.job_post_html_data = response.text
                return True

            except requests.RequestException as e:
                if response.status_code == 429:
                    config.logger.warning(
                        f"Rate limit exceeded. Retrying in {backoff_factor * 2 ** attempt} seconds..."
                    )
                    time.sleep(backoff_factor * 2 ** attempt)
                    use_proxy = True
                else:
                    config.logger.error(f"Failed to download URL {self.url}: {e}")
                    return False

        config.logger.error(f"Exceeded maximum retries for URL {self.url}")
        return False

    def write_objective(self, **chain_kwargs) -> str:
        """Write an objective for the resume."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSummarizerOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            # Create chain
            prompt = ChatPromptTemplate(messages=Prompts.lookup["OBJECTIVE_WRITER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeSummarizerOutput)

            # Get inputs
            chain_inputs = self._get_formatted_chain_inputs(chain=chain)
            logger.debug(f"Objective chain inputs: {list(chain_inputs.keys())}")

            # Generate
            result = chain.invoke(chain_inputs)
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
            from prompts import Prompts
            from dataModels.resume import ResumeSkillsMatcherOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            chain = ChatPromptTemplate(messages=Prompts.lookup["SKILLS_MATCHER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)

            # Keep using function_calling method since json_mode requires "json" in prompts
            runnable = chain | llm.with_structured_output(schema=ResumeSkillsMatcherOutput, method="function_calling")

            chain_inputs = self._get_formatted_chain_inputs(chain=runnable)
            extracted_skills = runnable.invoke(chain_inputs)

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

            return result

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
                new_highlights = self.rewrite_section(section=exp, **chain_kwargs)
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
                new_highlights = self.rewrite_section(section=proj, **chain_kwargs)
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

    def rewrite_section(self, section, **chain_kwargs) -> list:
        """Rewrite a section of the resume."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSectionHighlighterOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            logger.debug(f"Starting rewrite_section for: {section.get('title') or section.get('name', 'Unknown')}")

            prompt = ChatPromptTemplate(messages=Prompts.lookup["SECTION_HIGHLIGHTER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeSectionHighlighterOutput)

            chain_inputs = self._get_formatted_chain_inputs(chain=chain, section=section)
            logger.debug(f"Chain inputs keys: {list(chain_inputs.keys())}")

            # Log some key inputs for debugging
            if 'section' in chain_inputs:
                section_info = chain_inputs['section']
                if isinstance(section_info, str):
                    logger.debug(f"Section input (first 200 chars): {section_info[:200]}...")
                else:
                    logger.debug(f"Section input type: {type(section_info)}")

            logger.debug("Invoking LLM chain...")
            section_revised = chain.invoke(chain_inputs)
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
                        limit = 5 if section_type == 'experience' else 3 if section_type == 'project' else len(sorted_highlights)

                        # Apply limit
                        limited_highlights = sorted_highlights[:limit]
                        result = [s.highlight for s in limited_highlights]

                        logger.info(f"Limited to top {limit} highlights for {section_type} section")
                        logger.debug(f"Final highlights: {result}")
                        return result

                elif isinstance(section_revised, dict):
                    # Dictionary response
                    highlights = section_revised.get("final_answer", [])
                    logger.info(f"Dictionary: Got {len(highlights)} highlights")
                    if highlights:
                        sorted_highlights = sorted(highlights, key=lambda d: d.get("relevance", 0) * -1)

                        # Determine limit based on section type
                        section_type = self._determine_section_type(section)
                        limit = 5 if section_type == 'experience' else 3 if section_type == 'project' else len(sorted_highlights)

                        # Apply limit
                        limited_highlights = sorted_highlights[:limit]
                        result = [s.get("highlight", "") for s in limited_highlights]

                        logger.info(f"Limited to top {limit} highlights for {section_type} section")
                        logger.debug(f"Final highlights: {result}")
                        return result
                else:
                    logger.error(f"Unexpected response type: {type(section_revised)}")
                    logger.error(f"Response content: {section_revised}")

            logger.warning("No valid highlights generated by LLM, returning original")
            original_highlights = section.get("highlights", [])
            logger.debug(f"Returning original highlights: {original_highlights}")
            return original_highlights

        except Exception as e:
            logger.error(f"Error in rewrite_section: {e}")
            import traceback
            logger.error(f"Rewrite section traceback: {traceback.format_exc()}")
            return section.get("highlights", [])

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

    def _get_formatted_chain_inputs(self, chain, section=None):
        """Get formatted inputs for chain with proper skills formatting"""
        from services.langchain_helpers import chain_formatter

        output_dict = {}
        raw_self_data = self.__dict__
        if section is not None:
            raw_self_data = raw_self_data.copy()
            raw_self_data["section"] = section

        for key in chain.get_input_schema().schema().get("required", []):
            value = raw_self_data.get(key) or (self.parsed_job.get(key) if self.parsed_job else None)

            # Special handling for skills - pass the raw structure to chain_formatter
            # Don't pre-format it here since chain_formatter will handle the formatting
            if key == "skills" and self.skills:
                # Pass the raw skills structure to chain_formatter, not a pre-formatted string
                value = self.skills
                logger.debug(f"Passing raw skills structure to chain_formatter: {len(self.skills)} categories")

            output_dict[key] = chain_formatter(key, value)

            # Debug log for skills specifically
            if key == "skills":
                logger.debug(f"After chain_formatter, skills input type: {type(output_dict[key])}")

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

    def _combine_skills_in_category(self, l1: list[str], l2: list[str]):
        """Combine two lists of skills without duplicating lowercase entries."""
        l1_lower = {i.lower() for i in l1}
        for i in l2:
            if i.lower() not in l1_lower:
                l1.append(i)

    def _combine_skill_lists(self, l1: list[dict], l2: list[dict]):
        """Combine two lists of skill categories without duplicating lowercase entries."""
        l1_categories_lowercase = {s["category"].lower(): i for i, s in enumerate(l1)}

        for s in l2:
            category_lower = s["category"].lower()

            if category_lower in l1_categories_lowercase:
                # Get the existing category
                existing_idx = l1_categories_lowercase[category_lower]
                existing_category = l1[existing_idx]

                # Handle different structures: subcategories vs direct skills
                if "subcategories" in existing_category and "subcategories" in s:
                    # Both have subcategories - merge subcategories
                    existing_subcats = {sub["name"].lower(): sub for sub in existing_category["subcategories"]}

                    for new_subcat in s["subcategories"]:
                        subcat_name_lower = new_subcat["name"].lower()
                        if subcat_name_lower in existing_subcats:
                            # Merge skills within the subcategory
                            self._combine_skills_in_category(
                                existing_subcats[subcat_name_lower]["skills"],
                                new_subcat["skills"]
                            )
                        else:
                            # Add new subcategory
                            existing_category["subcategories"].append(new_subcat)

                elif "skills" in existing_category and "skills" in s:
                    # Both have direct skills - merge them
                    self._combine_skills_in_category(
                        existing_category["skills"],
                        s["skills"]
                    )

                elif "subcategories" in existing_category and "skills" in s:
                    # Existing has subcategories, new has direct skills
                    # Convert new skills to a subcategory or handle as needed
                    if not any(sub["name"].lower() == "general" for sub in existing_category["subcategories"]):
                        existing_category["subcategories"].append({
                            "name": "General",
                            "skills": s["skills"][:]
                        })
                    else:
                        # Find General subcategory and add skills
                        for sub in existing_category["subcategories"]:
                            if sub["name"].lower() == "general":
                                self._combine_skills_in_category(sub["skills"], s["skills"])
                                break

                elif "skills" in existing_category and "subcategories" in s:
                    # Existing has direct skills, new has subcategories
                    # Convert existing to subcategories format
                    existing_skills = existing_category["skills"][:]
                    existing_category["subcategories"] = [
                        {"name": "General", "skills": existing_skills}
                    ]
                    del existing_category["skills"]

                    # Now merge the subcategories
                    existing_subcats = {"general": existing_category["subcategories"][0]}
                    for new_subcat in s["subcategories"]:
                        subcat_name_lower = new_subcat["name"].lower()
                        if subcat_name_lower in existing_subcats:
                            self._combine_skills_in_category(
                                existing_subcats[subcat_name_lower]["skills"],
                                new_subcat["skills"]
                            )
                        else:
                            existing_category["subcategories"].append(new_subcat)
            else:
                # Add new category
                l1.append(s)

    def dict_to_yaml_string(self, data: dict) -> str:
        """Converts a dictionary to a YAML-formatted string."""
        yaml.allow_unicode = True
        try:
            from io import StringIO
            stream = StringIO()
            yaml.dump(data, stream=stream, default_flow_style=False, allow_unicode=True)
            return stream.getvalue()
        except YAMLError as e:
            logger.error("Failed to convert dict to YAML string.")
            raise e

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def optimize_resume_for_length(self, resume_draft: str) -> str:
        """Optimize the resume YAML (or plaintext) to fit within a single page, prioritizing job match."""
        try:
            from prompts import Prompts
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            # Create the LLM chain
            prompt = ChatPromptTemplate(messages=Prompts.lookup["ONE_PAGE_OPTIMIZER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
            chain = prompt | llm

            # Prepare inputs
            job_posting = self.job_post_raw or "No job description provided."
            inputs = {
                "job_posting": job_posting,
                "resume_draft": resume_draft
            }

            logger.info("Invoking LLM for one-page optimization...")
            result = chain.invoke(inputs)

            if not result:
                logger.warning("Second-pass optimization returned no result.")
                return resume_draft

            logger.info("One-page optimization completed successfully.")
            return result if isinstance(result, str) else str(result)

        except Exception as e:
            logger.error(f"Error during one-page resume optimization: {e}")
            return resume_draft

