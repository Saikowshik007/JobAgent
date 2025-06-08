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
    Now includes mandatory suggest_improvements and apply_improvements functionality.
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

    def create_complete_tailored_resume(self, require_objective: bool = True) -> Dict:
        """
        Main method: Create complete tailored resume with improvement suggestions AND apply them.
        This is what ResumeGenerator calls - does everything including mandatory improvements.

        Args:
            require_objective: Whether to generate an objective section (default: True)

        Returns:
            Dict: Contains the final improved resume YAML, original tailored resume, and improvement suggestions
        """
        try:
            logger.info("=== Creating Complete Tailored Resume with Applied Improvements (Parallel) ===")
            logger.info(f"Objective generation: {'enabled' if require_objective else 'disabled'}")

            # Step 1: Generate core resume content in parallel
            try:
                start_time = time.time()

                # Check if we're in an async context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, use thread-based approach
                        logger.info("Using thread-based parallel approach (async context detected)")
                        results = self._generate_content_parallel_threads(require_objective=require_objective)
                    else:
                        # No active loop, safe to use asyncio.run
                        logger.info("Using asyncio-based parallel approach")
                        results = asyncio.run(self._generate_content_async_parallel(require_objective=require_objective))
                except RuntimeError:
                    # No event loop, safe to use asyncio.run
                    logger.info("Using asyncio-based parallel approach (no existing loop)")
                    results = asyncio.run(self._generate_content_async_parallel(require_objective=require_objective))

                end_time = time.time()
                logger.info(f"Parallel generation completed in {end_time - start_time:.2f} seconds")

            except Exception as parallel_error:
                logger.warning(f"Parallel execution failed: {parallel_error}, falling back to sequential")
                # Fallback to sequential execution
                results = self._generate_content_sequential(require_objective=require_objective)

            # Extract results with detailed logging
            objective = results.get('objective')
            skills = results.get('skills', [])
            experiences = results.get('experiences', [])
            projects = results.get('projects', [])

            logger.info(f"Content generation results:")
            logger.info(f"  - Objective: {'✓' if objective else '✗'}")
            logger.info(f"  - Skills: {len(skills)} categories")
            logger.info(f"  - Experiences: {len(experiences)} items")
            logger.info(f"  - Projects: {len(projects)} items")

            # Step 2: Create initial tailored resume
            logger.info("Assembling initial tailored resume...")
            initial_resume = {
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
                    'objective_generated': require_objective
                }
            }

            # Step 3: Convert to YAML
            initial_yaml = self.dict_to_yaml_string(initial_resume)

            # Step 4: Generate improvement suggestions (MANDATORY)
            logger.info("Generating improvement suggestions...")
            improvements = self.suggest_improvements()

            # Step 5: Apply improvements automatically (MANDATORY)
            logger.info("Applying improvements to resume...")
            improved_yaml = self.apply_improvements_to_resume(initial_yaml, improvements)

            # Step 6: Final optimization for length
            logger.info("Optimizing final resume for length...")
            final_yaml = self.optimize_resume_for_length(improved_yaml)

            # Step 7: Return complete result
            result = {
                'tailored_resume': final_yaml,              # Final improved and optimized resume
                'original_tailored': initial_yaml,          # Original tailored version (before improvements)
                'improvements': improvements,               # List of improvements that were applied
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'job_url': self.url,
                    'has_improvements': improvements is not None,
                    'improvements_applied': True,
                    'objective_generated': require_objective,
                    'optimized': True
                }
            }

            logger.info("=== Complete Resume Creation with Applied Improvements Finished ===")
            logger.info(f"Final result includes:")
            logger.info(f"  - Final resume: {'✓' if final_yaml else '✗'}")
            logger.info(f"  - Original tailored: {'✓' if initial_yaml else '✗'}")
            logger.info(f"  - Improvements: {'✓' if improvements else '✗'}")

            return result

        except Exception as e:
            logger.error(f"Complete resume creation with applied improvements failed: {e}")
            raise

    def suggest_improvements(self, **chain_kwargs) -> List:
        """
        Suggest improvements for the resume based on job requirements.
        This is the core improvement functionality.

        Args:
            **chain_kwargs: Additional keyword arguments for the chain.

        Returns:
            List: The suggested improvements with categories and specific recommendations.
        """
        try:
            logger.info("Starting resume improvement analysis...")

            from prompts import Prompts
            from dataModels.resume import ResumeImproverOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            # Create the improvement chain
            prompt = ChatPromptTemplate(messages=Prompts.lookup["IMPROVER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeImproverOutput)

            # Get formatted inputs for the chain
            chain_inputs = self._get_formatted_chain_inputs(chain=chain)
            logger.debug(f"Improvement chain inputs: {list(chain_inputs.keys())}")

            # Invoke the LLM to get improvement suggestions
            logger.debug("Invoking LLM for improvement suggestions...")
            improvements = chain.invoke(chain_inputs)
            logger.debug(f"LLM response type: {type(improvements)}")

            if improvements:
                # Handle both Pydantic model and dictionary responses
                if hasattr(improvements, 'final_answer'):
                    # Pydantic model
                    improvement_data = improvements.final_answer
                    logger.info("Retrieved improvements from Pydantic model")
                elif isinstance(improvements, dict):
                    # Dictionary response
                    improvement_data = improvements.get('final_answer')
                    logger.info("Retrieved improvements from dictionary")
                else:
                    logger.error(f"Unexpected response type: {type(improvements)}")
                    return []

                if improvement_data:
                    logger.info("✓ Resume improvement suggestions generated successfully")

                    # Log summary of improvements
                    if isinstance(improvement_data, list):
                        logger.info(f"Generated {len(improvement_data)} improvement categories")
                        for imp in improvement_data:
                            if hasattr(imp, 'section'):
                                section = imp.section
                                count = len(imp.improvements) if hasattr(imp, 'improvements') else 0
                            elif isinstance(imp, dict):
                                section = imp.get('section', 'Unknown')
                                count = len(imp.get('improvements', []))
                            else:
                                section = 'Unknown'
                                count = 0
                            logger.info(f"  - {section}: {count} suggestions")

                    return improvement_data
                else:
                    logger.warning("No improvement data in LLM response")
                    return []
            else:
                logger.warning("No improvements returned from LLM")
                return []

        except Exception as e:
            logger.error(f"Error in suggest_improvements: {e}")
            import traceback
            logger.error(f"Improvement traceback: {traceback.format_exc()}")
            return []

    def apply_improvements_to_resume(self, resume_yaml: str, improvements: List) -> str:
        """
        Apply the improvement suggestions to the actual resume content automatically.

        Args:
            resume_yaml: The current resume in YAML format
            improvements: List of ResumeImprovements from suggest_improvements()

        Returns:
            str: The improved resume in YAML format
        """
        try:
            logger.info("Starting automatic application of improvements...")

            # If no improvements, return original
            if not improvements:
                logger.info("No improvements to apply, returning original resume")
                return resume_yaml

            from prompts import Prompts
            from dataModels.resume import ResumeImprovementApplierOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            # Convert improvements list to a readable format for the LLM
            improvements_text = self._format_improvements_for_application(improvements)
            logger.info(f"Formatted {len(improvements)} improvement categories for application")

            # Create the improvement application chain
            prompt = ChatPromptTemplate(messages=Prompts.lookup["IMPROVEMENT_APPLIER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeImprovementApplierOutput)

            # Prepare inputs
            chain_inputs = {
                "resume_yaml": resume_yaml,
                "improvements_text": improvements_text,
                "ats_keywords": ", ".join(self.parsed_job.get('ats_keywords', [])) if self.parsed_job else ""
            }

            logger.info("Invoking LLM to apply improvements...")
            result = chain.invoke(chain_inputs)

            if result:
                # Handle both Pydantic model and dictionary responses
                if hasattr(result, 'final_answer'):
                    # Pydantic model
                    improved_resume = result.final_answer
                    logger.info("Retrieved improved resume from Pydantic model")
                elif isinstance(result, dict):
                    # Dictionary response
                    improved_resume = result.get("final_answer")
                    logger.info("Retrieved improved resume from dictionary")
                else:
                    logger.error(f"Unexpected response type: {type(result)}")
                    return resume_yaml

                if improved_resume and isinstance(improved_resume, str):
                    # Validate that it's proper YAML
                    try:
                        import yaml
                        yaml.safe_load(improved_resume)
                        logger.info("✓ Improved resume is valid YAML")
                        logger.info(f"Applied improvements successfully")
                        return improved_resume
                    except yaml.YAMLError as e:
                        logger.warning(f"Improved resume is not valid YAML: {e}")
                        logger.warning("Returning original resume")
                        return resume_yaml
                else:
                    logger.warning("Invalid improved resume format")
                    return resume_yaml
            else:
                logger.warning("No improved resume returned from LLM")
                return resume_yaml

        except Exception as e:
            logger.error(f"Error applying improvements to resume: {e}")
            import traceback
            logger.error(f"Apply improvements traceback: {traceback.format_exc()}")
            return resume_yaml

    def _format_improvements_for_application(self, improvements: List) -> str:
        """
        Format the improvements list into a readable text format for the LLM.

        Args:
            improvements: List of ResumeImprovements objects

        Returns:
            str: Formatted improvements text
        """
        try:
            formatted_text = ""

            for improvement_section in improvements:
                # Handle both Pydantic objects and dictionaries
                if hasattr(improvement_section, 'section'):
                    section = improvement_section.section
                    section_improvements = improvement_section.improvements
                elif isinstance(improvement_section, dict):
                    section = improvement_section.get('section', 'Unknown')
                    section_improvements = improvement_section.get('improvements', [])
                else:
                    logger.warning(f"Unexpected improvement format: {type(improvement_section)}")
                    continue

                formatted_text += f"\n## {section.upper()} SECTION IMPROVEMENTS:\n"

                for i, improvement in enumerate(section_improvements, 1):
                    formatted_text += f"{i}. {improvement}\n"

                formatted_text += "\n"

            logger.debug(f"Formatted improvements text: {formatted_text[:500]}...")
            return formatted_text

        except Exception as e:
            logger.error(f"Error formatting improvements: {e}")
            return "No improvements to apply."

    async def _generate_content_async_parallel(self, require_objective: bool = True) -> Dict:
        """Generate all resume content in parallel using asyncio.gather."""
        # Create async tasks that run in thread pool (this gives true HTTP parallelism)
        if not hasattr(self, 'executor'):
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        loop = asyncio.get_event_loop()

        logger.info("Starting parallel task execution...")

        tasks = []
        task_names = []

        # Conditionally add objective task
        if require_objective:
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
            results = [None] * len(tasks)

        # Process results with detailed logging
        processed_results = {}

        for i, (result, task_name) in enumerate(zip(results, task_names)):
            if isinstance(result, Exception):
                logger.error(f"Task '{task_name}' failed with exception: {result}")
                processed_results[task_name] = self._get_default_value(task_name)
            else:
                logger.info(f"Task '{task_name}' completed successfully")
                processed_results[task_name] = result

        # If objective was not requested, set it to empty string
        if not require_objective:
            processed_results['objective'] = ""

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

    def _generate_content_parallel_threads(self, require_objective: bool = True) -> Dict:
        """Generate content using ThreadPoolExecutor for cases where we're already in async context."""
        logger.info("Using ThreadPoolExecutor for parallel generation...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks conditionally
            future_to_task = {}

            # Conditionally submit objective task
            if require_objective:
                future_to_task[executor.submit(self._safe_write_objective)] = 'objective'

            # Always submit these tasks
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
                if require_objective:
                    expected_tasks.append('objective')

                for task_name in expected_tasks:
                    if task_name not in results:
                        results[task_name] = self._get_default_value(task_name)
                        logger.warning(f"Using default value for {task_name} due to timeout")

            # If objective was not requested, set it to empty string
            if not require_objective:
                results['objective'] = ""

            logger.info(f"Thread-based parallel execution completed: {len(results)}/{total_tasks} tasks")
            return results

    def _generate_content_sequential(self, require_objective: bool = True) -> Dict:
        """Fallback sequential content generation."""
        logger.info("Running sequential content generation...")

        results = {}

        if require_objective:
            logger.info("Sequential: Generating objective...")
            results['objective'] = self._safe_write_objective()
        else:
            logger.info("Sequential: Skipping objective generation...")
            results['objective'] = ""

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
            'objective': "",
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
                        result = [s.highlight for s in sorted_highlights]
                        logger.debug(f"Final highlights: {result}")
                        return result

                elif isinstance(section_revised, dict):
                    # Dictionary response
                    highlights = section_revised.get("final_answer", [])
                    logger.info(f"Dictionary: Got {len(highlights)} highlights")
                    if highlights:
                        sorted_highlights = sorted(highlights, key=lambda d: d.get("relevance", 0) * -1)
                        result = [s.get("highlight", "") for s in sorted_highlights]
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

    def optimize_resume_for_length(self, resume_draft: str) -> str:
        """Optimize the resume YAML or plain text to fit within a single page, prioritizing job match."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeOnePageOptimizerOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            logger.info("Starting one-page resume optimization...")

            # Set up the prompt and LLM
            prompt = ChatPromptTemplate(messages=Prompts.lookup["ONE_PAGE_OPTIMIZER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeOnePageOptimizerOutput)

            # Prepare the chain inputs
            chain_inputs = {
                "job_posting": self.job_post_raw or "No job description available.",
                "resume_draft": resume_draft
            }

            logger.info(f"Input resume length: {len(resume_draft)} characters")
            logger.info("Invoking LLM for one-page resume optimization...")

            result = chain.invoke(chain_inputs)
            logger.info(f"LLM response type: {type(result)}")

            # Handle both Pydantic model and dictionary responses
            optimized_resume = None

            if hasattr(result, "final_answer"):
                # Pydantic model
                optimized_resume = result.final_answer
                logger.info("Retrieved optimized resume from Pydantic model")
            elif isinstance(result, dict):
                # Dictionary response
                optimized_resume = result.get("final_answer")
                logger.info("Retrieved optimized resume from dictionary")
            else:
                logger.error(f"Unexpected response type: {type(result)}")
                logger.error(f"Response content: {result}")
                return resume_draft

            if optimized_resume and isinstance(optimized_resume, str):
                logger.info(f"Optimized resume length: {len(optimized_resume)} characters")

                # Validate that it's proper YAML
                try:
                    import yaml
                    yaml.safe_load(optimized_resume)
                    logger.info("Optimized resume is valid YAML")
                    return optimized_resume
                except yaml.YAMLError as e:
                    logger.warning(f"Optimized resume is not valid YAML: {e}")
                    logger.warning("Returning original resume draft")
                    return resume_draft
            else:
                logger.warning(f"Invalid final_answer type: {type(optimized_resume)}")
                logger.warning("Returning original resume draft")
                return resume_draft

        except Exception as e:
            logger.error(f"One-page resume optimization failed: {e}")
            import traceback
            logger.error(f"Optimization traceback: {traceback.format_exc()}")
            return resume_draft

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)