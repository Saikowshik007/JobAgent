import yaml
import requests
from bs4 import BeautifulSoup
from fp.fp import FreeProxy
from yaml import YAMLError
import asyncio
import concurrent.futures
from datetime import datetime
import time
from typing import Dict, List, Optional
import instructor
from openai import AsyncOpenAI, OpenAI

# Import existing modules
from dataModels.job_post import JobPost
from dataModels.resume import (
    ResumeSectionHighlighterOutput,
    ResumeSkillsMatcherOutput,
    ResumeSummarizerOutput,
    ComprehensiveSuggestionsOutput,
    AllSuggestionsApplierOutput
)
from config import config

logger = config.getLogger("ResumeImprover")


class ResumeImprover:
    """
    Resume improver using direct OpenAI APIs with Instructor instead of LangChain.
    Maintains the exact same interface but with much better performance.
    """

    def __init__(self, url, api_key, parsed_job=None, llm_kwargs: dict = None, timeout: int = 500):
        """Initialize ResumeImprover with direct API setup."""
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

        # Direct API clients with instructor
        self.model = self.llm_kwargs.get("model_name", "gpt-4o")
        self.async_client = instructor.apatch(AsyncOpenAI(
            api_key=api_key,
            timeout=min(timeout, 60)  # OpenAI client timeout limit
        ))
        self.sync_client = instructor.patch(OpenAI(
            api_key=api_key,
            timeout=min(timeout, 60)
        ))

        # Keep thread pool for backward compatibility
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def create_complete_tailored_resume(self, require_objective) -> Dict:
        """
        Main method: Create complete tailored resume with direct APIs.
        SAME INTERFACE as before but much faster implementation.
        """
        try:
            logger.info("=== Creating Complete Tailored Resume with Direct APIs ===")
            logger.info(f"Objective generation: {'enabled' if require_objective else 'disabled'}")

            # Run the async version in the event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, use thread pool
                    logger.info("Using thread pool approach (async context detected)")
                    return self._generate_content_parallel_threads(require_objective=require_objective)
                else:
                    # We can run async directly
                    logger.info("Using direct async approach")
                    return asyncio.run(self._create_complete_tailored_resume_async(require_objective))
            except RuntimeError:
                # No event loop, create one
                logger.info("Creating new event loop for async operations")
                return asyncio.run(self._create_complete_tailored_resume_async(require_objective))

        except Exception as e:
            logger.error(f"Complete resume creation failed: {e}")
            raise

    async def _create_complete_tailored_resume_async(self, require_objective: bool) -> str:
        """Async version of the main resume creation logic."""
        # Step 1: Generate core resume content in parallel
        start_time = time.time()
        results = await self._generate_content_async_parallel(require_objective=require_objective)
        end_time = time.time()
        logger.info(f"Parallel generation completed in {end_time - start_time:.2f} seconds")

        # Extract results
        objective = results.get('objective', '')
        skills = results.get('skills', [])
        experiences = results.get('experiences', [])
        projects = results.get('projects', [])

        logger.info(f"Content generation results:")
        logger.info(f"  - Objective: {'✓' if objective else '✗'}")
        logger.info(f"  - Skills: {len(skills)} categories")
        logger.info(f"  - Experiences: {len(experiences)} items")
        logger.info(f"  - Projects: {len(projects)} items")

        # Step 2: Create initial tailored resume
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

        # Step 4: Get comprehensive improvement suggestions
        logger.info("Getting comprehensive improvement and optimization suggestions...")
        all_suggestions = await self.get_comprehensive_suggestions(initial_yaml)

        # # Step 5: Apply all suggestions at once
        logger.info("Applying all suggestions to resume...")
        final_yaml = await self.apply_all_suggestions_to_resume(initial_yaml, all_suggestions)

        logger.info("=== Complete Resume Creation with Direct APIs Finished ===")
        return initial_yaml

    async def _generate_content_async_parallel(self, require_objective: bool = True) -> Dict:
        """Generate all resume content in parallel using true async."""
        logger.info("Starting parallel task execution with direct APIs...")

        tasks = []
        task_names = []

        # Conditionally add objective task
        if require_objective:
            tasks.append(self._write_objective_async())
            task_names.append('objective')

        # Always add these tasks
        tasks.extend([
            self._extract_matched_skills_async(),
            self._rewrite_experiences_async(),
            self._rewrite_projects_async()
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
            results = [None] * len(tasks)

        # Process results
        processed_results = {}
        for i, (result, task_name) in enumerate(zip(results, task_names)):
            if isinstance(result, Exception):
                logger.error(f"Task '{task_name}' failed with exception: {result}")
                processed_results[task_name] = self._get_default_value(task_name)
            else:
                logger.info(f"Task '{task_name}' completed successfully")
                processed_results[task_name] = result

        if not require_objective:
            processed_results['objective'] = ""

        return processed_results

    # Direct API implementations of core methods
    async def _write_objective_async(self) -> str:
        """Async version using direct APIs."""
        try:
            prompt = self._build_prompt_from_yaml("OBJECTIVE_WRITER")

            result = await self.async_client.chat.completions.create(
                model=self.model,
                response_model=ResumeSummarizerOutput,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ]
            )

            return result.final_answer
        except Exception as e:
            logger.error(f"Error in async write_objective: {e}")
            return ""

    async def _extract_matched_skills_async(self) -> List:
        """Async version using direct APIs."""
        try:
            prompt = self._build_prompt_from_yaml("SKILLS_MATCHER")

            result = await self.async_client.chat.completions.create(
                model=self.model,
                response_model=ResumeSkillsMatcherOutput,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ]
            )

            extracted_skills = result.final_answer

            # Convert to expected format
            result_skills = []

            # Handle technical skills
            if extracted_skills.technical_skills:
                subcategories = []
                for category_name, skills_list in extracted_skills.technical_skills.items():
                    if skills_list:
                        subcategories.append({
                            "name": category_name,
                            "skills": skills_list
                        })

                if subcategories:
                    result_skills.append({
                        "category": "Technical",
                        "subcategories": subcategories
                    })

            # Handle non-technical skills
            if extracted_skills.non_technical_skills:
                result_skills.append({
                    "category": "Non-technical",
                    "skills": extracted_skills.non_technical_skills
                })

            return result_skills
        except Exception as e:
            logger.error(f"Error in async extract_matched_skills: {e}")
            return self.skills or []

    async def _rewrite_experiences_async(self) -> List:
        """Async version using direct APIs."""
        try:
            if not self.experiences:
                return []

            # Process all experiences in parallel
            tasks = []
            for exp in self.experiences:
                tasks.append(self._rewrite_single_section_async(exp))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            final_experiences = []
            for i, (exp, result) in enumerate(zip(self.experiences, results)):
                if isinstance(result, Exception):
                    logger.error(f"Experience {i} rewrite failed: {result}")
                    final_experiences.append(exp)
                else:
                    exp_copy = dict(exp)
                    exp_copy["highlights"] = result if result else exp.get("highlights", [])
                    final_experiences.append(exp_copy)

            return final_experiences
        except Exception as e:
            logger.error(f"Error in async rewrite_experiences: {e}")
            return self.experiences or []

    async def _rewrite_projects_async(self) -> List:
        """Async version using direct APIs."""
        try:
            if not self.projects:
                return []

            # Process all projects in parallel
            tasks = []
            for proj in self.projects:
                tasks.append(self._rewrite_single_section_async(proj))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            final_projects = []
            for i, (proj, result) in enumerate(zip(self.projects, results)):
                if isinstance(result, Exception):
                    logger.error(f"Project {i} rewrite failed: {result}")
                    final_projects.append(proj)
                else:
                    proj_copy = dict(proj)
                    proj_copy["highlights"] = result if result else proj.get("highlights", [])
                    final_projects.append(proj_copy)

            return final_projects
        except Exception as e:
            logger.error(f"Error in async rewrite_projects: {e}")
            return self.projects or []

    async def _rewrite_single_section_async(self, section: dict) -> List[str]:
        """Rewrite a single section using direct APIs."""
        try:
            prompt = self._build_prompt_from_yaml("SECTION_HIGHLIGHTER", section=section)

            result = await self.async_client.chat.completions.create(
                model=self.model,
                response_model=ResumeSectionHighlighterOutput,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ]
            )

            if result.final_answer:
                sorted_highlights = sorted(result.final_answer, key=lambda x: x.relevance, reverse=True)
                return [h.highlight for h in sorted_highlights]

            return section.get("highlights", [])
        except Exception as e:
            logger.error(f"Error rewriting section: {e}")
            return section.get("highlights", [])

    async def get_comprehensive_suggestions(self, resume_yaml: str) -> Dict:
        """Get comprehensive suggestions using direct APIs."""
        try:
            prompt = self._build_prompt_from_yaml("COMPREHENSIVE_SUGGESTIONS", resume_yaml=resume_yaml)

            result = await self.async_client.chat.completions.create(
                model=self.model,
                response_model=ComprehensiveSuggestionsOutput,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ]
            )

            if result.final_answer:
                return {
                    'content_improvements': result.final_answer.content_improvements,
                    'optimization_suggestions': result.final_answer.optimization_suggestions
                }
            return {}
        except Exception as e:
            logger.error(f"Error in get_comprehensive_suggestions: {e}")
            return {}

    async def apply_all_suggestions_to_resume(self, resume_yaml: str, all_suggestions: Dict) -> str:
        """Apply all suggestions using direct APIs."""
        try:
            if not all_suggestions:
                return resume_yaml

            suggestions_text = self._format_all_suggestions_for_application(all_suggestions)
            prompt = self._build_prompt_from_yaml("ALL_SUGGESTIONS_APPLIER",
                                                  resume_yaml=resume_yaml,
                                                  suggestions_text=suggestions_text)

            result = await self.async_client.chat.completions.create(
                model=self.model,
                response_model=AllSuggestionsApplierOutput,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ]
            )

            if result.final_answer:
                try:
                    yaml.safe_load(result.final_answer)
                    return result.final_answer
                except yaml.YAMLError:
                    logger.warning("Generated resume is not valid YAML")
                    return resume_yaml

            return resume_yaml
        except Exception as e:
            logger.error(f"Error applying suggestions: {e}")
            return resume_yaml

    # Backward compatibility methods (sync versions that call async)
    def write_objective(self, **chain_kwargs) -> str:
        """Backward compatible sync version."""
        try:
            return asyncio.run(self._write_objective_async())
        except Exception as e:
            logger.error(f"Error in sync write_objective: {e}")
            return ""

    def extract_matched_skills(self, **chain_kwargs) -> list:
        """Backward compatible sync version."""
        try:
            return asyncio.run(self._extract_matched_skills_async())
        except Exception as e:
            logger.error(f"Error in sync extract_matched_skills: {e}")
            return self.skills or []

    def rewrite_unedited_experiences(self, **chain_kwargs) -> list:
        """Backward compatible sync version."""
        try:
            return asyncio.run(self._rewrite_experiences_async())
        except Exception as e:
            logger.error(f"Error in sync rewrite_experiences: {e}")
            return self.experiences or []

    def rewrite_unedited_projects(self, **chain_kwargs) -> list:
        """Backward compatible sync version."""
        try:
            return asyncio.run(self._rewrite_projects_async())
        except Exception as e:
            logger.error(f"Error in sync rewrite_projects: {e}")
            return self.projects or []

    def rewrite_section(self, section, **chain_kwargs) -> list:
        """Backward compatible sync version."""
        try:
            return asyncio.run(self._rewrite_single_section_async(section))
        except Exception as e:
            logger.error(f"Error in sync rewrite_section: {e}")
            return section.get("highlights", [])

    # Thread pool fallback for when we're in async context
    def _generate_content_parallel_threads(self, require_objective: bool = True) -> str:
        """Thread-based fallback when in async context."""
        logger.info("Using thread-based approach for async context compatibility...")

        def run_async_in_thread():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._create_complete_tailored_resume_async(require_objective))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async_in_thread)
            return future.result(timeout=self.timeout)

    # Helper methods (same as before)
    def _build_prompt_from_yaml(self, prompt_name: str, **kwargs) -> Dict[str, str]:
        """Build prompt from YAML configuration using existing LangChain prompt structure."""
        try:
            from prompts import Prompts

            # Get the list of LangChain message objects
            prompt_messages = Prompts.lookup[prompt_name]

            # Extract system message (first message)
            system_message = prompt_messages[0].content if prompt_messages else ""

            # Process the human message templates
            user_parts = []

            # Get data for formatting
            job_data = self._get_job_posting_data()
            resume_data = self._get_resume_data(**kwargs)
            all_data = {**job_data, **resume_data, **kwargs}

            # Process each message template (skip system message at index 0)
            for i, message in enumerate(prompt_messages[1:], 1):
                try:
                    if hasattr(message, 'prompt') and hasattr(message.prompt, 'template'):
                        # This is a HumanMessagePromptTemplate - format it
                        formatted_content = message.prompt.template.format(**all_data)
                        if formatted_content.strip():  # Only add non-empty content
                            user_parts.append(formatted_content)
                    elif hasattr(message, 'content'):
                        # This is a HumanMessage with static content
                        if message.content.strip():
                            user_parts.append(message.content)
                except Exception as format_error:
                    logger.warning(f"Failed to format message {i} in {prompt_name}: {format_error}")
                    # Add the raw template if formatting fails
                    if hasattr(message, 'prompt') and hasattr(message.prompt, 'template'):
                        user_parts.append(message.prompt.template)
                    elif hasattr(message, 'content'):
                        user_parts.append(message.content)

            return {
                'system': system_message,
                'user': '\n\n'.join(user_parts)
            }

        except Exception as e:
            logger.error(f"Error building prompt for {prompt_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'system': '', 'user': ''}

    def _get_job_posting_data(self) -> Dict[str, str]:
        """Get job posting data for templates."""
        if not self.parsed_job:
            return {
                'company': '',
                'job_summary': self.job_post_raw or '',
                'duties': '',
                'qualifications': '',
                'ats_keywords': '',
                'technical_skills': '',
                'non_technical_skills': '',
                'job_posting': self.job_post_raw or ''
            }

        return {
            'company': self.parsed_job.get('company', ''),
            'job_summary': self.parsed_job.get('job_summary', ''),
            'duties': '\n'.join(self.parsed_job.get('duties', [])),
            'qualifications': '\n'.join(self.parsed_job.get('qualifications', [])),
            'ats_keywords': ', '.join(self.parsed_job.get('ats_keywords', [])),
            'technical_skills': '\n'.join(self.parsed_job.get('technical_skills', [])),
            'non_technical_skills': '\n'.join(self.parsed_job.get('non_technical_skills', [])),
            'job_posting': self.job_post_raw or ''
        }

    def _get_resume_data(self, **kwargs) -> Dict[str, str]:
        """Get resume data for templates."""
        return {
            'objective': self.objective or '',
            'experiences': self._format_experiences_for_prompt(self.experiences or []),
            'projects': self._format_projects_for_prompt(self.projects or []),
            'skills': self._format_skills_for_prompt(self.skills or []),
            'education': self._format_education_for_prompt(self.education or []),
            'section': str(kwargs.get('section', '')),
            'resume_yaml': kwargs.get('resume_yaml', ''),
            'suggestions_text': kwargs.get('suggestions_text', '')
        }

    def _format_skills_for_prompt(self, skills_data: List) -> str:
        """Format skills for prompt."""
        if not skills_data:
            return ""

        result = []
        for cat in skills_data:
            if not isinstance(cat, dict):
                continue

            curr = f"{cat.get('category', '')}: "

            if "subcategories" in cat:
                skills_list = []
                for subcat in cat["subcategories"]:
                    if isinstance(subcat, dict) and "skills" in subcat:
                        skills_list.extend(subcat["skills"])
                if skills_list:
                    curr += "Proficient in " + ", ".join(skills_list)
                    result.append(curr)
            elif "skills" in cat:
                if cat["skills"]:
                    curr += "Proficient in " + ", ".join(cat["skills"])
                    result.append(curr)

        return '\n- ' + '\n- '.join(result) if result else ""

    def _format_experiences_for_prompt(self, experiences: List) -> str:
        """Format experiences for prompt."""
        if not experiences:
            return ""

        result = []
        for exp in experiences:
            if "highlights" in exp:
                curr = '\n  - ' + '\n  - '.join(exp["highlights"]) + '\n'
                result.append(curr)

        return '\n- ' + '\n- '.join(result) if result else ""

    def _format_projects_for_prompt(self, projects: List) -> str:
        """Format projects for prompt."""
        if not projects:
            return ""

        result = []
        for proj in projects:
            curr = f"Side Project: {proj.get('name', '')}"
            if "highlights" in proj:
                curr += '\n  - ' + '\n  - '.join(proj["highlights"]) + '\n'
                result.append(curr)

        return '\n- ' + '\n- '.join(result) if result else ""

    def _format_education_for_prompt(self, education: List) -> str:
        """Format education for prompt."""
        if not education:
            return ""

        formatted = []
        for entry in education:
            school = entry.get('school', '')
            degrees = ', '.join(degree.get('names', ['Degree'])[0] for degree in entry.get('degrees', []))
            formatted.append(f"{school}: {degrees}")

        return '\n'.join(formatted)

    def _format_all_suggestions_for_application(self, all_suggestions: Dict) -> str:
        """Format suggestions for application."""
        try:
            formatted_text = ""

            # Content improvements
            content_improvements = all_suggestions.get('content_improvements', [])
            if content_improvements:
                formatted_text += "## CONTENT IMPROVEMENTS:\n\n"
                for improvement_section in content_improvements:
                    section = improvement_section.section
                    improvements = improvement_section.improvements
                    formatted_text += f"### {section.upper()} SECTION:\n"
                    for i, improvement in enumerate(improvements, 1):
                        formatted_text += f"{i}. {improvement}\n"
                    formatted_text += "\n"

            # Optimization suggestions
            optimization_suggestions = all_suggestions.get('optimization_suggestions', [])
            if optimization_suggestions:
                formatted_text += "## ONE-PAGE OPTIMIZATION SUGGESTIONS:\n\n"
                for i, suggestion in enumerate(optimization_suggestions, 1):
                    formatted_text += f"{i}. {suggestion}\n"

            return formatted_text
        except Exception as e:
            logger.error(f"Error formatting suggestions: {e}")
            return "No suggestions to apply."

    def _get_default_value(self, task_name: str):
        """Get default value for failed tasks."""
        defaults = {
            'objective': "",
            'skills': self.skills or [],
            'experiences': self.experiences or [],
            'projects': self.projects or []
        }
        return defaults.get(task_name)

    # Keep all existing methods unchanged
    def download_and_parse_job_post(self, url=None):
        """Download and parse job post - unchanged."""
        if url:
            self.url = url
        self._download_url()
        self._extract_html_data()
        self.job_post = JobPost(self.job_post_raw, self.api_key)
        self.parsed_job = self.job_post.parse_job_post(verbose=False)

    def _extract_html_data(self):
        """Extract HTML data - unchanged."""
        try:
            soup = BeautifulSoup(self.job_post_html_data, "html.parser")
            self.job_post_raw = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.error(f"Failed to extract HTML data: {e}")
            raise

    def _download_url(self, url=None):
        """Download URL - unchanged."""
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
                if hasattr(response, 'status_code') and response.status_code == 429:
                    logger.warning(f"Rate limit exceeded. Retrying in {backoff_factor * 2 ** attempt} seconds...")
                    time.sleep(backoff_factor * 2 ** attempt)
                    use_proxy = True
                else:
                    logger.error(f"Failed to download URL {self.url}: {e}")
                    return False

        logger.error(f"Exceeded maximum retries for URL {self.url}")
        return False

    def _combine_skills_in_category(self, l1: list[str], l2: list[str]):
        """Combine skills - unchanged."""
        l1_lower = {i.lower() for i in l1}
        for i in l2:
            if i.lower() not in l1_lower:
                l1.append(i)

    def _combine_skill_lists(self, l1: list[dict], l2: list[dict]):
        """Combine skill lists - unchanged."""
        l1_categories_lowercase = {s["category"].lower(): i for i, s in enumerate(l1)}

        for s in l2:
            category_lower = s["category"].lower()

            if category_lower in l1_categories_lowercase:
                existing_idx = l1_categories_lowercase[category_lower]
                existing_category = l1[existing_idx]

                if "subcategories" in existing_category and "subcategories" in s:
                    existing_subcats = {sub["name"].lower(): sub for sub in existing_category["subcategories"]}

                    for new_subcat in s["subcategories"]:
                        subcat_name_lower = new_subcat["name"].lower()
                        if subcat_name_lower in existing_subcats:
                            self._combine_skills_in_category(
                                existing_subcats[subcat_name_lower]["skills"],
                                new_subcat["skills"]
                            )
                        else:
                            existing_category["subcategories"].append(new_subcat)

                elif "skills" in existing_category and "skills" in s:
                    self._combine_skills_in_category(
                        existing_category["skills"],
                        s["skills"]
                    )
            else:
                l1.append(s)

    def _get_degrees(self, resume: dict):
        """Get degrees - unchanged."""
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
        """Convert dict to YAML - unchanged."""
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
        """Cleanup - unchanged."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)