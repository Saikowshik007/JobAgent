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
    Parallel ResumeImprover optimized for ONE-PAGE resume generation.
    """

    # ONE-PAGE CONSTRAINTS
    MAX_EXPERIENCES = 3  # Show only most recent/relevant experiences
    MAX_HIGHLIGHTS_PER_EXPERIENCE = 3  # Reduced from 5
    MAX_PROJECTS = 2  # Show only most relevant projects
    MAX_HIGHLIGHTS_PER_PROJECT = 2  # Reduced from 3
    MAX_TECHNICAL_SKILLS = 20  # Total technical skills
    MAX_SKILL_CATEGORIES = 4  # Maximum skill categories
    MAX_NON_TECHNICAL_SKILLS = 6  # Soft skills

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
        Create complete tailored ONE-PAGE resume with parallel processing.
        """
        try:
            logger.info("=== Creating Complete One-Page Tailored Resume (Parallel) ===")

            # Pre-filter content for one-page constraint
            self._filter_for_one_page()

            # Try parallel execution first
            try:
                start_time = time.time()

                # Check if we're in an async context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        logger.info("Using thread-based parallel approach (async context detected)")
                        results = self._generate_content_parallel_threads()
                    else:
                        logger.info("Using asyncio-based parallel approach")
                        results = asyncio.run(self._generate_content_async_parallel())
                except RuntimeError:
                    logger.info("Using asyncio-based parallel approach (no existing loop)")
                    results = asyncio.run(self._generate_content_async_parallel())

                end_time = time.time()
                logger.info(f"Parallel generation completed in {end_time - start_time:.2f} seconds")

            except Exception as parallel_error:
                logger.warning(f"Parallel execution failed: {parallel_error}, falling back to sequential")
                results = self._generate_content_sequential()

            # Extract results
            objective = results.get('objective')
            skills = results.get('skills', [])
            experiences = results.get('experiences', [])
            projects = results.get('projects', [])

            # Apply final one-page constraints
            experiences = self._limit_experiences(experiences)
            projects = self._limit_projects(projects)
            skills = self._limit_skills(skills)

            logger.info(f"One-page results summary:")
            logger.info(f"  - Objective: {'✓' if objective else '✗'} ({len(objective.split()) if objective else 0} words)")
            logger.info(f"  - Skills: {self._count_skills(skills)} skills in {len(skills)} categories")
            logger.info(f"  - Experiences: {len(experiences)} items (max {self.MAX_EXPERIENCES})")
            logger.info(f"  - Projects: {len(projects)} items (max {self.MAX_PROJECTS})")

            # Create final one-page resume
            final_resume = {
                'editing': False,
                'basic': self.basic_info or {},
                'objective': objective,
                'education': self._limit_education(self.education or []),
                'experiences': experiences,
                'projects': projects,
                'skills': skills,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'job_url': self.url,
                    'tailored': True,
                    'format': 'one-page'
                }
            }

            yaml_content = self.dict_to_yaml_string(final_resume)
            logger.info("=== One-Page Resume Creation Complete ===")
            return yaml_content

        except Exception as e:
            logger.error(f"Complete resume creation failed: {e}")
            raise

    def _filter_for_one_page(self):
        """Pre-filter resume content to prepare for one-page generation."""
        logger.info("Pre-filtering content for one-page constraint...")

        # Limit experiences to most recent/relevant
        if self.experiences and len(self.experiences) > self.MAX_EXPERIENCES:
            logger.info(f"Filtering experiences from {len(self.experiences)} to {self.MAX_EXPERIENCES}")
            # Sort by most recent first (if dates available)
            self.experiences = self._get_most_relevant_experiences(self.experiences)[:self.MAX_EXPERIENCES]

        # Limit projects
        if self.projects and len(self.projects) > self.MAX_PROJECTS:
            logger.info(f"Filtering projects from {len(self.projects)} to {self.MAX_PROJECTS}")
            self.projects = self._get_most_relevant_projects(self.projects)[:self.MAX_PROJECTS]

    def _get_most_relevant_experiences(self, experiences: List[dict]) -> List[dict]:
        """Get the most relevant experiences based on recency and relevance to job."""
        # Sort by end date (most recent first)
        def get_end_date(exp):
            if 'titles' in exp and exp['titles']:
                latest_title = max(exp['titles'], key=lambda t: self._parse_date_for_sorting(t.get('enddate', '2000-01-01')))
                return self._parse_date_for_sorting(latest_title.get('enddate', '2000-01-01'))
            return parse_date('2000-01-01')

        return sorted(experiences, key=get_end_date, reverse=True)

    def _parse_date_for_sorting(self, date_str: str) -> datetime:
        """Parse date for sorting purposes, handling 'Present' as today's date."""
        if not date_str:
            return parse_date('2000-01-01')

        date_str_lower = str(date_str).lower().strip()
        if date_str_lower in ['present', 'current', 'now', 'ongoing', 'today']:
            return datetime.today()

        try:
            return parse_date(date_str)
        except:
            return parse_date('2000-01-01')

    def _get_most_relevant_projects(self, projects: List[dict]) -> List[dict]:
        """Get the most relevant projects based on technology match."""
        if not self.parsed_job:
            return projects[:self.MAX_PROJECTS]

        job_tech_skills = set(self.parsed_job.get('technical_skills', []))

        def project_relevance_score(project):
            score = 0
            # Check project highlights for technology matches
            for highlight in project.get('highlights', []):
                for skill in job_tech_skills:
                    if skill.lower() in highlight.lower():
                        score += 1
            return score

        return sorted(projects, key=project_relevance_score, reverse=True)

    def _limit_experiences(self, experiences: List[dict]) -> List[dict]:
        """Limit experiences to fit one-page constraint."""
        return experiences[:self.MAX_EXPERIENCES]

    def _limit_projects(self, projects: List[dict]) -> List[dict]:
        """Limit projects to fit one-page constraint."""
        return projects[:self.MAX_PROJECTS]

    def _limit_education(self, education: List[dict]) -> List[dict]:
        """Limit education entries - keep only highest degree."""
        if len(education) <= 1:
            return education

        # Keep only the highest/most recent degree
        return [education[0]]  # Assuming they're ordered by relevance

    def _limit_skills(self, skills: List[dict]) -> List[dict]:
        """Limit skills to fit one-page constraint."""
        limited_skills = []
        total_technical_skills = 0

        for category in skills:
            if category.get('category') == 'Technical' and 'subcategories' in category:
                # Limit technical subcategories
                limited_subcats = []
                for subcat in category['subcategories'][:self.MAX_SKILL_CATEGORIES]:
                    remaining_slots = self.MAX_TECHNICAL_SKILLS - total_technical_skills
                    if remaining_slots > 0:
                        limited_skills_list = subcat['skills'][:remaining_slots]
                        if limited_skills_list:
                            limited_subcats.append({
                                'name': subcat['name'],
                                'skills': limited_skills_list
                            })
                            total_technical_skills += len(limited_skills_list)

                if limited_subcats:
                    limited_skills.append({
                        'category': 'Technical',
                        'subcategories': limited_subcats
                    })

            elif category.get('category') == 'Non-technical' and 'skills' in category:
                # Limit non-technical skills
                limited_skills.append({
                    'category': 'Non-technical',
                    'skills': category['skills'][:self.MAX_NON_TECHNICAL_SKILLS]
                })

        return limited_skills

    def _count_skills(self, skills: List[dict]) -> int:
        """Count total number of skills."""
        count = 0
        for category in skills:
            if 'subcategories' in category:
                for subcat in category['subcategories']:
                    count += len(subcat.get('skills', []))
            elif 'skills' in category:
                count += len(category['skills'])
        return count

    def rewrite_section(self, section, **chain_kwargs) -> list:
        """Rewrite a section with ONE-PAGE constraints."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSectionHighlighterOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            logger.debug(f"Starting one-page rewrite_section for: {section.get('title') or section.get('name', 'Unknown')}")

            prompt = ChatPromptTemplate(messages=Prompts.lookup["SECTION_HIGHLIGHTER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeSectionHighlighterOutput)

            chain_inputs = self._get_formatted_chain_inputs(chain=chain, section=section)
            section_revised = chain.invoke(chain_inputs)

            if section_revised:
                if hasattr(section_revised, 'final_answer'):
                    highlights = section_revised.final_answer or []
                    if highlights:
                        # Sort by relevance and apply strict one-page limits
                        sorted_highlights = sorted(highlights, key=lambda d: d.relevance * -1)

                        # One-page limits
                        section_type = self._determine_section_type(section)
                        if section_type == 'experience':
                            limit = self.MAX_HIGHLIGHTS_PER_EXPERIENCE
                        elif section_type == 'project':
                            limit = self.MAX_HIGHLIGHTS_PER_PROJECT
                        else:
                            limit = 2  # Default for other sections

                        # Only include high-relevance highlights (4-5 rating)
                        high_relevance_highlights = [h for h in sorted_highlights if h.relevance >= 4]
                        limited_highlights = high_relevance_highlights[:limit]

                        result = [s.highlight for s in limited_highlights]
                        logger.info(f"One-page constraint: Selected {len(result)} of {len(highlights)} highlights (relevance >= 4)")
                        return result

            return section.get("highlights", [])[:self.MAX_HIGHLIGHTS_PER_EXPERIENCE]

        except Exception as e:
            logger.error(f"Error in rewrite_section: {e}")
            return section.get("highlights", [])[:self.MAX_HIGHLIGHTS_PER_EXPERIENCE]

    # Include all other methods from the original file with the same implementation
    # (All the async/parallel methods, download methods, etc. remain the same)

    async def _generate_content_async_parallel(self) -> Dict:
        """Generate all resume content in parallel using asyncio.gather."""
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

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout
            )
            logger.info(f"All {len(results)} tasks completed")
        except asyncio.TimeoutError:
            logger.error(f"Parallel generation timed out after {self.timeout} seconds")
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = [None, [], [], []]

        processed_results = {}
        for i, (result, task_name) in enumerate(zip(results, task_names)):
            if isinstance(result, Exception):
                logger.error(f"Task '{task_name}' failed with exception: {result}")
                processed_results[task_name] = self._get_default_value(task_name)
            else:
                logger.info(f"Task '{task_name}' completed successfully")
                processed_results[task_name] = result

        return processed_results

    # Include all other helper methods from original file...
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

    # Copy all remaining methods from the original file exactly as they are
    def download_and_parse_job_post(self, url=None):
        """Download and parse the job post from the provided URL."""
        if url:
            self.url = url
        self._download_url()
        self._extract_html_data()
        self.job_post = JobPost(self.job_post_raw, self.api_key)
        self.parsed_job = self.job_post.parse_job_post(verbose=False)

    def _extract_html_data(self):
        """Extract text content from HTML, removing all HTML tags."""
        try:
            soup = BeautifulSoup(self.job_post_html_data, "html.parser")
            self.job_post_raw = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            config.logger.error(f"Failed to extract HTML data: {e}")
            raise

    def _download_url(self, url=None):
        """Download the content of the URL and return it as a string."""
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

            prompt = ChatPromptTemplate(messages=Prompts.lookup["OBJECTIVE_WRITER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeSummarizerOutput)

            chain_inputs = self._get_formatted_chain_inputs(chain=chain)
            logger.debug(f"Objective chain inputs: {list(chain_inputs.keys())}")

            result = chain.invoke(chain_inputs)
            if result:
                if hasattr(result, 'final_answer'):
                    objective = result.final_answer
                    logger.info("Using Pydantic model access")
                elif isinstance(result, dict):
                    objective = result.get('final_answer')
                    logger.info("Using dictionary access")
                else:
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

            runnable = chain | llm.with_structured_output(schema=ResumeSkillsMatcherOutput, method="function_calling")

            chain_inputs = self._get_formatted_chain_inputs(chain=runnable)
            extracted_skills = runnable.invoke(chain_inputs)

            if not extracted_skills:
                logger.warning("No extracted_skills returned from LLM")
                return self.skills or []

            if hasattr(extracted_skills, 'final_answer'):
                extracted_skills_dict = extracted_skills.final_answer
            elif isinstance(extracted_skills, dict):
                extracted_skills_dict = extracted_skills.get("final_answer", {})
            else:
                logger.error(f"Unexpected response type: {type(extracted_skills)}")
                return self.skills or []

            logger.info(f"LLM returned skills: {extracted_skills_dict}")

            result = []

            if hasattr(extracted_skills_dict, 'technical_skills'):
                technical_skills = extracted_skills_dict.technical_skills or {}
            elif isinstance(extracted_skills_dict, dict):
                technical_skills = extracted_skills_dict.get("technical_skills", {})
            else:
                technical_skills = {}

            if technical_skills and isinstance(technical_skills, dict):
                subcategories = []
                for category_name, skills_list in technical_skills.items():
                    if skills_list:
                        subcategories.append({
                            "name": category_name,
                            "skills": skills_list
                        })

                if subcategories:
                    result.append({
                        "category": "Technical",
                        "subcategories": subcategories
                    })

            if hasattr(extracted_skills_dict, 'non_technical_skills'):
                non_technical_skills = extracted_skills_dict.non_technical_skills or []
            elif isinstance(extracted_skills_dict, dict):
                non_technical_skills = extracted_skills_dict.get("non_technical_skills", [])
            else:
                non_technical_skills = []

            if non_technical_skills and isinstance(non_technical_skills, list):
                result.append({
                    "category": "Non-technical",
                    "skills": non_technical_skills
                })

            logger.info(f"Final skills structure: {len(result)} categories")
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

                original_highlights = exp.get("highlights", [])
                logger.info(f"  Original highlights: {len(original_highlights)} items")

                new_highlights = self.rewrite_section(section=exp, **chain_kwargs)
                logger.info(f"  New highlights: {len(new_highlights) if new_highlights else 0} items")

                if new_highlights:
                    exp["highlights"] = new_highlights
                else:
                    logger.warning(f"  No new highlights generated, keeping original")
                    exp["highlights"] = original_highlights[:self.MAX_HIGHLIGHTS_PER_EXPERIENCE]

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

                original_highlights = proj.get("highlights", [])
                logger.info(f"  Original highlights: {len(original_highlights)} items")

                new_highlights = self.rewrite_section(section=proj, **chain_kwargs)
                logger.info(f"  New highlights: {len(new_highlights) if new_highlights else 0} items")

                if new_highlights:
                    proj["highlights"] = new_highlights
                else:
                    logger.warning(f"  No new highlights generated, keeping original")
                    proj["highlights"] = original_highlights[:self.MAX_HIGHLIGHTS_PER_PROJECT]

                result.append(proj)

            logger.info(f"Completed rewriting {len(result)} projects")
            return result
        except Exception as e:
            logger.error(f"Error in rewrite_unedited_projects: {e}")
            import traceback
            logger.error(f"Project rewrite traceback: {traceback.format_exc()}")
            return self.projects or []

    def _determine_section_type(self, section) -> str:
        """Determine if section is experience or project based on its structure."""
        if 'titles' in section or 'title' in section or 'company' in section:
            return 'experience'
        elif 'name' in section:
            return 'project'
        else:
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

            if key == "skills" and self.skills:
                value = self.skills
                logger.debug(f"Passing raw skills structure to chain_formatter: {len(self.skills)} categories")

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

                elif "subcategories" in existing_category and "skills" in s:
                    if not any(sub["name"].lower() == "general" for sub in existing_category["subcategories"]):
                        existing_category["subcategories"].append({
                            "name": "General",
                            "skills": s["skills"][:]
                        })
                    else:
                        for sub in existing_category["subcategories"]:
                            if sub["name"].lower() == "general":
                                self._combine_skills_in_category(sub["skills"], s["skills"])
                                break

                elif "skills" in existing_category and "subcategories" in s:
                    existing_skills = existing_category["skills"][:]
                    existing_category["subcategories"] = [
                        {"name": "General", "skills": existing_skills}
                    ]
                    del existing_category["skills"]

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

    def _generate_content_parallel_threads(self) -> Dict:
        """Generate content using ThreadPoolExecutor for cases where we're already in async context."""
        logger.info("Using ThreadPoolExecutor for parallel generation...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(self._safe_write_objective): 'objective',
                executor.submit(self._safe_extract_matched_skills): 'skills',
                executor.submit(self._safe_rewrite_experiences): 'experiences',
                executor.submit(self._safe_rewrite_projects): 'projects'
            }

            results = {}
            completed_tasks = 0
            total_tasks = len(future_to_task)

            try:
                for future in concurrent.futures.as_completed(future_to_task, timeout=self.timeout):
                    task_name = future_to_task[future]
                    completed_tasks += 1

                    try:
                        result = future.result()
                        results[task_name] = result
                        logger.info(f"✓ {task_name} completed successfully ({completed_tasks}/{total_tasks})")
                    except Exception as e:
                        logger.error(f"✗ {task_name} failed with exception: {e}")
                        results[task_name] = self._get_default_value(task_name)

            except concurrent.futures.TimeoutError:
                logger.error(f"Parallel generation timed out after {self.timeout} seconds")

                for future in future_to_task:
                    if not future.done():
                        future.cancel()
                        task_name = future_to_task[future]
                        logger.warning(f"Cancelled task: {task_name}")

                for task_name in ['objective', 'skills', 'experiences', 'projects']:
                    if task_name not in results:
                        results[task_name] = self._get_default_value(task_name)

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
        logger.debug(f"Using default for {task_name}: {type(default_value)}")
        return default_value

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)