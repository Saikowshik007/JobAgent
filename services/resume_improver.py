import yaml
import requests
from bs4 import BeautifulSoup
from fp.fp import FreeProxy
from requests import RequestException
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
    ResumeImprover with both parallel and cumulative generation methods.
    The cumulative approach optimizes for ATS by building context sequentially.
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

        # Thread pool for running sync LLM calls
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def create_cumulative_tailored_resume(self, include_objective: bool = True) -> str:
        """
        NEW: Create complete tailored resume using cumulative approach for ATS optimization.
        Each section builds upon previous sections for better keyword distribution and narrative coherence.
        """
        try:
            logger.info("=== Creating Cumulative ATS-Optimized Resume ===")
            start_time = time.time()

            # Step 1: Comprehensive job analysis (new step)
            logger.info("Step 1: Analyzing job posting for ATS optimization...")
            job_analysis = self._analyze_job_posting_for_ats()

            # Step 2: Build resume context cumulatively
            resume_context = {
                'job_analysis': job_analysis,
                'resume_so_far': {}
            }

            # Step 3: Generate objective with job analysis context
            if include_objective:
                logger.info("Step 2: Generating objective with job context...")
                objective = self._generate_objective_with_context(resume_context)
                resume_context['resume_so_far']['objective'] = objective
                logger.info(f"✓ Objective generated: {len(objective) if objective else 0} chars")
            else:
                objective = None
                logger.info("Step 2: Skipping objective generation")

            # Step 4: Generate skills with objective context
            logger.info("Step 3: Generating skills with cumulative context...")
            skills = self._generate_skills_with_context(resume_context)
            resume_context['resume_so_far']['skills'] = skills
            logger.info(f"✓ Skills generated: {len(skills)} categories")

            # Step 5: Generate experiences with objective + skills context
            logger.info("Step 4: Generating experiences with cumulative context...")
            experiences = self._generate_experiences_with_context(resume_context)
            resume_context['resume_so_far']['experiences'] = experiences
            logger.info(f"✓ Experiences generated: {len(experiences)} items")

            # Step 6: Generate projects with full context
            logger.info("Step 5: Generating projects with full context...")
            projects = self._generate_projects_with_context(resume_context)
            resume_context['resume_so_far']['projects'] = projects
            logger.info(f"✓ Projects generated: {len(projects)} items")

            # Step 7: Global ATS optimization pass (new step)
            logger.info("Step 6: Performing global ATS optimization...")
            optimized_resume_data = self._perform_global_ats_optimization(resume_context)

            # Step 8: Assemble final resume
            logger.info("Step 7: Assembling final resume...")
            final_resume = {
                'editing': False,
                'basic': self.basic_info or {},
                'objective': optimized_resume_data.get('objective', objective),
                'education': self.education or [],
                'experiences': optimized_resume_data.get('experiences', experiences),
                'projects': optimized_resume_data.get('projects', projects),
                'skills': optimized_resume_data.get('skills', skills),
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'job_url': self.url,
                    'tailored': True,
                    'approach': 'cumulative_ats_optimized',
                    'ats_score': optimized_resume_data.get('ats_score', 'not_calculated')
                }
            }

            # Step 9: Convert to YAML
            yaml_content = self.dict_to_yaml_string(final_resume)

            end_time = time.time()
            logger.info(f"=== Cumulative Resume Creation Complete in {end_time - start_time:.2f} seconds ===")
            return yaml_content

        except Exception as e:
            logger.error(f"Cumulative resume creation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _analyze_job_posting_for_ats(self) -> Dict:
        """
        NEW: Comprehensive job analysis for ATS optimization.
        Extracts keywords, priorities, and distribution strategy.
        """
        try:
            logger.info("Analyzing job posting for ATS keywords and requirements...")

            # Ensure we have job posting data
            if not self.job_post_raw and not self.parsed_job:
                self.download_and_parse_job_post()

            # Extract comprehensive keyword analysis
            job_analysis = {
                'primary_keywords': [],
                'secondary_keywords': [],
                'technical_skills': [],
                'soft_skills': [],
                'company_specific_terms': [],
                'ats_priority_sections': {
                    'objective': [],
                    'skills': [],
                    'experiences': [],
                    'projects': []
                }
            }

            # If we have parsed job data, use it
            if self.parsed_job:
                job_analysis['primary_keywords'] = self.parsed_job.get('ats_keywords', [])[:10]
                job_analysis['technical_skills'] = self.parsed_job.get('technical_skills', [])
                job_analysis['soft_skills'] = self.parsed_job.get('non_technical_skills', [])

                # Extract company-specific terms
                company = self.parsed_job.get('company', '')
                if company:
                    job_analysis['company_specific_terms'].append(company)

                # Distribute keywords across sections for optimal ATS scoring
                all_keywords = job_analysis['primary_keywords']
                job_analysis['ats_priority_sections'] = {
                    'objective': all_keywords[:3],  # Top 3 keywords in objective
                    'skills': job_analysis['technical_skills'] + job_analysis['primary_keywords'],
                    'experiences': job_analysis['primary_keywords'] + job_analysis['soft_skills'],
                    'projects': job_analysis['technical_skills'][:5]  # Top 5 technical skills in projects
                }

            logger.info(f"ATS Analysis complete: {len(job_analysis['primary_keywords'])} primary keywords, "
                        f"{len(job_analysis['technical_skills'])} technical skills")

            return job_analysis

        except Exception as e:
            logger.error(f"Error in ATS job analysis: {e}")
            return {'primary_keywords': [], 'technical_skills': [], 'soft_skills': [],
                    'ats_priority_sections': {'objective': [], 'skills': [], 'experiences': [], 'projects': []}}

    def _generate_objective_with_context(self, resume_context: Dict) -> Optional[str]:
        """Generate objective with job analysis context for better ATS optimization."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSummarizerOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            logger.debug("Generating objective with ATS context...")

            # Create enhanced chain inputs with ATS context
            prompt = ChatPromptTemplate(messages=Prompts.lookup["OBJECTIVE_WRITER"])
            llm = create_llm(self.user, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeSummarizerOutput)

            # Get standard inputs and enhance with ATS context
            chain_inputs = self._get_formatted_chain_inputs(chain=chain)

            # Add ATS context to inputs
            job_analysis = resume_context['job_analysis']
            chain_inputs['ats_priority_keywords'] = ', '.join(job_analysis['ats_priority_sections']['objective'])

            logger.debug(f"Objective generation with ATS keywords: {chain_inputs.get('ats_priority_keywords', 'none')}")

            result = chain.invoke(chain_inputs)
            if result:
                if hasattr(result, 'final_answer'):
                    objective = result.final_answer
                elif isinstance(result, dict):
                    objective = result.get('final_answer')
                else:
                    objective = result

                logger.debug(f"Generated objective: {objective}")
                return objective

            logger.warning("Objective generation returned None")
            return None

        except Exception as e:
            logger.error(f"Error in objective generation with context: {e}")
            return None

    def _generate_skills_with_context(self, resume_context: Dict) -> List:
        """Generate skills with cumulative context (objective + job analysis)."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSkillsMatcherOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            logger.debug("Generating skills with cumulative context...")

            chain = ChatPromptTemplate(messages=Prompts.lookup["SKILLS_MATCHER"])
            llm = create_llm(self.user, **self.llm_kwargs)
            runnable = chain | llm.with_structured_output(schema=ResumeSkillsMatcherOutput, method="function_calling")

            # Get standard inputs and enhance with cumulative context
            chain_inputs = self._get_formatted_chain_inputs(chain=runnable)

            # Add cumulative context
            job_analysis = resume_context['job_analysis']
            resume_so_far = resume_context['resume_so_far']

            chain_inputs['ats_required_skills'] = ', '.join(job_analysis['technical_skills'])
            chain_inputs['objective_context'] = resume_so_far.get('objective', '')

            logger.debug(f"Skills generation with {len(job_analysis['technical_skills'])} required technical skills")

            extracted_skills = runnable.invoke(chain_inputs)

            if not extracted_skills:
                logger.warning("No extracted_skills returned from LLM")
                return self.skills or []

            # Process the result
            if hasattr(extracted_skills, 'final_answer'):
                extracted_skills_dict = extracted_skills.final_answer
            elif isinstance(extracted_skills, dict):
                extracted_skills_dict = extracted_skills.get("final_answer", {})
            else:
                logger.error(f"Unexpected response type: {type(extracted_skills)}")
                return self.skills or []

            logger.info(f"LLM returned skills: {extracted_skills_dict}")

            # Build the final skills structure
            result = []

            # Handle technical skills
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

            # Handle non-technical skills
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

            logger.info(f"Skills with context: {len(result)} categories")
            return result

        except Exception as e:
            logger.error(f"Error in skills generation with context: {e}")
            import traceback
            logger.error(f"Skills context traceback: {traceback.format_exc()}")
            return self.skills or []

    def _generate_experiences_with_context(self, resume_context: Dict) -> List:
        """Generate experiences with cumulative context (objective + skills + job analysis)."""
        try:
            if not self.experiences:
                logger.info("No experiences to rewrite")
                return []

            logger.info(f"Generating experiences with cumulative context for {len(self.experiences)} items...")
            result = []

            # Get context for experience generation
            job_analysis = resume_context['job_analysis']
            resume_so_far = resume_context['resume_so_far']

            for i, exp in enumerate(self.experiences):
                logger.info(f"Processing experience {i + 1}: {exp.get('title', 'Unknown')}")
                exp = dict(exp)

                # Enhanced context for this experience
                experience_context = {
                    'job_analysis': job_analysis,
                    'resume_so_far': resume_so_far,
                    'section_type': 'experience',
                    'ats_keywords': job_analysis['ats_priority_sections']['experiences']
                }

                new_highlights = self._rewrite_section_with_context(exp, experience_context)

                if new_highlights:
                    logger.info(f"  Generated {len(new_highlights)} new highlights")
                    exp["highlights"] = new_highlights
                else:
                    logger.warning(f"  No new highlights generated, keeping original")
                    exp["highlights"] = exp.get("highlights", [])

                result.append(exp)

            logger.info(f"Completed experiences generation with context: {len(result)} items")
            return result

        except Exception as e:
            logger.error(f"Error in experiences generation with context: {e}")
            import traceback
            logger.error(f"Experience context traceback: {traceback.format_exc()}")
            return self.experiences or []

    def _generate_projects_with_context(self, resume_context: Dict) -> List:
        """Generate projects with full cumulative context."""
        try:
            if not self.projects:
                logger.info("No projects to rewrite")
                return []

            logger.info(f"Generating projects with full context for {len(self.projects)} items...")
            result = []

            # Get full context for project generation
            job_analysis = resume_context['job_analysis']
            resume_so_far = resume_context['resume_so_far']

            for i, proj in enumerate(self.projects):
                logger.info(f"Processing project {i + 1}: {proj.get('name', 'Unknown')}")
                proj = dict(proj)

                # Enhanced context for this project
                project_context = {
                    'job_analysis': job_analysis,
                    'resume_so_far': resume_so_far,
                    'section_type': 'project',
                    'ats_keywords': job_analysis['ats_priority_sections']['projects']
                }

                new_highlights = self._rewrite_section_with_context(proj, project_context)

                if new_highlights:
                    logger.info(f"  Generated {len(new_highlights)} new highlights")
                    proj["highlights"] = new_highlights
                else:
                    logger.warning(f"  No new highlights generated, keeping original")
                    proj["highlights"] = proj.get("highlights", [])

                result.append(proj)

            logger.info(f"Completed projects generation with context: {len(result)} items")
            return result

        except Exception as e:
            logger.error(f"Error in projects generation with context: {e}")
            import traceback
            logger.error(f"Project context traceback: {traceback.format_exc()}")
            return self.projects or []

    def _rewrite_section_with_context(self, section: Dict, context: Dict) -> List:
        """
        Enhanced section rewriting with cumulative context and ATS optimization.
        """
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSectionHighlighterOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            section_name = section.get('title') or section.get('name', 'Unknown')
            logger.debug(f"Rewriting section with context: {section_name}")

            prompt = ChatPromptTemplate(messages=Prompts.lookup["SECTION_HIGHLIGHTER"])
            llm = create_llm(self.user, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeSectionHighlighterOutput)

            # Get base inputs and enhance with context
            chain_inputs = self._get_formatted_chain_inputs(chain=chain, section=section)

            # Add contextual information for ATS optimization
            chain_inputs['ats_keywords'] = ', '.join(context.get('ats_keywords', []))
            chain_inputs['resume_context'] = self._format_resume_context(context['resume_so_far'])
            chain_inputs['section_priority'] = context.get('section_type', 'unknown')

            logger.debug(f"Section rewrite with ATS keywords: {chain_inputs.get('ats_keywords', 'none')}")

            section_revised = chain.invoke(chain_inputs)

            if section_revised:
                if hasattr(section_revised, 'final_answer'):
                    highlights = section_revised.final_answer or []
                elif isinstance(section_revised, dict):
                    highlights = section_revised.get("final_answer", [])
                else:
                    logger.error(f"Unexpected response type: {type(section_revised)}")
                    return section.get("highlights", [])

                if highlights:
                    def get_relevance(highlight):
                        if hasattr(highlight, 'relevance'):
                            # Pydantic object
                            return highlight.relevance
                        elif isinstance(highlight, dict):
                            # Dictionary
                            return highlight.get('relevance', 0)
                        else:
                            return 0

                    sorted_highlights = sorted(highlights, key=lambda d: get_relevance(d) * -1)

                    # Determine limit based on section type
                    section_type = context.get('section_type', self._determine_section_type(section))
                    limit = 4 if section_type == 'experience' else 2 if section_type == 'project' else len(sorted_highlights)

                    limited_highlights = sorted_highlights[:limit]

                    # Extract highlight text properly
                    result = []
                    for highlight_obj in limited_highlights:
                        if hasattr(highlight_obj, 'highlight'):
                            # Pydantic object
                            result.append(highlight_obj.highlight)
                        elif isinstance(highlight_obj, dict):
                            # Dictionary
                            result.append(highlight_obj.get("highlight", ""))
                        else:
                            # Fallback
                            result.append(str(highlight_obj))

                    logger.debug(f"Generated {len(result)} contextual highlights")
                    return result

            logger.warning(f"No valid highlights generated for {section_name}, returning original")
            return section.get("highlights", [])

        except Exception as e:
            logger.error(f"Error in contextual section rewrite: {e}")
            import traceback
            logger.error(f"Context rewrite traceback: {traceback.format_exc()}")
            return section.get("highlights", [])

    def _format_resume_context(self, resume_so_far: Dict) -> str:
        """Format the resume context for LLM input."""
        context_parts = []

        if resume_so_far.get('objective'):
            context_parts.append(f"Objective: {resume_so_far['objective']}")

        if resume_so_far.get('skills'):
            skills_summary = f"Skills: {len(resume_so_far['skills'])} categories"
            context_parts.append(skills_summary)

        if resume_so_far.get('experiences'):
            exp_summary = f"Experiences: {len(resume_so_far['experiences'])} positions"
            context_parts.append(exp_summary)

        return " | ".join(context_parts)

    def _perform_global_ats_optimization(self, resume_context: Dict) -> Dict:
        """
        NEW: Perform global ATS optimization pass on the complete resume.
        Ensures optimal keyword distribution and ATS scoring.
        """
        try:
            logger.info("Performing global ATS optimization...")

            job_analysis = resume_context['job_analysis']
            resume_data = resume_context['resume_so_far']

            # Calculate current ATS score
            ats_score = self._calculate_ats_score(resume_data, job_analysis)
            logger.info(f"Current ATS score: {ats_score}")

            # If score is already good (>85), return as-is
            if ats_score >= 85:
                logger.info("ATS score already optimal, skipping global optimization")
                return {**resume_data, 'ats_score': ats_score}

            # Perform optimization (simplified version for now)
            optimized_data = self._optimize_keyword_distribution(resume_data, job_analysis)

            # Recalculate score
            final_score = self._calculate_ats_score(optimized_data, job_analysis)
            logger.info(f"Optimized ATS score: {final_score}")

            return {**optimized_data, 'ats_score': final_score}

        except Exception as e:
            logger.error(f"Error in global ATS optimization: {e}")
            return {**resume_context['resume_so_far'], 'ats_score': 'optimization_failed'}

    def _calculate_ats_score(self, resume_data: Dict, job_analysis: Dict) -> float:
        """Calculate a simplified ATS compatibility score."""
        try:
            total_score = 0
            max_score = 100

            # Keyword coverage (40 points)
            primary_keywords = job_analysis.get('primary_keywords', [])
            technical_skills = job_analysis.get('technical_skills', [])

            if primary_keywords:
                resume_text = str(resume_data).lower()
                keyword_hits = sum(1 for keyword in primary_keywords if keyword.lower() in resume_text)
                keyword_score = (keyword_hits / len(primary_keywords)) * 40
                total_score += keyword_score

            # Technical skills coverage (30 points)
            if technical_skills:
                skills_text = str(resume_data.get('skills', [])).lower()
                tech_hits = sum(1 for skill in technical_skills if skill.lower() in skills_text)
                tech_score = (tech_hits / len(technical_skills)) * 30
                total_score += tech_score

            # Section completeness (30 points)
            required_sections = ['objective', 'skills', 'experiences']
            section_score = sum(10 for section in required_sections if resume_data.get(section))
            total_score += section_score

            return min(total_score, max_score)

        except Exception as e:
            logger.error(f"Error calculating ATS score: {e}")
            return 50.0  # Default fallback score

    def _optimize_keyword_distribution(self, resume_data: Dict, job_analysis: Dict) -> Dict:
        """Optimize keyword distribution across resume sections."""
        try:
            # For now, return the original data
            # In a full implementation, this would:
            # 1. Identify keyword gaps
            # 2. Redistribute keywords optimally
            # 3. Ensure natural integration
            logger.info("Keyword optimization placeholder - returning original data")
            return resume_data

        except Exception as e:
            logger.error(f"Error in keyword optimization: {e}")
            return resume_data

    # Keep all existing methods for backward compatibility
    def create_complete_tailored_resume(self, include_objective) -> str:
        """
        EXISTING: Original parallel method - kept for backward compatibility.
        """
        try:
            logger.info("=== Creating Complete Tailored Resume (Parallel - Legacy) ===")

            # Try parallel execution first
            try:
                start_time = time.time()

                # Check if we're in an async context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        logger.info(f'Using thread-based parallel approach (async context detected) using model: {self.user.model}')
                        results = self._generate_content_parallel_threads(include_objective)
                    else:
                        logger.info("Using asyncio-based parallel approach")
                        results = asyncio.run(self._generate_content_async_parallel(include_objective))
                except RuntimeError:
                    logger.info("Using asyncio-based parallel approach (no existing loop)")
                    results = asyncio.run(self._generate_content_async_parallel(include_objective))

                end_time = time.time()
                logger.info(f"Parallel generation completed in {end_time - start_time:.2f} seconds")

            except Exception as parallel_error:
                logger.warning(f"Parallel execution failed: {parallel_error}, falling back to sequential")
                results = self._generate_content_sequential(include_objective)

            # Extract results
            objective = results.get('objective', "")
            skills = results.get('skills', [])
            experiences = results.get('experiences', [])
            projects = results.get('projects', [])

            logger.info(f"Results summary:")
            logger.info(f"  - Objective: {'✓' if objective else '✗'}")
            logger.info(f"  - Skills: {len(skills)} categories")
            logger.info(f"  - Experiences: {len(experiences)} items")
            logger.info(f"  - Projects: {len(projects)} items")

            # Create final resume
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
                    'approach': 'parallel_legacy'
                }
            }

            yaml_content = self.dict_to_yaml_string(final_resume)
            logger.info("=== Resume Creation Complete (Legacy) ===")
            return yaml_content

        except Exception as e:
            logger.error(f"Complete resume creation failed: {e}")
            raise

    async def _generate_content_async_parallel(self, include_objective: bool = True) -> Dict:
        """EXISTING: Generate all resume content in parallel using asyncio.gather."""
        if not hasattr(self, 'executor'):
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        loop = asyncio.get_event_loop()
        logger.info("Starting parallel task execution...")

        tasks = []
        task_names = []

        if include_objective:
            tasks.append(loop.run_in_executor(self.executor, self._safe_write_objective))
            task_names.append('objective')

        tasks.extend([
            loop.run_in_executor(self.executor, self._safe_extract_matched_skills),
            loop.run_in_executor(self.executor, self._safe_rewrite_experiences),
            loop.run_in_executor(self.executor, self._safe_rewrite_projects)
        ])
        task_names.extend(['skills', 'experiences', 'projects'])

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
            default_results = []
            if include_objective:
                default_results.append(None)
            default_results.extend([[], [], []])
            results = default_results

        processed_results = {}
        for i, (result, task_name) in enumerate(zip(results, task_names)):
            if isinstance(result, Exception):
                logger.error(f"Task '{task_name}' failed with exception: {result}")
                processed_results[task_name] = self._get_default_value(task_name)
            else:
                logger.info(f"Task '{task_name}' completed successfully")
                processed_results[task_name] = result

        if not include_objective:
            processed_results['objective'] = None

        return processed_results

    def _generate_content_parallel_threads(self, include_objective: bool = True) -> Dict:
        """EXISTING: Generate content using ThreadPoolExecutor."""
        logger.info("Using ThreadPoolExecutor for parallel generation...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
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

                expected_tasks = ['skills', 'experiences', 'projects']
                if include_objective:
                    expected_tasks.insert(0, 'objective')

                for task_name in expected_tasks:
                    if task_name not in results:
                        results[task_name] = self._get_default_value(task_name)

            if not include_objective:
                results['objective'] = None

            return results

    def _generate_content_sequential(self, include_objective: bool = True) -> Dict:
        """EXISTING: Fallback sequential content generation."""
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

    def _safe_write_objective(self) -> Optional[str]:
        """EXISTING: Thread-safe wrapper for write_objective."""
        try:
            logger.debug("Starting objective generation...")
            result = self.write_objective()
            logger.debug(f"Objective generation result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in parallel objective generation: {e}")
            return None

    def _safe_extract_matched_skills(self) -> List:
        """EXISTING: Thread-safe wrapper for extract_matched_skills."""
        try:
            logger.debug("Starting skills extraction...")
            result = self.extract_matched_skills()
            logger.debug(f"Skills extraction result: {len(result) if result else 0} categories")
            return result
        except Exception as e:
            logger.error(f"Error in parallel skills extraction: {e}")
            return self.skills or []

    def _safe_rewrite_experiences(self) -> List:
        """EXISTING: Thread-safe wrapper for rewrite_unedited_experiences."""
        try:
            logger.debug("Starting experience rewriting...")
            result = self.rewrite_unedited_experiences()
            logger.debug(f"Experience rewriting result: {len(result) if result else 0} experiences")
            return result
        except Exception as e:
            logger.error(f"Error in parallel experience rewriting: {e}")
            return self.experiences or []

    def _safe_rewrite_projects(self) -> List:
        """EXISTING: Thread-safe wrapper for rewrite_unedited_projects."""
        try:
            logger.debug("Starting project rewriting...")
            result = self.rewrite_unedited_projects()
            logger.debug(f"Project rewriting result: {len(result) if result else 0} projects")
            return result
        except Exception as e:
            logger.error(f"Error in parallel project rewriting: {e}")
            return self.projects or []

    def _get_default_value(self, task_name: str):
        """EXISTING: Get default value for a task that failed."""
        defaults = {
            'objective': None,
            'skills': self.skills or [],
            'experiences': self.experiences or [],
            'projects': self.projects or []
        }
        return defaults.get(task_name)

    def download_and_parse_job_post(self, url=None):
        """EXISTING: Download and parse the job post from the provided URL."""
        if url:
            self.url = url
        self._download_url()
        self._extract_html_data()
        self.job_post = JobPost(self.job_post_raw, self.user)
        self.parsed_job = self.job_post.parse_job_post(verbose=True)

    def _extract_html_data(self):
        """EXISTING: Extract text content from HTML, removing all HTML tags."""
        try:
            soup = BeautifulSoup(self.job_post_html_data, "html.parser")
            self.job_post_raw = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.error(f"Failed to extract HTML data: {e}")
            raise

    def _download_url(self, url=None):
        """EXISTING: Download the content of the URL and return it as a string."""
        if url:
            self.url = url

        max_retries = config.get("settings.max_retries", 3)
        backoff_factor = config.get("settings.backoff_factor", 2)
        use_proxy = False

        for attempt in range(max_retries):
            response = None
            try:
                proxies = None
                if use_proxy:
                    proxy = FreeProxy(rand=True).get()
                    proxies = {"http": proxy, "https": proxy}

                response = requests.get(
                    self.url, headers=config.get_enhanced_headers(), proxies=proxies
                )
                response.raise_for_status()
                if response and response.status_code != 200:
                    raise RequestException
                self.job_post_html_data = response.text
                return True

            except requests.RequestException as e:
                if response and (response.status_code == 429 or response.status_code == 999):
                    logger.warning(
                        f"Rate limit exceeded. Retrying in {backoff_factor * 2 ** attempt} seconds..."
                    )
                    time.sleep(backoff_factor * 2**attempt)
                    use_proxy = True
                else:
                    logger.error(f"Failed to download URL {self.url}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(backoff_factor * 2**attempt)

        logger.error(f"Exceeded maximum retries for URL {self.url}")
        return False

    def write_objective(self, **chain_kwargs) -> str:
        """EXISTING: Write an objective for the resume."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSummarizerOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate(messages=Prompts.lookup["OBJECTIVE_WRITER"])
            llm = create_llm(self.user, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeSummarizerOutput)

            chain_inputs = self._get_formatted_chain_inputs(chain=chain)
            logger.debug(f"Objective chain inputs: {list(chain_inputs.keys())}")

            result = chain.invoke(chain_inputs)
            if result:
                if hasattr(result, 'final_answer'):
                    objective = result.final_answer
                elif isinstance(result, dict):
                    objective = result.get('final_answer')
                else:
                    objective = result

                logger.debug(f"Objective result: {objective}")
                return objective

            logger.warning("Objective generation returned None")
            return None

        except Exception as e:
            logger.error(f"Error in write_objective: {e}")
            return None

    def extract_matched_skills(self, **chain_kwargs) -> list:
        """EXISTING: Extract matched skills from the resume and job post."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSkillsMatcherOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            chain = ChatPromptTemplate(messages=Prompts.lookup["SKILLS_MATCHER"])
            llm = create_llm(self.user, **self.llm_kwargs)
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
        """EXISTING: Rewrite unedited experiences in the resume."""
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
        """EXISTING: Rewrite unedited projects in the resume."""
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
        """EXISTING: Rewrite a section of the resume."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSectionHighlighterOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            logger.debug(f"Starting rewrite_section for: {section.get('title') or section.get('name', 'Unknown')}")

            prompt = ChatPromptTemplate(messages=Prompts.lookup["SECTION_HIGHLIGHTER"])
            llm = create_llm(self.user, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeSectionHighlighterOutput)

            chain_inputs = self._get_formatted_chain_inputs(chain=chain, section=section)
            logger.debug(f"Chain inputs keys: {list(chain_inputs.keys())}")

            logger.debug("Invoking LLM chain...")
            section_revised = chain.invoke(chain_inputs)

            if section_revised:
                if hasattr(section_revised, 'final_answer'):
                    highlights = section_revised.final_answer or []
                    if highlights:
                        sorted_highlights = sorted(highlights, key=lambda d: d.relevance * -1)
                        section_type = self._determine_section_type(section)
                        limit = 4 if section_type == 'experience' else 2 if section_type == 'project' else len(sorted_highlights)
                        limited_highlights = sorted_highlights[:limit]
                        result = [s.highlight for s in limited_highlights]
                        return result

                elif isinstance(section_revised, dict):
                    highlights = section_revised.get("final_answer", [])
                    if highlights:
                        sorted_highlights = sorted(highlights, key=lambda d: d.get("relevance", 0) * -1)
                        section_type = self._determine_section_type(section)
                        limit = 4 if section_type == 'experience' else 2 if section_type == 'project' else len(sorted_highlights)
                        limited_highlights = sorted_highlights[:limit]
                        result = [s.get("highlight", "") for s in limited_highlights]
                        return result

            logger.warning("No valid highlights generated by LLM, returning original")
            return section.get("highlights", [])

        except Exception as e:
            logger.error(f"Error in rewrite_section: {e}")
            import traceback
            logger.error(f"Rewrite section traceback: {traceback.format_exc()}")
            return section.get("highlights", [])

    def _determine_section_type(self, section) -> str:
        """EXISTING: Determine if section is experience or project based on its structure."""
        if 'titles' in section or 'title' in section or 'company' in section:
            return 'experience'
        elif 'name' in section:
            return 'project'
        else:
            return 'unknown'

    def _get_formatted_chain_inputs(self, chain, section=None):
        """EXISTING: Get formatted inputs for chain with proper skills formatting"""
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

            if key == "skills":
                logger.debug(f"After chain_formatter, skills input type: {type(output_dict[key])}")

        return output_dict

    def _get_degrees(self, resume: dict):
        """EXISTING: Extract degrees from the resume."""
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
        """EXISTING: Combine two lists of skills without duplicating lowercase entries."""
        l1_lower = {i.lower() for i in l1}
        for i in l2:
            if i.lower() not in l1_lower:
                l1.append(i)

    def _combine_skill_lists(self, l1: list[dict], l2: list[dict]):
        """EXISTING: Combine two lists of skill categories without duplicating lowercase entries."""
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
        """EXISTING: Converts a dictionary to a YAML-formatted string."""
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
        """EXISTING: Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)