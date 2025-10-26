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
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

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
        ENHANCED: Perform global ATS optimization with real improvements.
        """
        try:
            logger.info("Performing global ATS optimization...")

            job_analysis = resume_context['job_analysis']
            resume_data = resume_context['resume_so_far']

            # Calculate current ATS score
            ats_score = self._calculate_ats_score(resume_data, job_analysis)
            logger.info(f"Current ATS score: {ats_score}")

            # Always try to optimize unless score is perfect
            if ats_score >= 95:
                logger.info("ATS score already optimal (95+), skipping optimization")
                return {**resume_data, 'ats_score': ats_score}

            # Perform real optimization
            optimized_data = self._optimize_keyword_distribution(resume_data, job_analysis)

            # Recalculate score
            final_score = self._calculate_ats_score(optimized_data, job_analysis)
            improvement = final_score - ats_score

            if improvement > 0:
                logger.info(f"ATS optimization successful: {ats_score} → {final_score} (+{improvement:.1f} points)")
            else:
                logger.info(f"ATS score maintained at {final_score} (no optimization opportunities)")

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
        """
        REAL IMPLEMENTATION: Optimize keyword distribution across resume sections.
        """
        try:
            logger.info("Performing actual keyword optimization...")

            optimized_data = resume_data.copy()
            primary_keywords = job_analysis.get('primary_keywords', [])
            technical_skills = job_analysis.get('technical_skills', [])

            if not primary_keywords and not technical_skills:
                logger.info("No keywords to optimize - returning original data")
                return resume_data

            # Track optimization changes
            changes_made = []

            # 1. OBJECTIVE OPTIMIZATION
            if optimized_data.get('objective'):
                original_objective = optimized_data['objective']
                optimized_objective = self._optimize_objective_keywords(
                    original_objective, primary_keywords[:2]  # Top 2 keywords for objective
                )
                if optimized_objective != original_objective:
                    optimized_data['objective'] = optimized_objective
                    changes_made.append("Enhanced objective with ATS keywords")

            # 2. EXPERIENCE HIGHLIGHTS OPTIMIZATION
            if optimized_data.get('experiences'):
                for i, exp in enumerate(optimized_data['experiences']):
                    if 'highlights' in exp:
                        original_highlights = exp['highlights'][:]
                        optimized_highlights = self._optimize_highlights_keywords(
                            exp['highlights'], primary_keywords, technical_skills
                        )
                        if optimized_highlights != original_highlights:
                            exp['highlights'] = optimized_highlights
                            changes_made.append(f"Enhanced experience {i+1} highlights")

            # 3. PROJECT HIGHLIGHTS OPTIMIZATION
            if optimized_data.get('projects'):
                for i, proj in enumerate(optimized_data['projects']):
                    if 'highlights' in proj:
                        original_highlights = proj['highlights'][:]
                        optimized_highlights = self._optimize_highlights_keywords(
                            proj['highlights'], technical_skills[:3], primary_keywords[:2]
                        )
                        if optimized_highlights != original_highlights:
                            proj['highlights'] = optimized_highlights
                            changes_made.append(f"Enhanced project {i+1} highlights")

            # 4. SKILLS SECTION OPTIMIZATION
            if optimized_data.get('skills'):
                original_skills = optimized_data['skills']
                optimized_skills = self._optimize_skills_section(
                    optimized_data['skills'], technical_skills, primary_keywords
                )
                if optimized_skills != original_skills:
                    optimized_data['skills'] = optimized_skills
                    changes_made.append("Enhanced skills section with missing ATS keywords")

            # Log optimization results
            if changes_made:
                logger.info(f"Keyword optimization completed: {len(changes_made)} improvements made")
                for change in changes_made:
                    logger.info(f"  - {change}")
            else:
                logger.info("No keyword optimization opportunities found")

            return optimized_data

        except Exception as e:
            logger.error(f"Error in keyword optimization: {e}")
            return resume_data

    def _optimize_objective_keywords(self, objective: str, priority_keywords: List[str]) -> str:
        """Add missing priority keywords to objective if space allows."""
        if not priority_keywords or not objective:
            return objective

        words = objective.split()
        objective_lower = objective.lower()

        # Check which keywords are missing
        missing_keywords = [kw for kw in priority_keywords if kw.lower() not in objective_lower]

        if not missing_keywords:
            return objective  # All keywords already present

        # Try to integrate one missing keyword naturally
        if len(words) < 23 and missing_keywords:  # Leave room for keyword
            keyword = missing_keywords[0]

            # Simple integration - add to technical context if possible
            if any(tech_word in objective_lower for tech_word in ['engineer', 'developer', 'architect', 'specialist']):
                # Try to integrate naturally
                if 'engineer' in objective_lower:
                    enhanced = objective.replace('Engineer', f'{keyword} Engineer', 1)
                elif 'developer' in objective_lower:
                    enhanced = objective.replace('Developer', f'{keyword} Developer', 1)
                else:
                    # Add at end if space allows
                    enhanced = f"{objective} Specialized in {keyword}."

                # Check word limit
                if len(enhanced.split()) <= 25:
                    return enhanced

        return objective

    def _optimize_highlights_keywords(self, highlights: List[str], primary_keywords: List[str], secondary_keywords: List[str]) -> List[str]:
        """Optimize highlights by naturally integrating missing keywords."""
        if not highlights or (not primary_keywords and not secondary_keywords):
            return highlights

        optimized = highlights[:]
        all_keywords = primary_keywords + secondary_keywords

        # Get current keyword coverage
        highlights_text = ' '.join(highlights).lower()
        missing_keywords = [kw for kw in all_keywords if kw.lower() not in highlights_text]

        if not missing_keywords:
            return highlights  # All keywords covered

        # Try to integrate missing keywords into existing highlights
        for i, highlight in enumerate(optimized):
            if not missing_keywords:
                break

            words = highlight.split()
            if len(words) >= 20:  # Already at limit
                continue

            # Try to add one missing keyword
            keyword = missing_keywords[0]

            # Simple integration strategies
            if len(words) < 18:  # Room for keyword
                # Strategy 1: Add technology context
                if any(tech in highlight.lower() for tech in ['developed', 'built', 'implemented', 'created']):
                    enhanced = highlight.replace('.', f' using {keyword}.')
                    if len(enhanced.split()) <= 20:
                        optimized[i] = enhanced
                        missing_keywords.remove(keyword)
                        continue

                # Strategy 2: Add to metrics context
                if any(metric in highlight for metric in ['%', 'users', 'performance', 'efficiency']):
                    enhanced = highlight.replace('.', f' via {keyword} optimization.')
                    if len(enhanced.split()) <= 20:
                        optimized[i] = enhanced
                        missing_keywords.remove(keyword)
                        continue

        return optimized

    def _optimize_skills_section(self, skills: List[Dict], technical_requirements: List[str], primary_keywords: List[str]) -> List[Dict]:
        """Add missing critical technical skills to skills section."""
        if not skills or not technical_requirements:
            return skills

        optimized_skills = [skill.copy() for skill in skills]

        # Get current skills coverage
        current_skills_text = str(skills).lower()
        missing_tech_skills = [skill for skill in technical_requirements
                               if skill.lower() not in current_skills_text]

        if not missing_tech_skills:
            return skills  # All critical skills covered

        # Find technical skills category to add missing skills
        for skill_category in optimized_skills:
            if skill_category.get('category', '').lower() == 'technical':
                if 'subcategories' in skill_category:
                    # Add to first subcategory (usually most relevant)
                    if skill_category['subcategories']:
                        first_subcat = skill_category['subcategories'][0]
                        current_count = len(first_subcat.get('skills', []))

                        # Add critical missing skills (up to 2)
                        skills_to_add = missing_tech_skills[:min(2, 6 - current_count)]
                        if skills_to_add:
                            first_subcat['skills'].extend(skills_to_add)
                            logger.info(f"Added missing ATS skills to {first_subcat.get('name', 'technical')}: {skills_to_add}")

                elif 'skills' in skill_category:
                    # Direct skills list
                    current_count = len(skill_category['skills'])
                    skills_to_add = missing_tech_skills[:min(2, 8 - current_count)]
                    if skills_to_add:
                        skill_category['skills'].extend(skills_to_add)
                        logger.info(f"Added missing ATS skills: {skills_to_add}")
                break

        return optimized_skills

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
        """
        Download and parse the job post from the provided URL.
        Uses hybrid approach: static HTML first, Playwright fallback for JS-heavy sites.
        """
        if url:
            self.url = url

        # Check if we should use Playwright immediately (for known JS-heavy sites)
        url_lower = self.url.lower()
        force_playwright = any(site in url_lower for site in [
            'linkedin.com', 'myworkdayjobs.com', 'workday',
            'greenhouse.io', 'lever.co', 'ashbyhq.com'
        ])

        success = False

        if force_playwright:
            logger.info(f"Using Playwright directly for known JS-heavy site: {self.url}")
            success = self._download_url_with_playwright()
        else:
            # Try static HTML first
            logger.info(f"Attempting static HTML download for: {self.url}")
            success = self._download_url()

        if not success:
            logger.error(f"Failed to download job post from {self.url}")
            raise Exception(f"Failed to download job post from {self.url}")

        # Extract text from HTML
        self._extract_html_data()

        # Check if we should retry with Playwright (content too short)
        if not force_playwright and self._should_use_playwright():
            logger.info("Retrying with Playwright due to insufficient content")
            if self._download_url_with_playwright():
                self._extract_html_data()

        # Parse the job post with LLM
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

    def _download_url_with_playwright(self, url=None, wait_time: int = 3000) -> bool:
        """
        Download content using Playwright for JavaScript-rendered pages.

        Args:
            url: URL to download (uses self.url if not provided)
            wait_time: Time in milliseconds to wait for page load (default: 3000ms)

        Returns:
            bool: True if successful, False otherwise
        """
        if url:
            self.url = url

        logger.info(f"Attempting to download URL with Playwright (JS-enabled): {self.url}")

        try:
            with sync_playwright() as p:
                # Launch browser in headless mode
                browser = p.chromium.launch(
                    headless=config.get("selenium.headless", True),
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )

                # Create context with realistic user agent
                context = browser.new_context(
                    user_agent=config.get(
                        "request_headers.user_agent",
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    ),
                    viewport={'width': 1920, 'height': 1080}
                )

                # Open new page
                page = context.new_page()

                try:
                    # Navigate to URL with timeout
                    page.goto(self.url, wait_until='networkidle', timeout=30000)

                    # Wait for dynamic content to load
                    page.wait_for_timeout(wait_time)

                    # Try to detect and handle common job board patterns
                    self._handle_job_board_specific_logic(page)

                    # Extract the rendered HTML content
                    self.job_post_html_data = page.content()

                    logger.info(f"Successfully downloaded {len(self.job_post_html_data)} characters with Playwright")

                    return True

                except PlaywrightTimeoutError as e:
                    logger.error(f"Playwright timeout while loading {self.url}: {e}")
                    # Try to get partial content
                    try:
                        self.job_post_html_data = page.content()
                        if self.job_post_html_data and len(self.job_post_html_data) > 500:
                            logger.warning("Retrieved partial content despite timeout")
                            return True
                    except:
                        pass
                    return False

                finally:
                    # Clean up resources
                    page.close()
                    context.close()
                    browser.close()

        except Exception as e:
            logger.error(f"Playwright extraction failed for {self.url}: {e}")
            return False

    def _handle_job_board_specific_logic(self, page):
        """
        Handle site-specific logic for common job boards.

        Args:
            page: Playwright page object
        """
        url_lower = self.url.lower()

        try:
            # LinkedIn - close sign-in modal if present
            if 'linkedin.com' in url_lower:
                try:
                    page.click('button[aria-label="Dismiss"]', timeout=2000)
                except:
                    pass
                # Wait for job description to load
                try:
                    page.wait_for_selector('.jobs-description', timeout=5000)
                except:
                    logger.debug("LinkedIn job description selector not found")

            # Indeed - expand full job description
            elif 'indeed.com' in url_lower:
                try:
                    page.click('#jobDescriptionText', timeout=2000)
                except:
                    pass

            # Glassdoor - handle sign-in overlay
            elif 'glassdoor.com' in url_lower:
                try:
                    page.click('button[data-test="close-modal"]', timeout=2000)
                except:
                    pass

            # Workday - wait for iframe content
            elif 'myworkdayjobs.com' in url_lower or 'workday' in url_lower:
                try:
                    page.wait_for_selector('[data-automation-id="jobPostingDescription"]', timeout=5000)
                except:
                    logger.debug("Workday job description selector not found")

            # Greenhouse - standard wait
            elif 'greenhouse.io' in url_lower:
                try:
                    page.wait_for_selector('#content', timeout=5000)
                except:
                    pass

            # Lever - wait for posting content
            elif 'lever.co' in url_lower:
                try:
                    page.wait_for_selector('.posting', timeout=5000)
                except:
                    pass

        except Exception as e:
            logger.debug(f"Site-specific handling failed (non-critical): {e}")

    def _should_use_playwright(self) -> bool:
        """
        Determine if Playwright should be used based on URL or content quality.

        Returns:
            bool: True if Playwright should be used
        """
        url_lower = self.url.lower()

        # Known JavaScript-heavy job boards
        js_heavy_sites = [
            'linkedin.com',
            'myworkdayjobs.com',
            'workday',
            'greenhouse.io',
            'lever.co',
            'ashbyhq.com',
            'jobs.lever.co',
            'boards.greenhouse.io'
        ]

        # Check if URL matches known JS-heavy sites
        for site in js_heavy_sites:
            if site in url_lower:
                logger.info(f"Detected JS-heavy site: {site} - will use Playwright")
                return True

        # Check if static content is too short (likely incomplete)
        if self.job_post_raw and len(self.job_post_raw.strip()) < 200:
            logger.info("Static content too short - will retry with Playwright")
            return True

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
        """Get formatted inputs for chain with proper skills formatting and experience level"""
        from services.langchain_helpers import chain_formatter, get_cumulative_time_from_titles

        output_dict = {}
        raw_self_data = self.__dict__

        # Add experience level calculation
        if self.experiences and len(self.experiences) > 0:
            if 'titles' in self.experiences[0]:
                years_exp = get_cumulative_time_from_titles(self.experiences[0].get('titles', []))
            else:
                # Fallback: calculate from all experiences
                years_exp = 0
                for exp in self.experiences:
                    if 'titles' in exp:
                        years_exp += get_cumulative_time_from_titles(exp.get('titles', []))

            raw_self_data = raw_self_data.copy()  # Create a copy to avoid modifying original
            raw_self_data['experience_level'] = self.get_experience_level(years_exp)
            raw_self_data['years_experience'] = years_exp

        if section is not None:
            if 'experience_level' not in raw_self_data:
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

    def get_experience_level(self, years: float) -> str:
        """Determine experience level from years of experience."""
        if years < 2:
            return "junior (0-2 years)"
        elif years < 5:
            return "mid-level (2-5 years)"
        else:
            return "senior (5+ years)"

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