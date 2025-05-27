import yaml
from yaml import YAMLError
from services.langchain_helpers import *
from dataModels.job_post import JobPost
from config import config

logger = config.getLogger("ResumeImprover")
class ResumeImprover:
    """
    Cleaned ResumeImprover - focused on doing ALL the resume work.
    Same interface, less complexity.
    """

    def __init__(self, url, api_key, resume_location=None, llm_kwargs: dict = None):
        """Initialize ResumeImprover with the job post URL and optional resume location."""
        super().__init__()
        self.job_post_html_data = None
        self.job_post_raw = None
        self.resume = None
        self.job_post = None
        self.parsed_job = None
        self.llm_kwargs = llm_kwargs or {}
        self.api_key = api_key
        self.url = url

        # Resume data fields
        self.basic_info = None
        self.education = None
        self.experiences = None
        self.projects = None
        self.skills = None
        self.objective = None
        self.degrees = None

    def create_complete_tailored_resume(self) -> str:
        """
        NEW main method: Create complete tailored resume.
        This is what ResumeGenerator calls - does everything.
        """
        try:
            logger.info("=== Creating Complete Tailored Resume ===")

            # Step 1: Parse job posting
            logger.info("Parsing job posting...")
            self.download_and_parse_job_post()

            # Step 2: Generate all content
            logger.info("Generating objective...")
            objective = self.write_objective()
            logger.info(f"Objective generated: {objective is not None}")

            logger.info("Extracting matched skills...")
            skills = self.extract_matched_skills()
            logger.info(f"Skills extracted: {len(skills) if skills else 0}")

            logger.info("Updating experiences...")
            experiences = self.rewrite_unedited_experiences()
            logger.info(f"Experiences updated: {len(experiences) if experiences else 0}")

            logger.info("Updating projects...")
            projects = self.rewrite_unedited_projects()
            logger.info(f"Projects updated: {len(projects) if projects else 0}")

            # Step 3: Create final resume
            logger.info("Assembling final resume...")
            final_resume = {
                'editing': False,
                'basic': self.basic_info or {},
                'objective': objective,  # KEY LINE FOR DEBUGGING
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

            # Debug objective specifically
            logger.info(f"Final resume objective: {final_resume.get('objective')}")
            logger.info(f"Objective is None: {final_resume.get('objective') is None}")

            # Step 4: Convert to YAML
            yaml_content = self.dict_to_yaml_string(final_resume)

            # Final check
            if 'objective:' in yaml_content:
                logger.info("✓ Objective found in final YAML")
            else:
                logger.warning("✗ Objective NOT found in final YAML")

            logger.info("=== Resume Creation Complete ===")
            return yaml_content

        except Exception as e:
            logger.error(f"Complete resume creation failed: {e}")
            raise

    def download_and_parse_job_post(self, url=None):
        """Download and parse the job post from the provided URL."""
        if url:
            self.url = url
        self._download_url()
        self._extract_html_data()
        self.job_post = JobPost(self.job_post_raw, self.api_key)
        self.parsed_job = self.job_post.parse_job_post(verbose=False)

    def _download_url(self, url=None):
        """Download the content of the URL and return it as a string."""
        if url:
            self.url = url

        try:
            import requests
            response = requests.get(self.url)
            response.raise_for_status()
            self.job_post_html_data = response.text
            return True
        except Exception as e:
            logger.error(f"Failed to download URL {self.url}: {e}")
            raise

    def _extract_html_data(self):
        """Extract text content from HTML, removing all HTML tags."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(self.job_post_html_data, "html.parser")
            self.job_post_raw = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.error(f"Failed to extract HTML data: {e}")
            raise

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

            if result and hasattr(result, 'dict'):
                objective_dict = result.dict()
                objective = objective_dict.get('final_answer')
                logger.debug(f"Objective result: {objective}")
                return objective

            logger.warning("Objective generation returned None")
            return None

        except Exception as e:
            logger.error(f"Error in write_objective: {e}")
            return None

def extract_matched_skills(self, **chain_kwargs) -> list:
    """Extract matched skills from the resume and job post."""
    try:
        from prompts import Prompts
        from dataModels.resume import ResumeSkillsMatcherOutput
        from services.langchain_helpers import create_llm
        from langchain.prompts import ChatPromptTemplate

        chain = ChatPromptTemplate(messages=Prompts.lookup["SKILLS_MATCHER"])
        llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
        runnable = chain | llm.with_structured_output(schema=ResumeSkillsMatcherOutput)

        chain_inputs = self._get_formatted_chain_inputs(chain=runnable)
        extracted_skills = runnable.invoke(chain_inputs)

        if not extracted_skills or not hasattr(extracted_skills, 'dict'):
            return self.skills or []

        extracted_skills = extracted_skills.dict().get("final_answer", {})
        result = []

        # Process technical skills with proper YAML structure
        if "technical_skills" in extracted_skills:
            technical_skills_dict = {}

            # Parse the technical skills response to create proper subcategories
            tech_skills = extracted_skills["technical_skills"]
            if isinstance(tech_skills, dict):
                # If already structured as dict with subcategories
                technical_skills_dict = tech_skills
            elif isinstance(tech_skills, list):
                # If it's a list, we need to parse it to create subcategories
                technical_skills_dict = self._parse_skills_list_to_dict(tech_skills)

            result.append({
                "category": "Technical",
                "skills": technical_skills_dict
            })

        # Process non-technical skills as simple list
        if "non_technical_skills" in extracted_skills:
            non_tech_skills = extracted_skills["non_technical_skills"]
            if isinstance(non_tech_skills, list):
                # Filter out any category headers that might be mixed in
                clean_skills = [skill for skill in non_tech_skills if not skill.startswith('>')]
                result.append({
                    "category": "Non-technical",
                    "skills": clean_skills
                })

        # Combine with existing skills
        self._combine_skill_lists(result, self.skills or [])
        return result

    except Exception as e:
        logger.error(f"Error in extract_matched_skills: {e}")
        return self.skills or []

def _parse_skills_list_to_dict(self, skills_list: list) -> dict:
    """Parse a mixed list of skills and category headers into a proper dictionary structure."""
    result = {}
    current_category = "Other"

    for item in skills_list:
        if isinstance(item, str):
            if item.startswith('>'):
                # This is a category header
                current_category = item.replace('>', '').strip()
                if current_category not in result:
                    result[current_category] = []
            else:
                # This is a skill
                if current_category not in result:
                    result[current_category] = []
                result[current_category].append(item)

    return result

def _combine_skill_lists(self, l1: list[dict], l2: list[dict]):
    """Combine two lists of skill categories without duplicating lowercase entries."""
    l1_categories_lowercase = {s["category"].lower(): i for i, s in enumerate(l1)}

    for s in l2:
        category_lower = s["category"].lower()

        if category_lower in l1_categories_lowercase:
            existing_index = l1_categories_lowercase[category_lower]
            existing_skills = l1[existing_index]["skills"]
            new_skills = s["skills"]

            # Handle both dict and list formats
            if isinstance(existing_skills, dict) and isinstance(new_skills, dict):
                # Merge dictionaries
                for subcat, skills in new_skills.items():
                    if subcat in existing_skills:
                        self._combine_skills_in_category(existing_skills[subcat], skills)
                    else:
                        existing_skills[subcat] = skills
            elif isinstance(existing_skills, list) and isinstance(new_skills, list):
                # Merge lists
                self._combine_skills_in_category(existing_skills, new_skills)
            # If formats don't match, keep the new format
            elif isinstance(new_skills, dict):
                l1[existing_index]["skills"] = new_skills
        else:
            l1.append(s)

    def rewrite_unedited_experiences(self, **chain_kwargs) -> list:
        """Rewrite unedited experiences in the resume."""
        try:
            if not self.experiences:
                return []

            result = []
            for exp in self.experiences:
                exp = dict(exp)
                exp["highlights"] = self.rewrite_section(section=exp, **chain_kwargs)
                result.append(exp)
            return result
        except Exception as e:
            logger.error(f"Error in rewrite_unedited_experiences: {e}")
            return self.experiences or []

    def rewrite_unedited_projects(self, **chain_kwargs) -> list:
        """Rewrite unedited projects in the resume."""
        try:
            if not self.projects:
                return []

            result = []
            for proj in self.projects:
                proj = dict(proj)
                proj["highlights"] = self.rewrite_section(section=proj, **chain_kwargs)
                result.append(proj)
            return result
        except Exception as e:
            logger.error(f"Error in rewrite_unedited_projects: {e}")
            return self.projects or []

    def rewrite_section(self, section, **chain_kwargs) -> list:
        """Rewrite a section of the resume."""
        try:
            from prompts import Prompts
            from dataModels.resume import ResumeSectionHighlighterOutput
            from services.langchain_helpers import create_llm
            from langchain.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate(messages=Prompts.lookup["SECTION_HIGHLIGHTER"])
            llm = create_llm(api_key=self.api_key, **self.llm_kwargs)
            chain = prompt | llm.with_structured_output(schema=ResumeSectionHighlighterOutput)

            chain_inputs = self._get_formatted_chain_inputs(chain=chain, section=section)
            section_revised = chain.invoke(chain_inputs)

            if section_revised and hasattr(section_revised, 'dict'):
                highlights = section_revised.dict().get("final_answer", [])
                sorted_highlights = sorted(highlights, key=lambda d: d.get("relevance", 0) * -1)
                return [s["highlight"] for s in sorted_highlights]

            return section.get("highlights", [])

        except Exception as e:
            logger.error(f"Error in rewrite_section: {e}")
            return section.get("highlights", [])

    def _get_formatted_chain_inputs(self, chain, section=None):
        """Get formatted inputs for chain"""
        from services.langchain_helpers import chain_formatter

        output_dict = {}
        raw_self_data = self.__dict__
        if section is not None:
            raw_self_data = raw_self_data.copy()
            raw_self_data["section"] = section

        for key in chain.get_input_schema().schema().get("required", []):
            value = raw_self_data.get(key) or (self.parsed_job.get(key) if self.parsed_job else None)
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
            if s["category"].lower() in l1_categories_lowercase:
                self._combine_skills_in_category(
                    l1[l1_categories_lowercase[s["category"].lower()]]["skills"],
                    s["skills"],
                )
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