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
        """Extract matched skills from the resume and job post with dynamic subcategories."""
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

            if not extracted_skills or not hasattr(extracted_skills, 'dict'):
                logger.warning("No extracted_skills returned from LLM")
                return self.skills or []

            extracted_skills_dict = extracted_skills.dict().get("final_answer", {})
            result = []
            logger.debug(f"Extracted skills: {extracted_skills_dict}")

            # Handle technical skills (nested dictionary structure)
            technical_skills = extracted_skills_dict.get("technical_skills", {})
            if technical_skills:
                if isinstance(technical_skills, dict) and technical_skills:
                    result.append({
                        "category": "Technical",
                        "subcategories": [
                            {"name": k, "skills": v} for k, v in technical_skills.items()
                        ]
                    })
                elif isinstance(technical_skills, list) and technical_skills:
                    result.append({
                        "category": "Technical",
                        "skills": technical_skills
                    })

            # Handle non-technical skills (simple list structure)
            non_technical_skills = extracted_skills_dict.get("non_technical_skills", [])
            if non_technical_skills:
                if isinstance(non_technical_skills, list):
                    result.append({
                        "category": "Non-technical",
                        "skills": non_technical_skills
                    })
                elif isinstance(non_technical_skills, dict) and non_technical_skills:
                    result.append({
                        "category": "Non-technical",
                        "subcategories": [
                            {"name": k, "skills": v} for k, v in non_technical_skills.items()
                        ]
                    })

            # Safely combine with existing skills
            try:
                if result:
                    existing_skills = self.skills or []
                    logger.debug(f"Existing skills type: {type(existing_skills)}")
                    logger.debug(f"Existing skills: {existing_skills}")

                    # If no existing skills, just return the new results
                    if not existing_skills:
                        logger.info("No existing skills, returning new results")
                        return result

                    # Ensure existing_skills is a list of dicts with expected structure
                    if isinstance(existing_skills, list):
                        # Validate structure of existing skills
                        validated_existing = []
                        for skill_item in existing_skills:
                            if isinstance(skill_item, dict) and "category" in skill_item:
                                validated_existing.append(skill_item)
                            else:
                                logger.warning(f"Skipping invalid existing skill item: {skill_item}")

                        if validated_existing:
                            logger.info(f"Combining {len(result)} new skill categories with {len(validated_existing)} existing")
                            self._combine_skill_lists(result, validated_existing)
                        else:
                            logger.info("No valid existing skills to combine")
                    else:
                        logger.warning(f"Existing skills is not a list: {type(existing_skills)}")

                    return result
                else:
                    logger.info("No new skills extracted, returning existing skills")
                    return self.skills or []

            except Exception as combine_error:
                logger.error(f"Error combining skill lists: {combine_error}")
                logger.error(f"Combine error traceback: {traceback.format_exc()}")
                # If combining fails, just return the new results
                return result if result else (self.skills or [])

        except Exception as e:
            logger.error(f"Error in extract_matched_skills: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self.skills or []


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