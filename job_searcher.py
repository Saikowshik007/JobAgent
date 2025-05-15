import logging
from typing import Dict, List, Optional, Any
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementNotInteractableException
import time
import uuid
import hashlib
from datetime import datetime

from selenium.webdriver.support.wait import WebDriverWait
from data_models import Job, JobStatus

logger = logging.getLogger(__name__)

class JobSearcher:
    """Class for searching jobs on LinkedIn with integrated database and caching."""

    def __init__(self, linkedin_driver, db_manager=None):
        """Initialize the JobSearcher.

        Args:
            linkedin_driver: LinkedInDriver instance
            db_manager: Optional database manager for job tracking
        """
        self.driver = linkedin_driver
        # Use the driver instance's WebDriverWait if available, otherwise create a new one
        if hasattr(linkedin_driver, 'wait') and linkedin_driver.wait is not None:
            self.wait = linkedin_driver.wait
        else:
            self.wait = WebDriverWait(linkedin_driver.driver, 10)

        # Database manager for job tracking
        self.db_manager = db_manager

        # Cache for recently seen jobs to prevent duplicates in a single session
        self._job_url_cache = set()

        self.jobs_viewed = 0

        # Define mapping of experience level names to IDs
        self.experience_level_mapping = {
            "Internship": "1",
            "Entry level": "2",
            "Associate": "3",
            "Mid-Senior level": "4",
            "Director": "5",
            "Executive": "6"
        }

        # Define mapping of job type names to IDs
        self.job_type_mapping = {
            "Full-time": "F",
            "Part-time": "P",
            "Contract": "C",
            "Temporary": "T",
            "Volunteer": "V",
            "Internship": "I",
            "Other": "O"
        }

        # Define mapping of date posted names to IDs
        self.date_posted_mapping = {
            "Any time": "",
            "Past month": "r2592000",
            "Past week": "r604800",
            "Past Week": "r604800",  # Add capitalized version
            "Past 24 hours": "r86400",
            "Past 24 Hours": "r86400"  # Add capitalized version
        }

        # Define mapping of workplace type to IDs
        self.workplace_type_mapping = {
            "On-site": "1",
            "Remote": "2",
            "Hybrid": "3"
        }

    def search_jobs(
            self,
            keywords: str,
            location: str,
            filters: Dict[str, Any] = None,
            max_listings: int = 50,
            scroll_pages: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for jobs based on keywords and location.

        Args:
            keywords: Job title or keywords
            location: Job location
            filters: Additional filters like experience level, job type, etc.
            max_listings: Maximum number of job listings to return
            scroll_pages: Number of pages to scroll through

        Returns:
            List of job details dictionaries
        """
        # Check if we can get results from database cache
        if self.db_manager:
            cached_results = self.db_manager.get_cached_search_results(keywords, location, filters or {})
            if cached_results:
                logger.info(f"Using cached results for search: {keywords} in {location}")
                return cached_results

        # Clear session cache for new search
        self._job_url_cache.clear()

        try:
            # Navigate to jobs page
            self.driver.driver.get("https://www.linkedin.com/jobs/")
            self.driver.random_delay()

            # Enter keywords
            keyword_field = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//input[contains(@aria-label, 'Search by title')]")
            ))
            keyword_field.clear()
            keyword_field.send_keys(keywords)

            # Enter location
            location_field = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//input[contains(@aria-label, 'City')]")
            ))
            location_field.clear()
            location_field.send_keys(location)
            self.driver.random_delay()
            location_field.send_keys(Keys.RETURN)

            # Wait for search results to load
            self.driver.random_delay(self.driver.search_delay)

            # Apply additional filters if provided
            if filters:
                try:
                    self._apply_filters(filters)
                except Exception as e:
                    logger.warning(f"Error applying filters: {e}")
                    # Take debug screenshot if filter application fails
                    self._dump_page_for_debugging()

            # Collect job listings
            job_listings = self._collect_job_listings(max_listings, scroll_pages)

            # Save search results to database if available
            if self.db_manager and job_listings:
                self.db_manager.save_search_results(keywords, location, filters or {}, job_listings)
                logger.info(f"Saved search history for: {keywords} in {location}")

            return job_listings

        except Exception as e:
            logger.error(f"Job search failed: {e}")
            self._dump_page_for_debugging()
            return []

    def _apply_filters(self, filters: Dict[str, Any]) -> None:
        """Apply all search filters using the All Filters modal for consistency.

        Args:
            filters: Dictionary of filters
        """
        try:
            # Always use the All Filters button as it provides more consistent behavior
            all_filters_button = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(., 'All filters')]")
            ))
            all_filters_button.click()
            self.driver.random_delay()
            logger.info("Opened All Filters modal")

            # Apply each filter within the All Filters modal

            # Experience level filter
            if 'experience_level' in filters and filters['experience_level']:
                self._select_experience_levels_in_modal(filters['experience_level'])

            # Date posted filter
            if 'date_posted' in filters and filters['date_posted']:
                self._select_date_posted_in_modal(filters['date_posted'])

            # Job type filter
            if 'job_type' in filters and filters['job_type']:
                self._select_job_types_in_modal(filters['job_type'])

            # Remote/workplace type filter
            if 'workplace_type' in filters and filters['workplace_type']:
                self._select_workplace_types_in_modal(filters['workplace_type'])

            # Easy Apply filter
            if 'easy_apply' in filters and filters['easy_apply']:
                self._select_easy_apply_in_modal()

            # Apply all filters by clicking Show Results button
            self._click_show_results_button()

        except Exception as e:
            logger.error(f"Error applying filters via All Filters modal: {e}")
            self._dump_page_for_debugging()

    def _collect_job_listings(self, max_listings: int, scroll_pages: int) -> List[Dict[str, Any]]:
        """Collect job listings from search results.

        Args:
            max_listings: Maximum number of job listings to collect
            scroll_pages: Number of pages to scroll through

        Returns:
            List of job details dictionaries
        """
        # Try to find job list container
        try:
            job_list_container = self.wait.until(EC.presence_of_element_located(
                (By.XPATH, "//div[contains(@class, 'jobs-search-results-list')]")
            ))
        except TimeoutException:
            # If we can't find the container, just scroll the whole page
            job_list_container = self.driver.driver.find_element(By.TAG_NAME, "body")

        job_urls = set()

        # Scroll through results to load more jobs
        for _ in range(scroll_pages):
            # Scroll to bottom of container
            try:
                self.driver.driver.execute_script(
                    "arguments[0].scrollTo(0, arguments[0].scrollHeight);",
                    job_list_container
                )
            except Exception:
                # If scrolling fails, try scrolling the whole page
                self.driver.driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )

            self.driver.random_delay()

            # Collect job cards that are currently visible
            job_cards = []

            # Try multiple selectors for job cards
            selectors = [
                "//div[contains(@class, 'job-card-container')]",
                "//li[contains(@class, 'jobs-search-results__list-item')]",
                "//div[contains(@class, 'job-search-card')]"
            ]

            for selector in selectors:
                try:
                    cards = self.driver.driver.find_elements(By.XPATH, selector)
                    if cards:
                        job_cards = cards
                        break
                except Exception:
                    continue

            for card in job_cards:
                try:
                    # Try multiple selectors for job links
                    job_link = None
                    link_selectors = [
                        ".//a[contains(@class, 'job-card-list__title')]",
                        ".//a[contains(@class, 'job-card-container__link')]",
                        ".//a[contains(@class, 'base-card__full-link')]",
                        ".//a[contains(@href, '/jobs/view/')]"
                    ]

                    for link_selector in link_selectors:
                        try:
                            job_link = card.find_element(By.XPATH, link_selector)
                            if job_link:
                                break
                        except NoSuchElementException:
                            continue

                    if job_link:
                        job_url = job_link.get_attribute('href')
                        if job_url:
                            # Check if this URL is already in our tracking system
                            if self.db_manager and self.db_manager.job_exists(job_url):
                                logger.debug(f"Job already exists in database: {job_url}")
                                continue

                            # Check if we've already seen this URL in this session
                            if job_url in self._job_url_cache:
                                continue

                            # Add URL to local cache and result set
                            self._job_url_cache.add(job_url)
                            job_urls.add(job_url)

                            if len(job_urls) >= max_listings:
                                break
                except Exception as e:
                    logger.debug(f"Error processing job card: {e}")
                    continue

            if len(job_urls) >= max_listings:
                break

        # Extract details for each job
        job_details_list = []
        for job_url in list(job_urls)[:max_listings]:
            job_details = self.extract_job_details(job_url)
            job_details['status']=JobStatus.NEW
            if job_details:
                # Save job to database if available
                if self.db_manager:
                    # Convert to Job object
                    job_obj = self._convert_to_job_object(job_details)
                    self.db_manager.save_job(job_obj)

                job_details_list.append(job_details)

        logger.info(f"Found {len(job_details_list)} jobs matching search criteria")
        return job_details_list

    def _dump_page_for_debugging(self):
        """Dump page source and take screenshot for debugging."""
        try:
            # Take screenshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = f"debug_screenshot_{timestamp}.png"
            self.driver.driver.save_screenshot(screenshot_path)

            # Save page source
            page_source_path = f"debug_source_{timestamp}.html"
            with open(page_source_path, "w", encoding="utf-8") as f:
                f.write(self.driver.driver.page_source)

            logger.info(f"Saved debug info: {screenshot_path} and {page_source_path}")
        except Exception as e:
            logger.warning(f"Failed to dump debug info: {e}")

    def extract_job_details(self, job_url: str) -> Optional[Dict[str, Any]]:
        """Extract details from a job posting.

        Args:
            job_url: URL of the job posting

        Returns:
            Dictionary containing job details or None if extraction fails
        """
        # Check if job exists in database
        if self.db_manager:
            existing_job_id = self.db_manager.job_exists(job_url)
            if existing_job_id:
                job = self.db_manager.get_job(existing_job_id)
                if job:
                    logger.info(f"Job retrieved from database: {job_url}")
                    return job.to_dict()

        try:
            self.driver.driver.get(job_url)
            self.driver.random_delay(self.driver.search_delay)

            # Extract job title
            title_selectors = [
                "//h1[contains(@class, 'job-title')]",
                "//h1[contains(@class, 'topcard__title')]",
                "//h1"
            ]
            job_title = self._get_text_by_selectors(title_selectors, "Unknown Title")

            # Extract company name
            company_selectors = [
                "//a[contains(@class, 'company-name')]",
                "//span[contains(@class, 'company-name')]",
                "//a[contains(@class, 'topcard__org-name-link')]",
                "//span[contains(@class, 'topcard__org-name-link')]",
                "//span[contains(@class, 'topcard__flavor')]",
                "//div[contains(@class, 'company-name')]"
            ]
            company_name = self._get_text_by_selectors(company_selectors, "Unknown Company")

            # Extract location
            location_selectors = [
                "//span[contains(@class, 'job-location')]",
                "//span[contains(@class, 'location')]",
                "//span[contains(@class, 'topcard__flavor--bullet')]",
                "//span[contains(@class, 'workplace-type')]"
            ]
            location = self._get_text_by_selectors(location_selectors, "Unknown Location")

            # Extract job description
            description_selectors = [
                "//div[contains(@class, 'description__text')]",
                "//div[contains(@class, 'job-description')]",
                "//div[contains(@class, 'show-more-less-html__markup')]",
                "//div[contains(@id, 'job-details')]"
            ]
            job_description = self._get_text_by_selectors(description_selectors, "No description available")

            # Try to extract additional metadata
            metadata = {}

            # Try to extract workplace type (remote, hybrid, on-site)
            workplace_selectors = [
                "//span[contains(@class, 'workplace-type')]",
                "//li[contains(., 'remote')]/span",
                "//li[contains(., 'hybrid')]/span",
                "//li[contains(., 'on-site')]/span",
                "//li[contains(., 'on site')]/span"
            ]
            workplace_type = self._get_text_by_selectors(workplace_selectors)
            if workplace_type:
                metadata["workplace_type"] = workplace_type

            # Try to extract employment type
            try:
                employment_selectors = [
                    "//h3[text()='Employment type']/following-sibling::span",
                    "//h3[contains(., 'Employment type')]/following-sibling::span",
                    "//span[contains(@class, 'job-criteria__text--criteria')]"
                ]
                employment_type = self._get_text_by_selectors(employment_selectors)
                if employment_type:
                    metadata["employment_type"] = employment_type
            except NoSuchElementException:
                pass

            # Try to extract seniority level
            try:
                seniority_selectors = [
                    "//h3[text()='Seniority level']/following-sibling::span",
                    "//h3[contains(., 'Seniority level')]/following-sibling::span",
                    "//span[contains(@class, 'job-criteria__text--criteria')]"
                ]
                seniority_level = self._get_text_by_selectors(seniority_selectors)
                if seniority_level:
                    metadata["seniority_level"] = seniority_level
            except NoSuchElementException:
                pass

            # Check if it's an Easy Apply job
            try:
                easy_apply_selectors = [
                    "//button[contains(@class, 'jobs-apply-button')]",
                    "//button[contains(., 'Apply') and not(contains(., 'on company site'))]",
                    "//a[contains(@class, 'jobs-apply-button')]"
                ]
                is_easy_apply = self._element_exists(easy_apply_selectors)
            except NoSuchElementException:
                is_easy_apply = False

            # Generate a unique ID for this job based on URL
            job_id = hashlib.md5(job_url.encode()).hexdigest()

            job_details = {
                "id": job_id,
                "title": job_title,
                "company": company_name,
                "location": location,
                "description": job_description,
                "url": job_url,
                "is_easy_apply": is_easy_apply,
                "metadata": metadata,
                "date_found": datetime.now().isoformat()
            }

            logger.info(f"Extracted job details for: {job_title} at {company_name}")
            self.jobs_viewed += 1

            # Save to database if available
            if self.db_manager:
                job_obj = self._convert_to_job_object(job_details)
                self.db_manager.save_job(job_obj)

            return job_details

        except Exception as e:
            logger.error(f"Failed to extract job details: {e}")
            return None

    def _get_text_by_selectors(self, selectors: List[str], default_value: str = "") -> str:
        """Try multiple XPath selectors and return the text of the first one that works.

        Args:
            selectors: List of XPath selectors to try
            default_value: Default value to return if no selector works

        Returns:
            Text content of the first element found, or default_value
        """
        for selector in selectors:
            try:
                element = self.driver.driver.find_element(By.XPATH, selector)
                if element and element.text.strip():
                    return element.text.strip()
            except NoSuchElementException:
                continue
        return default_value

    def _element_exists(self, selectors: List[str]) -> bool:
        """Check if any of the elements specified by the selectors exists.

        Args:
            selectors: List of XPath selectors to try

        Returns:
            True if any element exists, False otherwise
        """
        for selector in selectors:
            try:
                elements = self.driver.driver.find_elements(By.XPATH, selector)
                if elements:
                    return True
            except Exception:
                continue
        return False

    def _convert_to_job_object(self, job_details: Dict[str, Any]) -> Job:
        """Convert a job details dictionary to a Job object.

        Args:
            job_details: Dictionary containing job details

        Returns:
            Job object
        """
        # Parse date if it's a string
        date_found = job_details.get("date_found")
        if isinstance(date_found, str):
            try:
                date_found = datetime.fromisoformat(date_found)
            except ValueError:
                date_found = datetime.now()
        elif not date_found:
            date_found = datetime.now()

        # Create Job object
        job = Job(
            id=job_details.get("id", str(uuid.uuid4())),
            title=job_details.get("title", "Unknown Title"),
            company=job_details.get("company", "Unknown Company"),
            location=job_details.get("location", "Unknown Location"),
            description=job_details.get("description", ""),
            url=job_details.get("url", ""),
            status=JobStatus.NEW,
            date_found=date_found,
            metadata=job_details.get("metadata", {})
        )

        return job


    def _select_experience_levels_in_modal(self, experience_levels: List[str]) -> None:
        """Select experience levels within the All Filters modal.

        Args:
            experience_levels: List of experience level strings
        """
        try:
            try:
                exp_sections = self.driver.driver.find_elements(
                    By.XPATH, "//h3[contains(., 'Experience') or contains(., 'Seniority')]"
                )
                if exp_sections and len(exp_sections) > 0:
                    self.driver.driver.execute_script("arguments[0].click();", exp_sections[0])
                    self.driver.random_delay(0.5)
                    logger.info("Expanded Experience section")
            except Exception as e:
                logger.debug(f"Note: Experience section might already be expanded: {e}")

            # Select each experience level
            for level in experience_levels:
                try:
                    # Get proper id for the level if available
                    level_id = self.experience_level_mapping.get(level, "")

                    # Try to find by id first (most reliable)
                    if level_id:
                        try:
                            checkbox = self.driver.driver.find_element(By.ID, f"advanced-filter-experience-{level_id}")
                            self.driver.driver.execute_script("arguments[0].click();", checkbox)
                            self.driver.random_delay((0.5,1))
                            logger.info(f"Selected experience level by ID: {level}")
                            continue
                        except NoSuchElementException:
                            pass

                    # Try by label text
                    xpath = f"//label[contains(., '{level}')]"
                    labels = self.driver.driver.find_elements(By.XPATH, xpath)
                    if labels and len(labels) > 0:
                        self.driver.driver.execute_script("arguments[0].click();", labels[0])
                        self.driver.random_delay((0.5,1.0))
                        logger.info(f"Selected experience level by text: {level}")
                    else:
                        logger.warning(f"Could not find experience level: {level}")
                except Exception as e:
                    logger.warning(f"Error selecting experience level {level}: {e}")
        except Exception as e:
            logger.warning(f"Error in experience level selection: {e}")

    def _select_date_posted_in_modal(self, date_posted: str) -> None:
        """Select date posted filter within the All Filters modal.

        Args:
            date_posted: Date posted option string
        """
        try:
            # Try to find and expand the Date Posted section if it's collapsed
            try:
                date_sections = self.driver.driver.find_elements(
                    By.XPATH, "//h3[contains(., 'Date posted') or contains(., 'Time')]"
                )
                if date_sections and len(date_sections) > 0:
                    self.driver.driver.execute_script("arguments[0].click();", date_sections[0])
                    self.driver.random_delay((0.5,1))
                    logger.info("Expanded Date Posted section")
            except Exception as e:
                logger.debug(f"Note: Date Posted section might already be expanded: {e}")

            # Get ID for the date option
            date_id = self.date_posted_mapping.get(date_posted, "")

            # Try to select by ID first (most reliable)
            if date_id:
                try:
                    radio_id = f"advanced-filter-timePostedRange-{date_id}"
                    radio = self.driver.driver.find_element(By.ID, radio_id)
                    self.driver.driver.execute_script("arguments[0].click();", radio)
                    self.driver.random_delay((0.5,1))
                    logger.info(f"Selected date posted by ID: {date_posted}")
                    return
                except NoSuchElementException:
                    pass

            # Try various text patterns
            text_patterns = [
                date_posted,
                date_posted.lower(),
                date_posted.upper(),
                date_posted.capitalize(),
                ' '.join(word.capitalize() for word in date_posted.split())
            ]

            for pattern in text_patterns:
                try:
                    xpath = f"//label[contains(., '{pattern}')]"
                    labels = self.driver.driver.find_elements(By.XPATH, xpath)
                    if labels and len(labels) > 0:
                        self.driver.driver.execute_script("arguments[0].click();", labels[0])
                        self.driver.random_delay(0.5)
                        logger.info(f"Selected date posted by text: {pattern}")
                        return
                except Exception:
                    continue

            logger.warning(f"Could not select date posted: {date_posted}")
        except Exception as e:
            logger.warning(f"Error in date posted selection: {e}")

    def _select_job_types_in_modal(self, job_types: List[str]) -> None:
        """Select job types within the All Filters modal.

        Args:
            job_types: List of job type strings
        """
        try:
            # Try to find and expand the Job Type section if it's collapsed
            try:
                job_type_sections = self.driver.driver.find_elements(
                    By.XPATH, "//h3[contains(., 'Job type') or contains(., 'Employment type')]"
                )
                if job_type_sections and len(job_type_sections) > 0:
                    self.driver.driver.execute_script("arguments[0].click();", job_type_sections[0])
                    self.driver.random_delay((0.5,1))
                    logger.info("Expanded Job Type section")
            except Exception as e:
                logger.debug(f"Note: Job Type section might already be expanded: {e}")

            # Select each job type
            for job_type in job_types:
                try:
                    # Get proper id for the job type if available
                    job_type_id = self.job_type_mapping.get(job_type, "")

                    # Try to find by id first (most reliable)
                    if job_type_id:
                        try:
                            checkbox = self.driver.driver.find_element(By.ID, f"advanced-filter-jobType-{job_type_id}")
                            self.driver.driver.execute_script("arguments[0].click();", checkbox)
                            self.driver.random_delay((0.5,1))
                            logger.info(f"Selected job type by ID: {job_type}")
                            continue
                        except NoSuchElementException:
                            pass

                    # Try by label text
                    xpath = f"//label[contains(., '{job_type}')]"
                    labels = self.driver.driver.find_elements(By.XPATH, xpath)
                    if labels and len(labels) > 0:
                        self.driver.driver.execute_script("arguments[0].click();", labels[0])
                        self.driver.random_delay((0.5,1))
                        logger.info(f"Selected job type by text: {job_type}")
                    else:
                        logger.warning(f"Could not find job type: {job_type}")
                except Exception as e:
                    logger.warning(f"Error selecting job type {job_type}: {e}")
        except Exception as e:
            logger.warning(f"Error in job type selection: {e}")

    def _select_workplace_types_in_modal(self, workplace_types: List[str]) -> None:
        """Select workplace types within the All Filters modal.

        Args:
            workplace_types: List of workplace type strings
        """
        try:
            # Try to find and expand the Workplace Type section if it's collapsed
            try:
                workplace_sections = self.driver.driver.find_elements(
                    By.XPATH, "//h3[contains(., 'Remote') or contains(., 'Work') or contains(., 'On-site')]"
                )
                if workplace_sections and len(workplace_sections) > 0:
                    self.driver.driver.execute_script("arguments[0].click();", workplace_sections[0])
                    self.driver.random_delay(0.5)
                    logger.info("Expanded Workplace Type section")
            except Exception as e:
                logger.debug(f"Note: Workplace Type section might already be expanded: {e}")

            # Select each workplace type
            for workplace_type in workplace_types:
                try:
                    # Get proper id for the workplace type if available
                    workplace_id = self.workplace_type_mapping.get(workplace_type, "")

                    # Try to find by id first (most reliable)
                    if workplace_id:
                        try:
                            checkbox = self.driver.driver.find_element(By.ID, f"advanced-filter-workplaceType-{workplace_id}")
                            self.driver.driver.execute_script("arguments[0].click();", checkbox)
                            self.driver.random_delay(0.5)
                            logger.info(f"Selected workplace type by ID: {workplace_type}")
                            continue
                        except NoSuchElementException:
                            pass

                    # Try by label text
                    xpath = f"//label[contains(., '{workplace_type}')]"
                    labels = self.driver.driver.find_elements(By.XPATH, xpath)
                    if labels and len(labels) > 0:
                        self.driver.driver.execute_script("arguments[0].click();", labels[0])
                        self.driver.random_delay(0.5)
                        logger.info(f"Selected workplace type by text: {workplace_type}")
                    else:
                        logger.warning(f"Could not find workplace type: {workplace_type}")
                except Exception as e:
                    logger.warning(f"Error selecting workplace type {workplace_type}: {e}")
        except Exception as e:
            logger.warning(f"Error in workplace type selection: {e}")

    def _select_easy_apply_in_modal(self) -> None:
        """Select Easy Apply option within the All Filters modal."""
        try:
            # Try to find and expand the Easy Apply section if it exists and is collapsed
            try:
                easy_apply_sections = self.driver.driver.find_elements(
                    By.XPATH, "//h3[contains(., 'Easy Apply') or contains(., 'Application')]"
                )
                if easy_apply_sections and len(easy_apply_sections) > 0:
                    self.driver.driver.execute_script("arguments[0].click();", easy_apply_sections[0])
                    self.driver.random_delay(0.5)
                    logger.info("Expanded Easy Apply section")
            except Exception as e:
                logger.debug(f"Note: Easy Apply section might already be expanded or missing: {e}")

            # Try different patterns for Easy Apply checkbox
            selectors = [
                "//label[contains(., 'Easy Apply')]",
                "//label[contains(., 'LinkedIn Easy Apply')]",
                "//span[contains(., 'Easy Apply')]/ancestor::label",
            ]

            for selector in selectors:
                try:
                    labels = self.driver.driver.find_elements(By.XPATH, selector)
                    if labels and len(labels) > 0:
                        self.driver.driver.execute_script("arguments[0].click();", labels[0])
                        self.driver.random_delay(0.5)
                        logger.info("Selected Easy Apply option")
                        return
                except Exception:
                    continue

            logger.warning("Could not find Easy Apply option")
        except Exception as e:
            logger.warning(f"Error selecting Easy Apply option: {e}")

    def _click_show_results_button(self) -> None:
        """Click the Show Results button in the All Filters modal."""
        try:
            # Try different button patterns with JavaScript click for reliability
            button_selectors = [
                "//button[contains(., 'Show') and contains(., 'results')]",
                "//button[contains(., 'Apply')]",
                "//button[contains(@class, 'artdeco-button--primary')]",
                "//footer//button[contains(@class, 'artdeco-button--primary')]"
            ]

            for selector in button_selectors:
                try:
                    buttons = self.driver.driver.find_elements(By.XPATH, selector)
                    if buttons and len(buttons) > 0:
                        self.driver.driver.execute_script("arguments[0].click();", buttons[0])
                        self.driver.random_delay()
                        logger.info(f"Clicked Show Results button with selector: {selector}")
                        return
                except Exception:
                    continue

            logger.warning("Could not find Show Results button")
        except Exception as e:
            logger.warning(f"Error clicking Show Results button: {e}")