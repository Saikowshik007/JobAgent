import os
import logging
import time
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from drivers.browser_manager import BrowserDriverManager

# Configure logging
logger = logging.getLogger(__name__)

class LinkedInIntegration:
    """LinkedIn integration that uses the shared browser manager with cookie management."""

    def __init__(self, email=None, password=None, headless=True):
        """
        Initialize the LinkedIn integration with credentials.

        Args:
            email: LinkedIn account email (defaults to env variable)
            password: LinkedIn account password (defaults to env variable)
            headless: Whether to run browser in headless mode
        """
        self.email = email or os.environ.get('LINKEDIN_EMAIL', '')
        self.password = password or os.environ.get('LINKEDIN_PASSWORD', '')
        self.headless = headless
        self.driver = None
        self.logged_in = False
        self.search_delay = (2, 5)
        self.action_delay = (1, 3)
        self.base_url = "https://www.linkedin.com"
        self.service_name = "linkedin"
        self.wait = None  # Initialize WebDriverWait attribute

    def random_delay(self, delay_range=None):
        """Wait for a random period within the specified range."""
        if delay_range is None:
            delay_range = self.action_delay
        time.sleep(random.uniform(delay_range[0], delay_range[1]))

    def initialize_driver(self):
        """Initialize or get the shared WebDriver."""
        if self.driver is None:
            try:
                # Get shared browser driver
                self.driver = BrowserDriverManager.get_driver(headless=self.headless)
                # Initialize WebDriverWait after driver is set
                self.wait = WebDriverWait(self.driver, 10)
                logger.info("Successfully initialized LinkedIn driver")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize LinkedIn driver: {e}")
                return False
        return True

    def is_logged_in(self):
        """Check if currently logged in to LinkedIn."""
        if not self.initialize_driver():
            return False

        try:
            # Navigate to LinkedIn homepage if not already on a LinkedIn page
            current_url = self.driver.current_url
            if not current_url.startswith(self.base_url):
                self.driver.get(self.base_url)
                time.sleep(2)  # Brief delay to load the page

            # Try to find an element that's only visible when logged in
            self.driver.find_element(By.XPATH, "//div[contains(@class, 'global-nav__me')]")
            self.logged_in = True
            logger.info("User is logged in to LinkedIn")
            return True
        except:
            self.logged_in = False
            logger.info("User is not logged in to LinkedIn")
            return False

    def login(self):
        """Login to LinkedIn account using credentials or cookies."""
        if self.logged_in or self.is_logged_in():
            return True

        if not self.initialize_driver():
            return False

        # First try with cookies if available
        if BrowserDriverManager.cookies_exist(self.service_name):
            logger.info("Attempting to login to LinkedIn using saved cookies")
            if BrowserDriverManager.load_cookies(self.service_name, self.base_url):
                # Navigate to LinkedIn homepage to verify login
                self.driver.get(self.base_url)
                time.sleep(2)

                if self.is_logged_in():
                    logger.info("Successfully logged in to LinkedIn using cookies")
                    return True
                else:
                    logger.info("Cookie login failed for LinkedIn, attempting with credentials")
                    # Delete invalid cookies
                    BrowserDriverManager.delete_cookies(self.service_name)

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Navigate to LinkedIn login page
                self.driver.get(f"{self.base_url}/login")
                self.random_delay()

                # Wait for login page to load and enter credentials
                wait = WebDriverWait(self.driver, 10)

                # Enter email
                email_field = wait.until(EC.presence_of_element_located((By.ID, "username")))
                email_field.clear()
                email_field.send_keys(self.email)

                # Enter password
                password_field = wait.until(EC.presence_of_element_located((By.ID, "password")))
                password_field.clear()
                password_field.send_keys(self.password)
                self.random_delay()

                # Click login button
                login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
                login_button.click()

                # Wait for successful login - try multiple indicators
                success = False
                try:
                    # Look for the global nav element which indicates we're logged in
                    wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'global-nav__me')]")))
                    success = True
                except TimeoutException:
                    try:
                        # Alternative: Look for the feed element
                        wait.until(EC.presence_of_element_located((By.ID, "voyager-feed")))
                        success = True
                    except TimeoutException:
                        # Another alternative: Check if we're redirected to the feed URL
                        if "/feed/" in self.driver.current_url:
                            success = True

                if success:
                    # Save cookies for future use
                    BrowserDriverManager.save_cookies(self.service_name)
                    self.logged_in = True
                    logger.info("Successfully logged in to LinkedIn with credentials")
                    return True
                else:
                    # Check if there's a CAPTCHA or other verification
                    try:
                        # Look for common elements in verification screens
                        if any([
                            self.driver.find_elements(By.XPATH, "//input[@id='input__phone_verification']"),
                            self.driver.find_elements(By.XPATH, "//iframe[contains(@src, 'recaptcha')]"),
                            self.driver.find_elements(By.XPATH, "//div[contains(text(), 'Verify')]")
                        ]):
                            logger.warning("LinkedIn login requires verification (CAPTCHA/phone). Manual intervention needed.")
                            # Ask for manual intervention in the console
                            print("\n" + "="*60)
                            print("LinkedIn login requires manual verification.")
                            print("Please complete the verification in the browser window.")
                            print("The script will wait for 45 seconds.")
                            print("="*60 + "\n")

                            # Wait for the user to complete verification
                            time.sleep(45)

                            # Check if login successful after verification
                            if self.is_logged_in():
                                # Save cookies after successful manual verification
                                BrowserDriverManager.save_cookies(self.service_name)
                                return True
                            else:
                                logger.error("LinkedIn login failed even after verification wait time")
                                if attempt < max_retries - 1:
                                    logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
                                    continue
                                return False
                    except Exception as e:
                        logger.error(f"Error during LinkedIn verification check: {e}")

                    logger.error("LinkedIn login failed - could not detect login success")
                    if attempt < max_retries - 1:
                        logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
                        continue
                    return False

            except Exception as e:
                logger.error(f"Login to LinkedIn failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
                    continue
                return False

        return False

    def create_job_searcher(self,dbm):
        """Create and return a JobSearcher instance using this integration."""
        # Ensure we're logged in
        if not self.logged_in and not self.login():
            logger.error("Cannot create job searcher - not logged in to LinkedIn")
            return None

        try:
            from drivers.job_searcher import JobSearcher
            # Initialize the JobSearcher with self as the driver
            job_searcher = JobSearcher(self,dbm)
            return job_searcher
        except Exception as e:
            logger.error(f"Error creating JobSearcher: {e}")
            return None