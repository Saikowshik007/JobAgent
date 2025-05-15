#!/usr/bin/env python3

import os
import logging
import time
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from browser_manager import BrowserDriverManager

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
        if self.logged_in:
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
                    self.logged_in = True
                    return True
                else:
                    logger.info("Cookie login failed for LinkedIn, attempting with credentials")
                    # Delete invalid cookies
                    BrowserDriverManager.delete_cookies(self.service_name)

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

            # Wait for successful login
            try:
                wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'global-nav__me')]")))

                # Save cookies for future use
                BrowserDriverManager.save_cookies(self.service_name)

                self.logged_in = True
                logger.info("Successfully logged in to LinkedIn with credentials")
                return True
            except TimeoutException:
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
                            return False
                except Exception as e:
                    logger.error(f"Error during LinkedIn verification check: {e}")

                logger.error("LinkedIn login failed - could not detect login success")
                return False

        except Exception as e:
            logger.error(f"Login to LinkedIn failed: {e}")
            return False

    def create_job_searcher(self):
        """Create and return a JobSearcher instance using this integration."""
        if not self.logged_in and not self.login():
            logger.error("Cannot create job searcher - not logged in to LinkedIn")
            return None

        from job_searcher import JobSearcher
        return JobSearcher(self)
