#!/usr/bin/env python3

import os
import logging
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from drivers.browser_manager import BrowserDriverManager

# Configure logging
logger = logging.getLogger(__name__)

class SimplifyIntegration:
    """Integration with Simplify.jobs using the shared browser manager."""
    
    def __init__(self, email=None, password=None, headless=True, user_intervention=False, wait_time=30):
        """
        Initialize the Simplify integration with credentials.
        
        Args:
            email: Simplify account email (defaults to env variable)
            password: Simplify account password (defaults to env variable)
            headless: Whether to run browser in headless mode
            user_intervention: Whether to allow user intervention for CAPTCHA
            wait_time: Time to wait for manual CAPTCHA solving in seconds
        """
        self.email = email or os.environ.get('SIMPLIFY_EMAIL', '')
        self.password = password or os.environ.get('SIMPLIFY_PASSWORD', '')
        self.headless = headless
        self.driver = None
        self.logged_in = False
        self.user_intervention = user_intervention
        self.wait_time = wait_time
        self.base_url = "https://simplify.jobs"
        self.service_name = "simplify"
    
    def initialize_driver(self):
        """Initialize or get the shared WebDriver."""
        if self.driver is None:
            try:
                # Get shared browser driver
                self.driver = BrowserDriverManager.get_driver(headless=self.headless)
                logger.info("Successfully initialized Simplify driver")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Simplify driver: {e}")
                return False
        return True
    
    def is_session_valid(self):
        """Check if the current Simplify session is still valid."""
        if not self.initialize_driver():
            return False
        
        try:
            # Try accessing a page that requires login
            self.driver.get(f"{self.base_url}/dashboard")
            # Wait briefly to let the page load
            time.sleep(2)
            
            # Check if we're still on the dashboard (logged in)
            is_valid = "dashboard" in self.driver.current_url
            self.logged_in = is_valid
            return is_valid
        except Exception as e:
            logger.error(f"Error checking Simplify session validity: {str(e)}")
            self.logged_in = False
            return False
    
    def login_to_simplify(self, max_retries=3):
        """
        Log in to Simplify.jobs with provided credentials or saved cookies.
        
        Args:
            max_retries: Maximum number of login attempts.
            
        Returns:
            bool: True if login successful, False otherwise
        """
        if self.logged_in or self.is_session_valid():
            return True
            
        if not self.initialize_driver():
            return False
        
        # First try with cookies if available
        if BrowserDriverManager.cookies_exist(self.service_name):
            logger.info("Attempting to login to Simplify using saved cookies")
            if BrowserDriverManager.load_cookies(self.service_name, self.base_url):
                # Navigate to dashboard to verify login
                self.driver.get(f"{self.base_url}/dashboard")
                time.sleep(2)
                
                if "dashboard" in self.driver.current_url:
                    logger.info("Successfully logged in to Simplify using cookies")
                    self.logged_in = True
                    return True
                else:
                    logger.info("Cookie login failed for Simplify, attempting with credentials")
                    # Delete invalid cookies
                    BrowserDriverManager.delete_cookies(self.service_name)
        
        # Login with credentials
        attempt = 0
        while attempt < max_retries:
            try:
                # Navigate to login page
                logger.info("Navigating to Simplify.jobs login page")
                self.driver.get(f"{self.base_url}/auth/login")
                
                # Wait for login page to load
                wait = WebDriverWait(self.driver, 15)
                email_field = wait.until(
                    EC.presence_of_element_located((By.ID, "email"))
                )
                
                # Enter credentials
                logger.info(f"Entering Simplify email: {self.email}")
                email_field.clear()
                email_field.send_keys(self.email)
                
                password_field = self.driver.find_element(By.ID, "password")
                password_field.clear()
                password_field.send_keys(self.password)
                
                # Click sign-in button
                logger.info("Clicking Simplify sign-in button")
                signin_button = self.driver.find_element(
                    By.XPATH, "//button[@type='submit']//span[text()='Sign in']/.."
                )
                signin_button.click()
                
                try:
                    wait.until(
                        EC.url_contains("simplify.jobs/dashboard")
                    )
                    logger.info("Simplify login successful")
                    self.logged_in = True
                    
                    # Save cookies for future use
                    BrowserDriverManager.save_cookies(self.service_name)
                    
                    return True
                except TimeoutException:
                    # Check if we're still dealing with a CAPTCHA
                    try:
                        captcha_element = self.driver.find_element(
                            By.XPATH, "//*[contains(@title, 'reCAPTCHA') or contains(@title, 'captcha')]"
                        )
                        logger.warning("Still on Simplify CAPTCHA page. CAPTCHA may not have been solved correctly.")
                        
                        if self.user_intervention:
                            print("\n" + "="*60)
                            print("CAPTCHA detected for Simplify login. Please solve it manually.")
                            print(f"You have {self.wait_time} seconds to solve the CAPTCHA.")
                            print("="*60 + "\n")
                            time.sleep(self.wait_time)
                            
                            # Check if login successful after CAPTCHA
                            if "dashboard" in self.driver.current_url:
                                logger.info("Simplify login successful after CAPTCHA")
                                self.logged_in = True
                                BrowserDriverManager.save_cookies(self.service_name)
                                return True
                        else:
                            attempt += 1
                            continue
                    except NoSuchElementException:
                        # Check if there's an error message
                        try:
                            error_msg = self.driver.find_element(
                                By.XPATH, "//div[contains(@class, 'text-red')]"
                            ).text
                            logger.error(f"Simplify login failed: {error_msg}")
                        except NoSuchElementException:
                            logger.error("Simplify login failed: Unknown error")
                    
                    attempt += 1
                    if attempt < max_retries:
                        backoff_time = 2 * attempt
                        logger.info(f"Simplify login failed, retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
            
            except Exception as e:
                logger.error(f"Error during Simplify login attempt {attempt+1}: {str(e)}")
                attempt += 1
                if attempt < max_retries:
                    backoff_time = 2 * attempt
                    logger.info(f"Simplify login attempt failed, retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
        
        return False
    
    def upload_resume(self, pdf_path, timeout=30):
        """
        Upload a resume to Simplify.jobs.
        
        Args:
            pdf_path: Path to the PDF resume file
            timeout: Timeout in seconds for waiting for page elements
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.logged_in and not self.is_session_valid():
            if not self.login_to_simplify():
                logger.error("Not logged in to Simplify. Please log in before uploading.")
                return False
        
        try:
            # Check if the file exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found at path: {pdf_path}")
                return False
            
            # Navigate to documents page
            logger.info("Navigating to Simplify documents page")
            self.driver.get(f"{self.base_url}/documents")
            
            # Wait for page to load
            wait = WebDriverWait(self.driver, timeout)
            
            # Check if we're on the documents page
            time.sleep(2)  # Give page time to stabilize
            
            # Try to find and click the upload button with retry logic
            max_click_attempts = 3
            for click_attempt in range(max_click_attempts):
                try:
                    # Find the upload button
                    upload_button = wait.until(
                        EC.element_to_be_clickable((
                            By.XPATH,
                            "//button[.//span[text()='Upload a resume']]"
                        ))
                    )
                    
                    # Scroll to the button to ensure it's in view
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", upload_button)
                    time.sleep(1)  # Allow time for scrolling
                    self.driver.execute_script("arguments[0].click();", upload_button)
                    
                    # Wait to see if file input becomes available
                    time.sleep(2)
                    try:
                        file_input = wait.until(
                            EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
                        )
                        break
                    except TimeoutException:
                        if click_attempt < max_click_attempts - 1:
                            logger.warning(f"Failed to open Simplify upload modal on attempt {click_attempt+1}, retrying...")
                            time.sleep(2)  # Wait before retrying
                        else:
                            logger.error("Failed to open Simplify upload modal after multiple attempts")
                            return False
                
                except Exception as click_err:
                    logger.error(f"Error clicking Simplify upload button: {click_err}")
                    if click_attempt < max_click_attempts - 1:
                        time.sleep(2)
                    else:
                        return False
            
            # Upload the file
            logger.info(f"Uploading file to Simplify: {pdf_path}")
            file_input.send_keys(pdf_path)
            time.sleep(5)
            
            # Wait for the Upload button to appear after selecting the file
            logger.info("Waiting for Simplify Upload button to appear")
            try:
                wait_for_button = WebDriverWait(self.driver, 10)
                upload_confirm_button = wait_for_button.until(
                    EC.element_to_be_clickable((
                        By.XPATH,
                        "//button[contains(@class, 'bg-primary') and .//span[text()='Upload']]"
                    ))
                )
                
                self.driver.execute_script("arguments[0].click();", upload_confirm_button)
                time.sleep(5)
                logger.info("Resume uploaded successfully to Simplify")
                return True
            except TimeoutException:
                logger.error("Timed out waiting for Simplify upload confirmation")
                # Take a screenshot for debugging
                try:
                    screenshot_path = "simplify_upload_timeout.png"
                    self.driver.save_screenshot(screenshot_path)
                    logger.info(f"Screenshot saved to {screenshot_path}")
                except Exception as ss_err:
                    logger.error(f"Failed to save screenshot: {ss_err}")
                return False
        
        except Exception as e:
            logger.error(f"Error during resume upload to Simplify: {str(e)}")
            # Take a screenshot for debugging
            try:
                screenshot_path = "simplify_upload_error.png"
                self.driver.save_screenshot(screenshot_path)
                logger.info(f"Screenshot saved to {screenshot_path}")
            except Exception as ss_err:
                logger.error(f"Failed to save screenshot: {ss_err}")
            return False

class SimplifyIntegrationAdapter:
    """
    Adapter class for using Simplify integration as a singleton.
    This allows the system to share a single instance of SimplifyIntegration.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, force_new=False, **kwargs):
        """
        Get or create a SimplifyIntegration instance.
        
        Args:
            force_new: Whether to force creation of a new instance
            **kwargs: Arguments to pass to SimplifyIntegration constructor
            
        Returns:
            SimplifyIntegration instance
        """
        if cls._instance is None or force_new:
            cls._instance = SimplifyIntegration(**kwargs)
            logger.info("Created new SimplifyIntegration instance")
        return cls._instance
    
    @classmethod
    def upload_resume_to_simplify(cls, pdf_path, **kwargs):
        """
        Upload a resume to Simplify.jobs using the shared instance.
        
        Args:
            pdf_path: Path to the PDF resume file
            **kwargs: Additional arguments for SimplifyIntegration
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        simplify = cls.get_instance(**kwargs)
        return simplify.upload_resume(pdf_path)
