import os
import logging
import time
import pickle
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class BrowserDriverManager:
    """
    Manage shared browser driver instances and cookies.
    This singleton class helps avoid multiple browser instances while managing
    cookies for different services to maintain sessions.
    """

    _driver_instance = None
    _cookies_dir = Path("../cookies")

    @classmethod
    def get_driver(cls, headless=True, force_new=False):
        """
        Get a shared WebDriver instance.

        Args:
            headless: Whether to run browser in headless mode
            force_new: Whether to force creation of a new driver instance

        Returns:
            WebDriver instance
        """
        if cls._driver_instance is None or force_new:
            # Close existing driver if forcing new
            if force_new and cls._driver_instance:
                cls.release_driver()

            # Initialize the driver
            try:
                # Setup WebDriver options
                chrome_options = Options()
                if headless:
                    chrome_options.add_argument("--headless")

                # Common options for stability
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-notifications")
                chrome_options.add_argument("--start-maximized")

                # Add options that might help avoid detection
                chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option("useAutomationExtension", False)

                # Set a more realistic user agent
                chrome_options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

                # Initialize WebDriver
                try:
                    service = Service('/usr/local/bin/chromedriver')
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                except:
                    # Fall back to automatic installation
                    from webdriver_manager.chrome import ChromeDriverManager
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=chrome_options)

                # Execute CDP commands to prevent detection
                driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    """
                })

                cls._driver_instance = driver
                logger.info("Created new WebDriver instance")

                # Create cookies directory if it doesn't exist
                cls._cookies_dir.mkdir(exist_ok=True)

            except Exception as e:
                logger.error(f"Failed to initialize WebDriver: {e}")
                raise

        return cls._driver_instance

    @classmethod
    def release_driver(cls):
        """Release the WebDriver instance."""
        if cls._driver_instance:
            try:
                cls._driver_instance.quit()
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {e}")
            finally:
                cls._driver_instance = None

    @classmethod
    def close_all_drivers(cls):
        """Close all browser drivers."""
        cls.release_driver()

    @classmethod
    def save_cookies(cls, service_name):
        """
        Save cookies for a specific service.

        Args:
            service_name: Name of the service (e.g., 'linkedin', 'simplify')

        Returns:
            bool: True if successful, False otherwise
        """
        if cls._driver_instance is None:
            logger.error("Cannot save cookies - no active WebDriver")
            return False

        try:
            cookies = cls._driver_instance.get_cookies()
            cookies_file = cls._cookies_dir / f"{service_name}_cookies.pkl"

            with open(cookies_file, 'wb') as f:
                pickle.dump(cookies, f)

            logger.info(f"Saved cookies for {service_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving cookies for {service_name}: {e}")
            return False

    @classmethod
    def load_cookies(cls, service_name, url):
        """
        Load cookies for a specific service.

        Args:
            service_name: Name of the service (e.g., 'linkedin', 'simplify')
            url: Base URL for the service, needed before adding cookies

        Returns:
            bool: True if successful, False otherwise
        """
        if cls._driver_instance is None:
            logger.error("Cannot load cookies - no active WebDriver")
            return False

        cookies_file = cls._cookies_dir / f"{service_name}_cookies.pkl"
        if not cookies_file.exists():
            logger.warning(f"No cookies file found for {service_name}")
            return False

        try:
            # First navigate to the site's domain
            current_url = cls._driver_instance.current_url
            if not current_url.startswith(url):
                cls._driver_instance.get(url)
                time.sleep(1)  # Short delay to ensure page is loaded

            # Load the cookies
            with open(cookies_file, 'rb') as f:
                cookies = pickle.load(f)

            # Add cookies to driver
            for cookie in cookies:
                try:
                    # Some cookies might cause issues, so we handle each individually
                    cls._driver_instance.add_cookie(cookie)
                except Exception as e:
                    logger.debug(f"Skipping cookie: {e}")

            logger.info(f"Loaded cookies for {service_name}")

            # Refresh to apply cookies
            cls._driver_instance.refresh()
            return True
        except Exception as e:
            logger.error(f"Error loading cookies for {service_name}: {e}")
            return False

    @classmethod
    def cookies_exist(cls, service_name):
        """
        Check if cookies exist for a service.

        Args:
            service_name: Name of the service

        Returns:
            bool: True if cookies exist, False otherwise
        """
        cookies_file = cls._cookies_dir / f"{service_name}_cookies.pkl"
        return cookies_file.exists()

    @classmethod
    def delete_cookies(cls, service_name=None):
        """
        Delete cookies for a specific service or all services.

        Args:
            service_name: Name of the service or None to delete all

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if service_name:
                cookies_file = cls._cookies_dir / f"{service_name}_cookies.pkl"
                if cookies_file.exists():
                    cookies_file.unlink()
                    logger.info(f"Deleted cookies for {service_name}")
            else:
                # Delete all cookie files
                for cookies_file in cls._cookies_dir.glob("*_cookies.pkl"):
                    cookies_file.unlink()
                logger.info("Deleted all cookie files")

            return True
        except Exception as e:
            logger.error(f"Error deleting cookies: {e}")
            return False
