import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import logging

# Configure logging
logger = logging.getLogger(__name__)

class RemoteWebDriverManager:
    """Manager for Selenium remote webdriver in Docker environment."""
    
    _driver_instance = None
    
    @classmethod
    def get_remote_driver(cls, headless=True, force_new=False):
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

            # Initialize the remote driver
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
                chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

                # Get the remote URL from environment variable
                remote_url = os.environ.get('SELENIUM_REMOTE_URL', 'http://selenium-chrome:4444/wd/hub')
                
                # Initialize Remote WebDriver
                driver = webdriver.Remote(
                    command_executor=remote_url,
                    options=chrome_options
                )

                cls._driver_instance = driver
                logger.info(f"Created new Remote WebDriver instance at {remote_url}")

            except Exception as e:
                logger.error(f"Failed to initialize Remote WebDriver: {e}")
                raise

        return cls._driver_instance

    @classmethod
    def release_driver(cls):
        """Release the WebDriver instance."""
        if cls._driver_instance:
            try:
                cls._driver_instance.quit()
                logger.info("Remote WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing Remote WebDriver: {e}")
            finally:
                cls._driver_instance = None

    @classmethod
    def close_all_drivers(cls):
        """Close all browser drivers."""
        cls.release_driver()

# Monkey patch the BrowserDriverManager to use the remote driver in docker environment
def patch_browser_manager():
    """Patch the BrowserDriverManager to use remote driver in Docker."""
    from browser_manager import BrowserDriverManager
    
    # Save original method
    original_get_driver = BrowserDriverManager.get_driver
    
    # Replace with remote driver in Docker environment
    def get_driver_patched(cls, headless=True, force_new=False):
        """Patched method to use remote driver in Docker."""
        # Check if we're in Docker environment
        if os.environ.get('SELENIUM_REMOTE_URL'):
            return RemoteWebDriverManager.get_remote_driver(headless, force_new)
        else:
            # Use original method if not in Docker
            return original_get_driver(cls, headless, force_new)
    
    # Apply the patch
    BrowserDriverManager.get_driver = classmethod(get_driver_patched)
    
    # Also patch release and close methods
    BrowserDriverManager.release_driver = RemoteWebDriverManager.release_driver
    BrowserDriverManager.close_all_drivers = RemoteWebDriverManager.close_all_drivers
    
    logger.info("Patched BrowserDriverManager to use remote driver in Docker environment")

# Apply the patch when this module is imported
patch_browser_manager()
