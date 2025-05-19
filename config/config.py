"""
Configuration provider for the JobTrak application.
Provides a stateless interface to access configuration values from anywhere in the project.
"""

import os
import yaml
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
import logging
import os
from datetime import datetime

import config


class ConfigProvider:
    """
    A stateless configuration provider that loads and caches the configuration.
    Any file in the project can import this class and use it to access configuration values.
    """

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_config() -> Dict[str, Any]:
        """
        Load the configuration file and cache it.
        Uses lru_cache to ensure the file is only read once.

        Returns:
            Dict[str, Any]: The configuration dictionary
        """
        # Path to config folder in the current project
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config', 'config.yaml')

        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    @classmethod
    def get(cls, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation path.

        Args:
            path: Dot-separated path to the configuration value (e.g., 'api.port')
            default: Default value to return if the path is not found

        Returns:
            The configuration value at the specified path, or the default value
        """
        config = cls._load_config()
        keys = path.split('.')

        # Navigate through the nested dictionaries
        current = config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]

        return current

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.

        Returns:
            Dict[str, Any]: The complete configuration
        """
        return cls._load_config()

    @classmethod
    def reload(cls) -> None:
        """
        Force reload the configuration from the file.
        Clears the cache and loads the configuration again.
        """
        # Clear the cache
        cls._load_config.cache_clear()
        # Load the configuration again
        cls._load_config()


# Create convenience functions for better readability in code
def get(path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation path."""
    return ConfigProvider.get(path, default)


def get_all() -> Dict[str, Any]:
    """Get the entire configuration dictionary."""
    return ConfigProvider.get_all()


def reload() -> None:
    """Force reload the configuration from the file."""
    ConfigProvider.reload()

def getLogger(name=None):
    """
    Get a configured logger instance.

    Args:
        name (str, optional): The name for the logger. If None, uses the root logger.
        log_level (str, optional): The logging level to use. If None, uses the level from config.
            Valid values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

    Returns:
        logging.Logger: A configured logger instance
    """
    # Use the provided name or default to 'jobtrak'
    log_level = config.get("api.level")
    logger_name = name if name else 'jobtrak'
    logger = logging.getLogger(logger_name)

    # If the logger already has handlers, return it to avoid duplicate handlers
    if logger.handlers:
        return logger

    # Get log level from config or parameter, default to INFO
    if log_level:
        level = getattr(logging, log_level.upper())
    else:
        level = getattr(logging, get("logging.level", "INFO"))

    logger.setLevel(level)

    # Create log directory if it doesn't exist
    log_dir = get("paths.logs", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler
    log_file = os.path.join(log_dir, f"{logger_name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a console handler with a higher log level (to reduce console noise)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(max(level, logging.INFO))  # At least INFO for console

    # Create a formatter and set it for both handlers
    log_format = get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent logs from being passed to the root logger
    logger.propagate = False

    return logger