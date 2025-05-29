from datetime import datetime
from typing import List
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
import langchain
from langchain_community.cache import InMemoryCache
import config
import utils

# Set up LLM cache
langchain.llm_cache = InMemoryCache()
logger = config.getLogger("langchain_helper")  # Get the logger from config


def create_llm(**kwargs):
    """Create an LLM instance with specified parameters."""
    chat_model = kwargs.pop("chat_model", ChatOpenAI)
    model_name = kwargs.get("model_name", config.get("model.name"))


    # Check if api_key exists in kwargs and use it
    if "api_key" in kwargs:
        api_key = kwargs.pop("api_key")
        logger.debug("Using provided API key for LLM")
        # Ensure it's directly passed as a separate parameter
        kwargs.setdefault("model_name", config.get("model.name"))
        kwargs.setdefault("cache", False)
        try:
            return chat_model(api_key=api_key, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create LLM with provided API key: {str(e)}")
            raise
    else:
        # No API key provided, fallback to environment variable
        logger.debug("No API key provided, falling back to environment variable")
        kwargs.setdefault("model_name", config.get("model.name"))
        kwargs.setdefault("cache", False)
        try:
            return chat_model(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create LLM with environment API key: {str(e)}")
            raise


def format_list_as_string(lst: list, list_sep: str = "\n- ") -> str:
    """Format a list as a string with a specified separator."""
    if isinstance(lst, list):
        return list_sep + list_sep.join(lst)
    return str(lst)


def format_prompt_inputs_as_strings(prompt_inputs: list[str], **kwargs):
    """Convert values to string for all keys in kwargs matching list in prompt inputs."""
    logger.debug(f"Formatting prompt inputs: {', '.join(prompt_inputs)}")
    return {
        k: format_list_as_string(v) for k, v in kwargs.items() if k in prompt_inputs
    }


def parse_date(date_str: str) -> datetime:
    """Given an arbitrary string, parse it to a date."""
    logger.debug(f"Parsing date string: {date_str}")
    default_date = datetime(datetime.today().year, 1, 1)
    try:
        parsed_date = dateparser.parse(str(date_str), default=default_date)
        logger.debug(f"Successfully parsed date '{date_str}' to {parsed_date}")
        return parsed_date
    except dateparser._parser.ParserError as e:
        langchain.llm_cache.clear()
        logger.error(f"Date input `{date_str}` could not be parsed: {str(e)}")
        raise e


def datediff_years(start_date: str, end_date: str) -> float:
    """Calculate the difference between two dates in fractional years.

    Args:
        start_date (str): The start date in string format.
        end_date (str): The end date in string format. Can be "Present" to use the current date.

    Returns:
        float: The difference in years, including fractional years.
    """
    logger.debug(f"Calculating years between {start_date} and {end_date}")
    if isinstance(end_date, str) and end_date.lower() == "present":
        end_date = datetime.today().strftime("%Y-%m-%d")
        logger.debug(f"End date is 'present', using current date: {end_date}")

    try:
        start = parse_date(start_date)
        end = parse_date(end_date)
        datediff = relativedelta(end, start)
        years_diff = datediff.years + datediff.months / 12.0
        logger.debug(f"Date difference calculated: {years_diff} years")
        return years_diff
    except Exception as e:
        logger.error(f"Error calculating date difference: {str(e)}")
        raise


def chain_formatter(format_type: str, input_data) -> str:
    """Format resume/job inputs for inclusion in a runnable sequence.

    Args:
        format_type (str): The type of data to format (e.g., 'experience', 'projects', 'skills', 'education').
        input_data: The data to be formatted.

    Returns:
        str: The formatted data as a string.
    """
    logger.debug(f"Formatting chain input of type: {format_type}")
    logger.debug(f"Input data type: {type(input_data)}, content preview: {str(input_data)[:200] if input_data else 'None'}")

    try:
        if format_type == 'experience':
            as_list = format_experiences_for_prompt(input_data)
            formatted = format_prompt_inputs_as_strings(["experience"], experience=as_list)
            return formatted.get("experience", "")

        elif format_type == 'projects':
            as_list = format_projects_for_prompt(input_data)
            formatted = format_prompt_inputs_as_strings(["projects"], projects=as_list)
            return formatted.get("projects", "")

        elif format_type == 'skills':
            # Don't lose the skills data here!
            as_list = format_skills_for_prompt(input_data)
            formatted = format_prompt_inputs_as_strings(["skills"], skills=as_list)
            result = formatted.get("skills", "")
            logger.debug(f"Final formatted skills string: {result}")
            return result

        elif format_type == 'education':
            return format_education_for_resume(input_data)

        else:
            logger.debug(f"No specific formatter for type '{format_type}', returning input as string")
            # Convert to string for non-specific types
            if isinstance(input_data, (list, dict)):
                return str(input_data)
            return input_data or ""

    except Exception as e:
        logger.error(f"Error formatting chain input of type '{format_type}': {str(e)}")
        logger.error(f"Input data: {input_data}")
        # Return empty string rather than crashing
        return ""


def format_education_for_resume(education_list: list[dict]) -> str:
    """Format education entries for inclusion in a resume.

    Args:
        education_list (list[dict]): A list of dictionaries containing education details.

    Returns:
        str: A formatted string of education entries.
    """
    logger.debug(f"Formatting education list with {len(education_list)} entries")
    try:
        formatted_education = []
        for entry in education_list:
            school = entry.get('school', '')
            degrees = ', '.join(degree.get('names', ['Degree'])[0] for degree in entry.get('degrees', []))
            formatted_education.append(f"{school}: {degrees}")
        return '\n'.join(formatted_education)
    except Exception as e:
        logger.error(f"Error formatting education list: {str(e)}")
        raise


def format_skills_for_prompt(input_data) -> list:
    """Format skills for inclusion in a prompt.

    Args:
        input_data: The list of skills or skills structure.

    Returns:
        list: A formatted list of skills.
    """
    logger.debug(f"Formatting skills input: {type(input_data)} with content: {input_data}")

    # Handle None or empty input
    if not input_data:
        logger.warning("No skills data provided to format_skills_for_prompt")
        return []

    try:
        result = []

        # Handle different input formats
        if isinstance(input_data, list):
            # Standard format: list of categories
            for cat in input_data:
                if not isinstance(cat, dict):
                    logger.warning(f"Unexpected skill category format: {type(cat)}")
                    continue

                curr = ""
                if cat.get("category", ""):
                    curr += f"{cat['category']}: "

                # Handle subcategories format (new structure)
                if "subcategories" in cat:
                    skills_list = []
                    for subcat in cat["subcategories"]:
                        if isinstance(subcat, dict) and "skills" in subcat:
                            skills_list.extend(subcat["skills"])
                    if skills_list:
                        curr += "Proficient in "
                        curr += ", ".join(skills_list)
                        result.append(curr)

                # Handle direct skills format (legacy structure)
                elif "skills" in cat:
                    skills_list = cat["skills"]
                    if skills_list:
                        curr += "Proficient in "
                        curr += ", ".join(skills_list)
                        result.append(curr)

        elif isinstance(input_data, dict):
            # Handle if input_data is a dictionary directly
            for category, skills_list in input_data.items():
                if skills_list:
                    curr = f"{category}: Proficient in "
                    curr += ", ".join(skills_list)
                    result.append(curr)

        logger.debug(f"Formatted skills result: {result}")
        return result

    except Exception as e:
        logger.error(f"Error formatting skills: {str(e)}")
        logger.error(f"Input data type: {type(input_data)}, content: {input_data}")
        # Return empty list rather than crashing
        return []


def get_cumulative_time_from_titles(titles) -> int:
    """Calculate the cumulative time from job titles.

    Args:
        titles (list): A list of job titles with start and end dates.

    Returns:
        int: The cumulative time in years.
    """
    logger.debug(f"Calculating cumulative time from {len(titles)} job titles")
    result = 0.0
    try:
        for t in titles:
            if "startdate" in t and "enddate" in t:
                if t["enddate"] == "current":
                    last_date = datetime.today().strftime("%Y-%m-%d")
                    logger.debug(f"End date is 'current', using today's date: {last_date}")
                else:
                    last_date = t["enddate"]
                time_at_position = datediff_years(start_date=t["startdate"], end_date=last_date)
                logger.debug(f"Time at position: {time_at_position} years")
                result += time_at_position
        rounded_result = round(result)
        logger.debug(f"Total cumulative time: {rounded_result} years")
        return rounded_result
    except Exception as e:
        logger.error(f"Error calculating cumulative time from titles: {str(e)}")
        raise


def format_experiences_for_prompt(input_data) -> list:
    """Format experiences for inclusion in a prompt.

    Returns:
        list: A formatted list of experiences.
    """
    logger.debug(f"Formatting {len(input_data)} experiences for prompt")
    try:
        result = []
        for exp in input_data:
            curr = ""
            if "titles" in exp:
                exp_time = get_cumulative_time_from_titles(exp["titles"])
                curr += f"{exp_time} years experience in:"
            if "highlights" in exp:
                curr += format_list_as_string(exp["highlights"], list_sep="\n  - ")
                curr += "\n"
                result.append(curr)
        return result
    except Exception as e:
        logger.error(f"Error formatting experiences for prompt: {str(e)}")
        raise


def format_projects_for_prompt(input_data) -> list:
    """Format projects for inclusion in a prompt.

    Returns:
        list: A formatted list of projects.
    """
    logger.debug(f"Formatting {len(input_data)} projects for prompt")
    try:
        result = []
        for exp in input_data:
            curr = ""
            if "name" in exp:
                name = exp["name"]
                curr += f"Side Project: {name}"
            if "highlights" in exp:
                curr += format_list_as_string(exp["highlights"], list_sep="\n  - ")
                curr += "\n"
                result.append(curr)
        return result
    except Exception as e:
        logger.error(f"Error formatting projects for prompt: {str(e)}")
        raise