from datetime import datetime
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache, get_llm_cache
from langchain_community.cache import InMemoryCache
import config
import redis


redis_client = redis.Redis(host='infra-redis', port=6379, db=0)
set_llm_cache(RedisCache(redis_client))

logger = config.getLogger("langchain_helper")  # Get the logger from config


def create_llm(user, **kwargs):
    """Create an LLM instance with specified parameters."""
    chat_model = kwargs.pop("chat_model", ChatOpenAI)
    kwargs.setdefault("model_name", user.model)
    kwargs.setdefault("cache", True) # Explicitly enable caching
    kwargs.setdefault("api_key", user.api_key)
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
        # Properly clear cache if it exists
        current_cache = get_llm_cache()
        if current_cache:
            current_cache.clear()
        logger.error(f"Date input `{date_str}` could not be parsed: {str(e)}")
        raise e


def datediff_years(start_date: str, end_date: str) -> float:
    """Calculate the difference between two dates in fractional years."""
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
    """Format resume/job inputs for inclusion in a runnable sequence."""
    logger.debug(f"Formatting chain input of type: {format_type}")

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
            as_list = format_skills_for_prompt(input_data)
            formatted = format_prompt_inputs_as_strings(["skills"], skills=as_list)
            result = formatted.get("skills", "")
            return result

        elif format_type == 'education':
            return format_education_for_resume(input_data)

        else:
            if isinstance(input_data, (list, dict)):
                return str(input_data)
            return input_data or ""

    except Exception as e:
        logger.error(f"Error formatting chain input of type '{format_type}': {str(e)}")
        return ""


def format_education_for_resume(education_list: list[dict]) -> str:
    """Format education entries for inclusion in a resume."""
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
    """Format skills for inclusion in a prompt."""
    if not input_data:
        return []

    try:
        result = []
        if isinstance(input_data, list):
            for cat in input_data:
                if not isinstance(cat, dict): continue
                curr = f"{cat.get('category', '')}: "

                if "subcategories" in cat:
                    skills_list = []
                    for subcat in cat["subcategories"]:
                        if isinstance(subcat, dict) and "skills" in subcat:
                            skills_list.extend(subcat["skills"])
                    if skills_list:
                        curr += "Proficient in " + ", ".join(skills_list)
                        result.append(curr)
                elif "skills" in cat:
                    if cat["skills"]:
                        curr += "Proficient in " + ", ".join(cat["skills"])
                        result.append(curr)

        elif isinstance(input_data, dict):
            for category, skills_list in input_data.items():
                if skills_list:
                    result.append(f"{category}: Proficient in " + ", ".join(skills_list))

        return result
    except Exception as e:
        logger.error(f"Error formatting skills: {str(e)}")
        return []


def get_cumulative_time_from_titles(titles) -> int:
    """Calculate the cumulative time from job titles."""
    result = 0.0
    try:
        for t in titles:
            if "startdate" in t and "enddate" in t:
                last_date = datetime.today().strftime("%Y-%m-%d") if t["enddate"] == "current" else t["enddate"]
                result += datediff_years(start_date=t["startdate"], end_date=last_date)
        return round(result)
    except Exception as e:
        logger.error(f"Error calculating cumulative time: {str(e)}")
        raise


def format_experiences_for_prompt(input_data) -> list:
    """Format experiences for inclusion in a prompt."""
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
        logger.error(f"Error formatting experiences: {str(e)}")
        raise


def format_projects_for_prompt(input_data) -> list:
    """Format projects for inclusion in a prompt."""
    try:
        result = []
        for exp in input_data:
            curr = ""
            if "name" in exp:
                curr += f"Side Project: {exp['name']}"
            if "highlights" in exp:
                curr += format_list_as_string(exp["highlights"], list_sep="\n  - ")
                curr += "\n"
                result.append(curr)
        return result
    except Exception as e:
        logger.error(f"Error formatting projects: {str(e)}")
        raise