from datetime import datetime
import json
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from openai import APIError, BadRequestError, OpenAI
from prompts import Prompts
import config

logger = config.getLogger("llm_helper")


def invoke_structured(user, prompt_type: str, schema, **prompt_values):
    """Generate and validate a structured response with the official OpenAI SDK."""
    preferences = user.preferences or {}
    request = {
        "model": user.model,
        "input": Prompts.render_messages(prompt_type, **prompt_values),
        "text_format": schema,
        "store": False,
    }
    temperature = preferences.get("temperature")
    if temperature is not None:
        request["temperature"] = temperature
    try:
        client = OpenAI(api_key=user.api_key, max_retries=2, timeout=60.0)
        try:
            response = client.responses.parse(**request)
        except BadRequestError as error:
            if "temperature" not in request or "temperature" not in str(error).lower():
                raise
            logger.info("Model does not accept temperature; retrying with its default sampling settings")
            request.pop("temperature")
            response = client.responses.parse(**request)
        if response.output_parsed is None:
            raise ValueError("Model returned no structured output")
        return response.output_parsed
    except APIError as e:
        logger.error(f"OpenAI request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Structured generation failed: {e}")
        raise


def format_list_as_string(lst: list, list_sep: str = "\n- ") -> str:
    """Format a list as a string with a specified separator."""
    if isinstance(lst, list):
        return list_sep + list_sep.join(lst)
    return str(lst)



def parse_date(date_str: str) -> datetime:
    """Given an arbitrary string, parse it to a date."""
    logger.debug(f"Parsing date string: {date_str}")
    default_date = datetime(datetime.today().year, 1, 1)
    try:
        parsed_date = dateparser.parse(str(date_str), default=default_date)
        logger.debug(f"Successfully parsed date '{date_str}' to {parsed_date}")
        return parsed_date
    except (TypeError, ValueError, OverflowError) as e:
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
        if format_type in {'experience', 'experiences'}:
            as_list = format_experiences_for_prompt(input_data)
            return format_list_as_string(as_list)

        elif format_type in {'project', 'projects'}:
            as_list = format_projects_for_prompt(input_data)
            return format_list_as_string(as_list)

        elif format_type == 'skills':
            as_list = format_skills_for_prompt(input_data)
            formatted = format_prompt_inputs_as_strings(["skills"], skills=as_list)
            result = formatted.get("skills", "")
            return result

        else:
            if isinstance(input_data, (list, dict)):
                return json.dumps(input_data, ensure_ascii=False)
            return input_data or ""

    except Exception as e:
        logger.error(f"Error formatting chain input of type '{format_type}': {str(e)}")
        return ""


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
        for exp in input_data or []:
            if not isinstance(exp, dict):
                continue
            title = exp.get("title") or exp.get("name") or "Role"
            company = exp.get("company") or ""
            curr = f"{title} {('at ' + company) if company else ''}".strip()
            if "titles" in exp:
                exp_time = get_cumulative_time_from_titles(exp["titles"])
                curr += f" ({exp_time} years)"
            highlights = exp.get("highlights") or []
            if highlights:
                curr += format_list_as_string(highlights, list_sep="\n  - ")
            result.append(curr)
        return result
    except Exception as e:
        logger.error(f"Error formatting experiences: {str(e)}")
        raise


def format_projects_for_prompt(input_data) -> list:
    """Format projects for inclusion in a prompt."""
    try:
        result = []
        for exp in input_data or []:
            if not isinstance(exp, dict):
                continue
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
