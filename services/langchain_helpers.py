from datetime import datetime
import json
import os
import time
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from openai import APIError, BadRequestError, OpenAI
from prompts import Prompts
import config

logger = config.getLogger("llm_helper")

_LOG_LLM_FLOW = os.getenv("LOG_LLM_FLOW", "true").lower() in {"1", "true", "yes", "on"}
_LOG_LLM_PROMPTS = os.getenv("LOG_LLM_PROMPTS", "false").lower() in {"1", "true", "yes", "on"}
_LOG_LLM_RESPONSES = os.getenv("LOG_LLM_RESPONSES", "false").lower() in {"1", "true", "yes", "on"}
_LOG_LLM_MAX_CHARS = int(os.getenv("LOG_LLM_MAX_CHARS", "4000"))


def _truncate_for_log(value, limit: int = _LOG_LLM_MAX_CHARS):
    """Bound log payload size so one model call does not flood the log stream."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    if len(value) <= limit:
        return value
    return value[:limit] + f"... [truncated {len(value) - limit} chars]"


def _safe_prompt_preview(messages):
    """Render provider-neutral messages into a bounded log preview."""
    preview = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        preview.append({
            "role": message.get("role"),
            "content": _truncate_for_log(message.get("content", "")),
        })
    return preview


def _safe_response_preview(response):
    """Serialize the parsed model response for logs."""
    if response is None:
        return None
    if hasattr(response, "output_parsed") and response.output_parsed is not None:
        parsed = response.output_parsed
    else:
        parsed = response

    if hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()
    return _truncate_for_log(parsed)


def invoke_structured(
    user,
    prompt_type: str,
    schema,
    *,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    **prompt_values,
):
    """Generate and validate a structured response with the official OpenAI SDK."""
    preferences = user.preferences or {}
    call_started_at = time.perf_counter()
    request = {
        "model": user.model,
        "input": Prompts.render_messages(prompt_type, **prompt_values),
        "text_format": schema,
        "store": False,
    }
    temperature = preferences.get("temperature")
    if temperature is not None:
        request["temperature"] = temperature
    if _LOG_LLM_FLOW:
        logger.info(
            "llm_request_started",
            extra={
                "prompt.type": prompt_type,
                "model.name": user.model,
                "timeout.seconds": timeout_seconds,
                "max_retries": max_retries,
                "schema.name": getattr(schema, "__name__", str(schema)),
                "llm.input_keys": sorted(prompt_values.keys()),
                "llm.temperature": temperature,
            },
        )
        if _LOG_LLM_PROMPTS:
            logger.info(
                "llm_request_prompt",
                extra={
                    "prompt.type": prompt_type,
                    "llm.prompt_preview": _safe_prompt_preview(request["input"]),
                },
            )
    try:
        client = OpenAI(
            api_key=user.api_key,
            max_retries=max_retries,
            timeout=timeout_seconds,
        )
        try:
            response = client.responses.parse(**request)
        except BadRequestError as error:
            if "temperature" not in request or "temperature" not in str(error).lower():
                raise
            logger.info("llm_temperature_not_supported_retrying", extra={"prompt.type": prompt_type})
            request.pop("temperature")
            response = client.responses.parse(**request)
        if response.output_parsed is None:
            raise ValueError("Model returned no structured output")
        if _LOG_LLM_FLOW:
            logger.info(
                "llm_request_completed",
                extra={
                    "prompt.type": prompt_type,
                    "model.name": user.model,
                    "event.duration_ms": round((time.perf_counter() - call_started_at) * 1000, 2),
                },
            )
            if _LOG_LLM_RESPONSES:
                logger.info(
                    "llm_response_parsed",
                    extra={
                        "prompt.type": prompt_type,
                        "llm.response_preview": _safe_response_preview(response),
                    },
                )
        return response.output_parsed
    except APIError as e:
        if _LOG_LLM_FLOW:
            logger.warning(
                "llm_request_api_error",
                extra={
                    "prompt.type": prompt_type,
                    "model.name": user.model,
                    "event.duration_ms": round((time.perf_counter() - call_started_at) * 1000, 2),
                    "timeout.seconds": timeout_seconds,
                    "error.reason": str(e),
                },
            )
        logger.warning("llm_request_failed", extra={"prompt.type": prompt_type, "timeout.seconds": timeout_seconds, "error.reason": str(e)})
        raise
    except Exception as e:
        if _LOG_LLM_FLOW:
            logger.error(
                "llm_request_unhandled_error",
                extra={
                    "prompt.type": prompt_type,
                    "model.name": user.model,
                    "event.duration_ms": round((time.perf_counter() - call_started_at) * 1000, 2),
                    "error.reason": str(e),
                },
            )
        logger.error("llm_structured_generation_failed", extra={"prompt.type": prompt_type, "error.reason": str(e)})
        raise


def format_list_as_string(lst: list, list_sep: str = "\n- ") -> str:
    """Format a list as a string with a specified separator."""
    if isinstance(lst, list):
        return list_sep + list_sep.join(lst)
    return str(lst)



def parse_date(date_str: str) -> datetime:
    """Given an arbitrary string, parse it to a date."""
    logger.debug("date_parse_started")
    default_date = datetime(datetime.today().year, 1, 1)
    try:
        parsed_date = dateparser.parse(str(date_str), default=default_date)
        logger.debug("date_parse_completed")
        return parsed_date
    except (TypeError, ValueError, OverflowError) as e:
        logger.error("date_parse_failed", extra={"error.reason": str(e)})
        raise e


def datediff_years(start_date: str, end_date: str) -> float:
    """Calculate the difference between two dates in fractional years."""
    logger.debug("date_difference_started")
    if isinstance(end_date, str) and end_date.lower() == "present":
        end_date = datetime.today().strftime("%Y-%m-%d")
        logger.debug("date_difference_current_end_date_used")

    try:
        start = parse_date(start_date)
        end = parse_date(end_date)
        datediff = relativedelta(end, start)
        years_diff = datediff.years + datediff.months / 12.0
        logger.debug("date_difference_completed", extra={"date_difference.years": years_diff})
        return years_diff
    except Exception as e:
        logger.error("date_difference_failed", extra={"error.reason": str(e)})
        raise


def chain_formatter(format_type: str, input_data) -> str:
    """Format resume/job inputs for inclusion in a runnable sequence."""
    logger.debug("prompt_input_formatting_started", extra={"format.type": format_type})

    try:
        if format_type in {'experience', 'experiences'}:
            as_list = format_experiences_for_prompt(input_data)
            return format_list_as_string(as_list)

        elif format_type in {'project', 'projects'}:
            as_list = format_projects_for_prompt(input_data)
            return format_list_as_string(as_list)

        elif format_type == 'skills':
            as_list = format_skills_for_prompt(input_data)
            return format_list_as_string(as_list)

        else:
            if isinstance(input_data, (list, dict)):
                return json.dumps(input_data, ensure_ascii=False)
            return input_data or ""

    except Exception as e:
        logger.error("prompt_input_formatting_failed", extra={"format.type": format_type, "error.reason": str(e)})
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
        logger.error("skills_formatting_failed", extra={"error.reason": str(e)})
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
        logger.error("cumulative_time_calculation_failed", extra={"error.reason": str(e)})
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
        logger.error("experience_formatting_failed", extra={"error.reason": str(e)})
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
        logger.error("project_formatting_failed", extra={"error.reason": str(e)})
        raise
