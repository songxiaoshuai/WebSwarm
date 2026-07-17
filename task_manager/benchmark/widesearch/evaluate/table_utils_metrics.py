"""
Table-evaluation metric functions.

Provides two evaluation approaches:
  1. Rule-based metrics: exact_match / url_match / in_match / number_near / date_near
     evaluate each cell independently and return (score, msg)
  2. LLM judge (llm_judge_column): package an entire column for batch scoring by the LLM
"""

import re
import dateparser
from urllib.parse import urlparse
from typing import Callable

from .table_utils import parse_markdown_json


# ── Metric-function registry ────────────────────────────────────
# Decorators register all metric functions in this dict for dynamic dispatch by name through metric_call.
metric_function_registry: dict[str, Callable[..., tuple[float, str]]] = {}


def register_metric_function(func: Callable[..., tuple[float, str]]):
    """Register a metric in the global dict for dynamic dispatch by string name."""
    metric_function_registry[func.__name__] = func
    return func


@register_metric_function
def exact_match(response: str, target: str) -> tuple[float, str]:
    """Case-insensitive exact match."""
    if response.lower() == target.lower():
        return 1.0, f"exact match, response: {response}, target: {target}"
    return 0.0, f"exact not match, response: {response}, target: {target}"


@register_metric_function
def url_match(response: str, target: str) -> tuple[float, str]:
    """URL match comparing only the domain (netloc) and ignoring path and scheme differences."""
    url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

    response_urls = url_pattern.findall(response)
    target_urls = url_pattern.findall(target)
    response_urls = [urlparse(url).netloc for url in response_urls]
    target_urls = [urlparse(url).netloc for url in target_urls]

    if set(response_urls) == set(target_urls):
        return 1.0, f"url match, response: {response}, target: {target}"
    return 0.0, f"url not match, response: {response}, target: {target}"


@register_metric_function
def in_match(response: str, target: str) -> tuple[float, str]:
    """Containment match checking whether response is a substring of target."""
    if response in target:
        return 1.0, f"response in target, response: {response}, target: {target}"
    return 0.0, f"response not in target, response: {response}, target: {target}"


@register_metric_function
def number_near(response: str, target: str, criterion: float) -> tuple[float, str]:
    """Approximate numeric match that passes when |response - target| ≤ |target| × criterion.

    Handles percentages automatically (for example, "12.5%" → 0.125).
    """
    if "%" in response:
        try:
            response_num = float(response.replace("%", "")) / 100.0
        except (ValueError, TypeError):
            response_num = None
    else:
        try:
            response_num = float(response)
        except (ValueError, TypeError):
            response_num = None
    if "%" in target:
        try:
            target_num = float(target.replace("%", "")) / 100.0
        except (ValueError, TypeError):
            target_num = None
    else:
        try:
            target_num = float(target)
        except (ValueError, TypeError):
            target_num = None

    if response_num is None or target_num is None:
        if response_num is None and target_num is None and response == target:
            return 1.0, f"number equal, response: {response}, target: {target}"
        return (
            0.0,
            f"number not convertable, response: {response_num}, target: {target_num}",
        )
    if abs((response_num - target_num)) <= abs(target_num) * criterion:
        return (
            1.0,
            f"number near in range {criterion * 100}%, response: {response_num}, target: {target_num}",
        )
    return 0.0, f"number not near, response: {response_num}, target: {target_num}"


@register_metric_function
def date_near(response: str, target: str) -> tuple[float, str]:
    """Approximate date match that passes within 31 days and uses dateparser for multiple formats."""
    try:
        response_date = dateparser.parse(response, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        response_date = None

    try:
        target_date = dateparser.parse(target, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        target_date = None

    if response_date is None or target_date is None:
        if response_date is None and target_date is None:
            return 1.0, f"date near, response: {response}, target: {target}"
        return 0.0, f"date not convertable, response: {response}, target: {target}"

    if abs((response_date - target_date).days) <= 31:
        return 1.0, f"date near, response: {response_date}, target: {target_date}"
    return 0.0, f"date not near, response: {response_date}, target: {target_date}"


def metric_call(response, target, criterion, metric_func_name) -> tuple[float, str]:
    """Unified dispatch entry point that looks up and calls a metric function by name."""
    assert (
        metric_func_name in metric_function_registry
    ), f"metric_func_name {metric_func_name} not in metric_function_registry"

    metric_func = metric_function_registry[metric_func_name]
    if metric_func_name == "number_near":
        score, msg = metric_func(response, target, criterion)
    else:
        score, msg = metric_func(response, target)

    print(f"metric_func_name {metric_func_name} score {score} msg {msg}")
    return score, msg


def llm_judge_column(
    response: list[str],
    target: list[str],
    criterion: str,
    prompt_id: str,
    call_llm: Callable[[str, dict], str],
    max_retries: int = 5,
) -> tuple[list[float], list[str]]:
    """LLM-judge mode: package a full (response, target) column for batch scoring by the LLM.

    Retry conditions, up to max_retries attempts:
      - The LLM returns an empty result
      - JSON code-block parsing fails
      - Too few score keys are returned (missing keys receive 0 by default)

    Return all-zero scores after every retry fails.
    """
    response_dict = {}
    for idx, (resp, tar) in enumerate(zip(response, target)):
        response_dict[f"idx_{idx}"] = {"response": resp, "target": tar}

    prompt_kwargs = {
        "criterion": criterion,
        "response": response_dict,
    }

    for attempt in range(max_retries):
        result = call_llm(prompt_id, prompt_kwargs)
        if not result:
            print(f"[llm_judge_column] WARNING: LLM returned an empty result (attempt {attempt + 1}/{max_retries})")
            continue

        score_dict = parse_markdown_json(result)
        if score_dict is None:
            print(f"[llm_judge_column] WARNING: Failed to parse JSON (attempt {attempt + 1}/{max_retries}), LLM output length={len(result)}")
            continue

        score_list = [score_dict.get(f"idx_{idx}", 0) for idx in range(len(response))]
        expected_keys = {f"idx_{idx}" for idx in range(len(response))}
        missing_keys = expected_keys - set(score_dict.keys())
        if missing_keys:
            print(f"[llm_judge_column] WARNING: Missing {len(missing_keys)}/{len(expected_keys)} scoring keys (attempt {attempt + 1}/{max_retries}): {sorted(missing_keys)[:5]}...")
            continue

        msg_list = [result] * len(response)
        return score_list, msg_list

    score_list = [0.0] * len(response)
    msg_list = [f"llm judge failed after {max_retries} retries"] * len(response)
    return score_list, msg_list
