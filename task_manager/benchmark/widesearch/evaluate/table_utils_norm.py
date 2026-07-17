"""
Table-value preprocessing and normalization.

Normalize cell-value formats before metric computation to eliminate superficial differences.
Supported preprocessors:
  - extract_number: extract the first numeric value from text, handling percent signs and commas
  - norm_str:       lowercase and remove spaces and Markdown bold markers
  - norm_date:      use dateparser to normalize multiple date formats to YYYY-MM-DD
"""

import re
import dateparser
from typing import Any, Callable


# ── Preprocessor registry ─────────────────────────────────────────
preprocess_function_registry: dict[str, Callable[..., str]] = {}


def register_preprocess_function(func: Callable[..., str]):
    """Register a preprocessor in the global dict for reference by name from configuration."""
    preprocess_function_registry[func.__name__] = func
    return func


@register_preprocess_function
def extract_number(content: str) -> str:
    """Extract the first decimal, negative, or percentage value and remove thousands separators; return "NULL" if absent."""
    numbers = re.findall(r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?", str(content).replace(",", ""))
    if len(numbers) == 0:
        return "NULL"
    return numbers[0]


@register_preprocess_function
def norm_str(content: Any) -> str:
    """Normalize a string: lowercase → strip → remove all spaces → remove Markdown bold marker *."""
    return str(content).lower().strip().replace(" ", "").replace("*", "")


@register_preprocess_function
def norm_date(content: str) -> str:
    """Normalize a parsed date to YYYY-MM-DD with dateparser, returning the original value on failure."""
    normalized_date = dateparser.parse(content, settings={"PREFER_DAY_OF_MONTH": "first"})

    if normalized_date is None:
        return content
    else:
        return normalized_date.strftime("%Y-%m-%d")


def preprocess_call(content, preprocess_func_name):
    """Unified dispatch entry point that looks up and calls a preprocessor by name."""
    assert (
        preprocess_func_name in preprocess_function_registry
    ), f"preprocess_func_name {preprocess_func_name} not in preprocess_function_registry"

    preprocess_func = preprocess_function_registry[preprocess_func_name]
    return preprocess_func(content)
