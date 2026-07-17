"""Schema inference and format injection for entity_collect.

entity_collect must merge results from multiple sampling paths into one table. When the user
does not specify an output format, this module infers the minimum necessary columns and appends
Markdown-table requirements to the subtask, reducing schema noise in split-verify-merge.
"""
from __future__ import annotations

import re

from llm_infer.llm_infer import llm_infer


# Step 1: detect whether the task already specifies format requirements.

_FORMAT_PATTERNS = [
    r"```markdown",
    r"output\s+format\s+is",
    r"organize\s+.*\s+in\s+.*\s*markdown",
    r"present\s+.*\s+in\s+.*\s*markdown",
    r"one\s+markdown\s+table",
    r"markdown\s+table\s+with",
    r"\bcolumn\s*(name|header)s?\s*[:：]",
    r"the\s+column\s+.*\s+is\s*[:：]",
    r"\{data_content\}",
]


def detect_has_format_requirement(task: str) -> bool:
    """Detect whether task text already contains Markdown-table format requirements."""
    for pattern in _FORMAT_PATTERNS:
        if re.search(pattern, task, re.IGNORECASE):
            return True
    return False


# Step 2: use the LLM to infer the schema (column names).

_MAX_COL_NAME_LENGTH = 50

SCHEMA_INFER_SYSTEM_PROMPT = """\
You are a data schema designer. Given a user's research/collection task, \
infer the most appropriate Markdown table column names to organize the results.

Rules:
1. **Focus on the core entity identifier** — the column(s) that uniquely identify each row \
(e.g., "Museum", "University", "Product Name"). These are the primary keys.
2. **Keep it minimal** — only include columns that are explicitly mentioned or strongly implied by the task. \
Do NOT add speculative columns the user didn't ask for.
3. **Use clear, concise English column names** — e.g., "Restaurant", "Launch Date", "Company Name".
4. **Respect the task's language** — if the task is in Chinese, use Chinese column names; \
if in English, use English column names.
5. **Order matters** — put the most important identifier column first.
6. **Each column name must be SHORT** — a few words at most (e.g., "Museum", "Launch Date"). \
Never include descriptions, parenthetical remarks, or explanations in a column name.

Output format:
Return ONLY a Markdown bullet list of column names. One column per line, prefixed with `- `.
Each line must contain ONLY the column name itself — no descriptions, no parentheses, no commentary.
Do NOT output anything else (no explanation, no reasoning, no code block, no numbering).

Example output:
- Museum
- Address

If you cannot determine any columns, return:
- Item
"""


def infer_schema(
    task: str,
    provider: str,
    model: str,
    max_retries: int = 3,
) -> list[str]:
    """Use an LLM to infer appropriate table-column names from the task description."""
    messages = [
        {"role": "system", "content": SCHEMA_INFER_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"## Task\n{task}\n\n"
            "## Instruction\n"
            "Based on the task above, infer the column names for a Markdown table "
            "to organize the search results. Return ONLY a Markdown bullet list of column names."
        )},
    ]

    content = ""
    for attempt in range(max_retries):
        try:
            raw_response = llm_infer(
                provider=provider,
                model=model,
                messages=messages,
                tools=None,
                generation_config={"max_tokens": 1024},
            )
            content = raw_response.get("content", "") or ""
            if content.strip():
                break
        except Exception as e:
            print(f"[SchemaInfer] LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            continue

    columns = _parse_columns_from_llm(content)
    if columns:
        print(f"[SchemaInfer] Inferred columns: {columns}")
        return columns

    print("[SchemaInfer] Failed to infer columns, using fallback ['Item']")
    return ["Item"]


_REASONING_NOISE = re.compile(
    r"(wait|let me|let's|actually|hmm|but |so |no[,.]|the user|recheck|implied|I think)",
    re.IGNORECASE,
)


def _parse_columns_from_llm(text: str) -> list[str] | None:
    """Parse a list of column names in Markdown bullet-list format from an LLM response."""
    columns: list[str] = []
    seen: set[str] = set()
    for line in text.strip().splitlines():
        line = line.strip()
        m = re.match(r"^(?:[-*•]|\d+[.)\]])\s+(.+)$", line)
        if m:
            col = m.group(1).strip().strip("`\"'")
            if not col:
                continue
            if len(col) > _MAX_COL_NAME_LENGTH:
                continue
            if _REASONING_NOISE.search(col):
                continue
            key = col.lower()
            if key in seen:
                continue
            seen.add(key)
            columns.append(col)
    return columns if columns else None


# Step 3: inject format requirements.

_FORMAT_TEMPLATE_SINGLE = """\

Please organize the results in one Markdown table with the following column:
{columns_str}

Don't ask me any questions, just output the results according to the column without omitting entries arbitrarily. The output format is ```markdown
{{data_content}}
```."""

_FORMAT_TEMPLATE_MULTI = """\

Please organize the results in one Markdown table with the following columns:
{columns_str}

Don't ask me any questions, just output the results according to the columns without omitting entries arbitrarily. The output format is ```markdown
{{data_content}}
```."""


def inject_format_requirement(task: str, columns: list[str]) -> str:
    """Turn inferred columns into standardized format instructions and append them to the task."""
    columns_str = ", ".join(columns)
    if len(columns) == 1:
        suffix = _FORMAT_TEMPLATE_SINGLE.format(columns_str=columns_str)
    else:
        suffix = _FORMAT_TEMPLATE_MULTI.format(columns_str=columns_str)
    return task.rstrip() + suffix


# Unified entry point.

def auto_enhance_task(
    task: str,
    provider: str,
    model: str,
) -> tuple[str, dict]:
    """Enhance a task automatically: detect → infer → inject."""
    log: dict = {"original_task": task}

    if detect_has_format_requirement(task):
        log["skipped"] = True
        log["reason"] = "task_already_has_format_requirement"
        print("[SchemaInfer] Task already has format requirement, skipping injection.")
        return task, log

    log["skipped"] = False

    columns = infer_schema(
        task=task,
        provider=provider,
        model=model,
    )
    log["inferred_columns"] = columns

    enhanced_task = inject_format_requirement(task, columns)
    log["enhanced_task"] = enhanced_task

    print(f"[SchemaInfer] Injected format requirement with columns: {columns}")
    return enhanced_task, log
