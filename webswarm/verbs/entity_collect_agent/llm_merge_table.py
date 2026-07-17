"""LLM table-merging utility for entity_collect.

In its final step, split-verify-merge combines intersection and verified_diff into one Markdown
table. This module encapsulates pure LLM union/intersection merging without search verification.
"""
from llm_infer.llm_infer import llm_infer


# System prompt for taking a recall-biased union.
MERGE_SYSTEM_PROMPT_UNION = """\
You are a precise data deduplication and merging expert.
Your job is to merge multiple tables (from different search agents) into ONE deduplicated, complete table.

Rules:
1. **Keep every unique row** — if a row appears in any source table, include it.
2. **Merge duplicate rows** — if two rows clearly describe the same entity (even with minor wording differences), keep only one merged version with the most complete/accurate values.
3. **Unify column names** — all source tables describe the same schema; reconcile any column name differences.
4. **Do NOT invent data** — only use information present in the source tables.
5. **Do NOT drop rows unless they are true duplicates** — when in doubt, keep both rows.
6. **Preserve the column order from the original task description if specified.**

Output format:
Output a Markdown table ```markdown\\n{data_content}\\n``` according to the format requirements in the subsequent Original Task; do not output any other content.
"""

# System prompt for taking a precision-biased intersection.
MERGE_SYSTEM_PROMPT_INTERSECTION = """\
You are a precise data deduplication and filtering expert.
Your job is to merge multiple tables (from different search agents) into ONE high-confidence table by keeping ONLY the rows that are corroborated across multiple sources.

Rules:
1. **Keep only corroborated rows** — a row should appear in the final table ONLY if the same entity (identified by its primary key / unique identifier) appears in **at least 2** source tables. Rows that appear in only one source table should be DROPPED, as they are likely noise or hallucinations.
2. **Merge duplicate rows** — if two or more rows clearly describe the same entity (even with minor wording differences), keep one merged version with the most complete/accurate values.
3. **Unify column names** — all source tables describe the same schema; reconcile any column name differences.
4. **Do NOT invent data** — only use information present in the source tables.
5. **When in doubt, DROP the row** — prioritize precision over recall. Only include rows you are confident are correct and confirmed by multiple sources.
6. **Preserve the column order from the original task description if specified.**
7. **Edge case**: if there are only 2 source tables and they share very few rows, still only keep the shared rows. An empty table is acceptable if no rows are corroborated.

Output format:
Output a Markdown table ```markdown\\n{data_content}\\n``` according to the format requirements in the subsequent Original Task; do not output any other content.
"""


def _build_merge_user_prompt(task: str, tables: list[str]) -> str:
    """Build the user prompt for the merge stage."""
    parts = [f"## Original Task\n{task}\n"]
    for i, table in enumerate(tables, 1):
        parts.append(f"## Source Table {i}\n{table}\n")
    parts.append(
        "## Instruction\n"
        "Merge all source tables above into a single deduplicated table.\n"
        "Output ONLY a Markdown table according to the format requirements in the Original Task; do not output any other content."
    )
    return "\n".join(parts)


def llm_merge_tables(
    task: str,
    tables: list[str],
    provider: str,
    model: str,
    max_tokens: int = 32768,
    return_log: bool = False,
    merge_strategy: str = "intersection",
) -> str | tuple[str, dict]:
    """
    Pure LLM merge function that combines k tables into one complete deduplicated Markdown table.

    Args:
        task: Original task description (query)
        tables: List containing k tables
        provider / model: LLM configuration; temperature/enable_thinking use llm_infer defaults
        max_tokens: Maximum output tokens
        return_log: Whether to return a merge log
        merge_strategy: "union" or "intersection"

    Returns:
        If return_log=False: str
        If return_log=True:  (str, dict)
    """
    if not tables:
        if return_log:
            return "", {"messages": [], "raw_response": {}, "skipped": "no_tables"}
        return ""

    if len(tables) == 1:
        if return_log:
            return tables[0], {"messages": [], "raw_response": {}, "skipped": "single_table"}
        return tables[0]

    strategy_prompts = {
        "union": MERGE_SYSTEM_PROMPT_UNION,
        "intersection": MERGE_SYSTEM_PROMPT_INTERSECTION,
    }
    system_prompt = strategy_prompts.get(merge_strategy)
    if system_prompt is None:
        raise ValueError(f"Unknown merge_strategy '{merge_strategy}'. Choose from: {list(strategy_prompts.keys())}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_merge_user_prompt(task, tables)},
    ]

    content = ""
    reasoning_content = ""
    cur_try = 0
    max_retries = 3
    raw_response = {}
    while cur_try < max_retries:
        try:
            raw_response = llm_infer(
                provider=provider,
                model=model,
                messages=messages,
                tools=None,
                generation_config={"max_tokens": max_tokens},
            )
        except Exception as e:
            print(f"[llm_merge_tables] LLM call failed (attempt {cur_try+1}/{max_retries}): {e}")
            cur_try += 1
            continue
        reasoning_content = raw_response.get("reasoning_content", "") or ""
        content = raw_response.get("content", "") or ""
        token_usage = raw_response.get("token_usage", {})
        if token_usage.get("completion_tokens", 0) >= max_tokens:
            print(f"[llm_merge_tables] Output tokens maxed out ({max_tokens}), retry {cur_try+1}/{max_retries}")
            cur_try += 1
            continue
        if content.strip():
            break
        cur_try += 1

    if not content.strip():
        print(f"[llm_merge_tables] Warning: all {max_retries} retries exhausted, content is empty")

    messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning_content})
    if return_log:
        log = {
            "messages": messages,
            "raw_response": raw_response,
        }
        return content, log
    return content
