"""split-verify-merge flow for entity_collect.

Multi-path sampling produces several candidate entity tables. An LLM first splits candidates
into intersection (seen in at least two paths) and diff (seen in one path). Search Verifier
then researches diff, and the high-confidence set is union-merged with verified_diff.
"""
from __future__ import annotations

import re

from llm_infer.llm_infer import llm_infer

from .search_verifier import run_search_verifier


# System prompt for LLM Split.
LLM_SPLIT_SYSTEM_PROMPT = """\
You are a precise data aggregation and verification expert.

You will receive:
1. A task description
2. Multiple source tables produced by different search agents for the same task

Your job is to:
- identify overlapping entities across tables
- split them into high-confidence and low-confidence groups
- STRICTLY preserve only the attributes explicitly required by the task

--------------------------------------------------
PRIMARY RULE: TASK DEFINES THE OUTPUT SCHEMA
--------------------------------------------------

The task description defines:
- which entities are relevant
- which attributes/columns are allowed in the final output

You MUST NOT infer additional output fields from the source tables.

Even if source tables contain extra metadata, attributes, statistics, descriptions, rankings, locations, dates, or related entities:
- discard them unless explicitly required by the task
- NEVER propagate extra columns into the final output

The source tables are noisy retrieval outputs.
The task is the ONLY authority for determining the final schema.

--------------------------------------------------
ENTITY SPLITTING
--------------------------------------------------

Split entities into two groups:

1. INTERSECTION
- entities appearing in AT LEAST 2 source tables
- considered higher-confidence

2. DIFF
- entities appearing in ONLY 1 source table
- considered lower-confidence

--------------------------------------------------
ENTITY MATCHING RULES
--------------------------------------------------

Two rows refer to the same entity if:
- their primary identifier is identical
- OR clearly equivalent despite minor variations:
  - abbreviations
  - spelling differences
  - formatting differences
  - language variants
  - aliases

Use semantic matching conservatively.
Do NOT merge unrelated entities.

--------------------------------------------------
MERGING RULES
--------------------------------------------------

For INTERSECTION rows:
- merge duplicate rows into ONE row
- keep the most complete and accurate values
- if conflicting values exist, prefer:
  1. values consistent across more sources
  2. more specific/canonical forms
  3. values explicitly aligned with the task

For DIFF rows:
- keep the original row content
- BUT remove attributes not required by the task

--------------------------------------------------
COLUMN / ATTRIBUTE RULES
--------------------------------------------------

IMPORTANT:
Before generating output, infer the MINIMAL REQUIRED COLUMN SET from the task description.

The final tables MUST:
- contain ONLY task-relevant columns
- remove all irrelevant attributes
- avoid carrying over retrieval noise

If the task specifies:
- exact fields
- attribute names
- ordering
- schema structure

You MUST follow them exactly.

If the task does NOT explicitly specify columns:
- infer the minimal necessary schema required to answer the task
- prefer fewer columns over more columns

NEVER add:
- auxiliary metadata
- rankings
- explanations
- notes
- source provenance
- confidence scores
- unrelated dates
- unrelated statistics
- extra entity attributes

--------------------------------------------------
DATA INTEGRITY RULES
--------------------------------------------------

- Do NOT invent data
- Do NOT hallucinate missing values
- Do NOT infer unsupported attributes
- Only use information explicitly present in the source tables

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Output format:
You MUST output exactly two Markdown tables in the following structure:

## INTERSECTION
```markdown
(table of entities appearing in ≥2 sources, or empty header if none)
```

## DIFF
```markdown
(table of entities appearing in only 1 source, or empty header if none)
```

Do not output anything else.
"""

def _build_split_user_prompt(task: str, tables: list[str]) -> str:
    """Build the user prompt for LLM split."""
    parts = [f"## Original Task\n{task}\n"]
    for i, table in enumerate(tables, 1):
        parts.append(f"## Source Table {i}\n{table}\n")
    parts.append(
        "## Instruction\n"
        "Analyze all source tables above and split entities into:\n"
        "1. INTERSECTION: entities appearing in ≥2 sources\n"
        "2. DIFF: entities appearing in only 1 source\n"
        "Output exactly two labeled Markdown tables as specified."
    )
    return "\n".join(parts)


def _parse_split_output(content: str) -> tuple[str, str]:
    """
    Parse the INTERSECTION and DIFF tables from LLM output.

    Returns:
        (intersection_md, diff_md): Two Markdown-table strings, empty on parse failure
    """
    # Match the ## INTERSECTION and ## DIFF blocks and extract their ```markdown ... ``` content.
    intersection_md = ""
    diff_md = ""

    # Extract the Markdown table from the INTERSECTION block.
    inter_match = re.search(
        r"##\s*INTERSECTION\s*```markdown(.*?)```",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if inter_match:
        intersection_md = "```markdown" + inter_match.group(1) + "```"

    # Extract the Markdown table from the DIFF block.
    diff_match = re.search(
        r"##\s*DIFF\s*```markdown(.*?)```",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if diff_match:
        diff_md = "```markdown" + diff_match.group(1) + "```"

    return intersection_md, diff_md


def llm_split_tables(
    task: str,
    tables: list[str],
    provider: str,
    model: str,
    max_tokens: int = 32768,
    max_retries: int = 3,
) -> tuple[str, str, dict]:
    """
    Ask the LLM to divide multiple source tables into intersection and diff groups.

    Args:
        task: Original task description
        tables: List of k source-table strings
        provider / model: LLM configuration; temperature/enable_thinking use llm_infer defaults
        max_tokens: Maximum output tokens
        max_retries: Maximum retries

    Returns:
        (intersection_md, diff_md, log):
            - intersection_md: Markdown table of entities appearing at least twice
            - diff_md: Markdown table of entities appearing only once
            - log: Runtime log
    """
    messages = [
        {"role": "system", "content": LLM_SPLIT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_split_user_prompt(task, tables)},
    ]

    content = ""
    reasoning_content = ""
    raw_response = {}

    for cur_try in range(max_retries):
        try:
            raw_response = llm_infer(
                provider=provider,
                model=model,
                messages=messages,
                tools=None,
                generation_config={"max_tokens": max_tokens},
            )
        except Exception as e:
            print(f"[LLMSplit] LLM call failed (attempt {cur_try+1}/{max_retries}): {e}")
            continue

        reasoning_content = raw_response.get("reasoning_content", "") or ""
        content = raw_response.get("content", "") or ""

        token_usage = raw_response.get("token_usage", {})
        if token_usage.get("completion_tokens", 0) >= max_tokens:
            print(f"[LLMSplit] Output tokens maxed out, retry {cur_try+1}/{max_retries}")
            continue

        if content.strip():
            break

    messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning_content})

    intersection_md, diff_md = _parse_split_output(content)
    print(f"[LLMSplit] intersection parsed: {bool(intersection_md)}, diff parsed: {bool(diff_md)}")

    log = {
        "messages": messages,
        "raw_response": raw_response,
        "intersection_md": intersection_md,
        "diff_md": diff_md,
    }
    return intersection_md, diff_md, log


def _merge_two_markdown_tables(task: str, table_a: str, table_b: str, provider: str, model: str,
                                max_tokens: int = 32768) -> str:
    """
    Use an LLM to union two Markdown tables without duplicates.
    Used to merge intersection and verified_diff.
    """
    if not table_a.strip() and not table_b.strip():
        return ""
    if not table_a.strip():
        return table_b
    if not table_b.strip():
        return table_a

    from .llm_merge_table import llm_merge_tables
    return llm_merge_tables(
        task=task,
        tables=[table_a, table_b],
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        return_log=False,
        merge_strategy="union",
    )


def merge_tables_with_verification(
    task: str,
    tables: list[str],
    provider: str,
    model: str,
    return_log: bool = False,
    # Search-verifier configuration
    sub_agent_env_config: dict | None = None,
) -> str | tuple[str, dict]:
    """
    Merge flow: LLM split + Search Verifier validation.

    Flow:
      1. LLM Split: ask the LLM to analyze all source tables and output:
         - intersection: entities appearing in at least two sources
         - diff: entities appearing in exactly one source
      2. Search Verifier: validate diff through real searches and return verified_md
      3. Final result = intersection ∪ verified_md through an LLM union merge

    verbs/runtime.run_with_prompt launches the search-verifier agent. No separate max_steps
    argument is passed because ReactRuntimeContext determines the global step budget.

    Args:
        task: Original task description
        tables: List containing k table strings
        provider / model: LLM configuration; temperature/enable_thinking use llm_infer defaults
        return_log: Whether to return a detailed log
        sub_agent_env_config: Tool-environment configuration for the verifier agent

    Returns:
        If return_log=False: str (Markdown table)
        If return_log=True:  (str, dict)
    """
    log: dict = {"strategy": "split_verify_merge", "steps": []}

    if not tables:
        result = ""
        if return_log:
            log["skipped"] = "no_tables"
            return result, log
        return result

    if len(tables) == 1:
        result = tables[0]
        if return_log:
            log["skipped"] = "single_table"
            return result, log
        return result

    # Step 1: LLM Split.
    print(f"[SplitVerifyMerge] Step 1: LLM split {len(tables)} tables...")
    intersection_md, diff_md, split_log = llm_split_tables(
        task=task,
        tables=tables,
        provider=provider,
        model=model,
    )
    log["steps"].append({
        "step": "llm_split",
        "has_intersection": bool(intersection_md),
        "has_diff": bool(diff_md),
        "split_log": split_log,
    })
    print(f"[SplitVerifyMerge] intersection={'yes' if intersection_md else 'empty'}, "
          f"diff={'yes' if diff_md else 'empty'}")

    # Step 2: have Search Verifier validate the diff.
    verified_md = ""
    if diff_md and diff_md.strip():
        print(f"[SplitVerifyMerge] Step 2: Search verifier for diff entities...")
        verified_md, verifier_log = run_search_verifier(
            task=task,
            diff_md=diff_md,
            sub_agent_env_config=sub_agent_env_config or {},
        )
        log["steps"].append(verifier_log)
    else:
        log["steps"].append({"step": "search_verifier", "skipped": "diff_empty"})
        print("[SplitVerifyMerge] Step 2: diff is empty, skipping verifier")

    # Step 3: merge intersection and verified_diff.
    print(f"[SplitVerifyMerge] Step 3: Merging intersection + verified_diff...")
    result = _merge_two_markdown_tables(
        task=task,
        table_a=intersection_md,
        table_b=verified_md,
        provider=provider,
        model=model,
    )
    log["steps"].append({
        "step": "final_merge",
        "has_intersection": bool(intersection_md),
        "has_verified": bool(verified_md),
        "result_length": len(result),
    })
    print(f"[SplitVerifyMerge] Final result length: {len(result)}")

    if return_log:
        return result, log
    return result
