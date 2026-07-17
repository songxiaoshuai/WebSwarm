"""Search Verifier for entity_collect.

Search Verifier is the validation stage of split-verify-merge. For low-confidence entities
that appear in only one sampling path, it reruns a leaf agent with search/fetch_url and retains
candidate rows supported by web evidence.
"""
from __future__ import annotations

import json as _json
import traceback
from ..runtime import run_with_prompt


# System prompt for Search Verifier.
SEARCH_VERIFIER_SYSTEM_PROMPT = """\
You are a precise entity verification agent.
You will receive a task description and a list of candidate entities that need verification.
Your job is to search the web and determine which candidates are VALID entities that match the task criteria.

Core principle:
- Verify entities by evidence, not by rigid string matching.
- A candidate should be accepted if credible sources clearly support that the candidate refers to the same real-world entity/event, even when the exact search phrase does not appear verbatim.

Rules:
1. **Search for each candidate** — use web search to verify whether each candidate entity exists and matches the task criteria.
2. **Use flexible search strategies** — do NOT rely only on exact-match search strings. If an exact phrase returns no results, it should be treated as inconclusive, not as evidence that the entity does not exist. Try normalized variants, partial names, aliases, reordered tokens, and date/location combinations.
3. **Be conservative** — only include a candidate in your output if you find credible evidence confirming it matches the task. 
4. **Reject unverifiable candidates** — if you cannot find sufficient evidence for a candidate, do NOT include it.
5. **Preserve table format** — your output must be a Markdown table with the same columns as the input.
6. **Do NOT add new entities** — only output a subset of the candidates you received.
7. **Do NOT fabricate sources** — every included entity must be confirmed by real search results.

Workflow:
1. Read the task and understand the exact criteria.
2. For each candidate entity, search for it specifically.
3. Prefer official sources when available.
4. Include only the entities you can confirm with evidence.
5. Submit the verified entities as a Markdown table using `submit_answer`.
"""


def _build_verifier_task(task: str, diff_md: str) -> str:
    """Build the task description for the search verifier."""
    return (
        f"## Original Task\n{task}\n\n"
        f"## Candidate Entities to Verify\n"
        f"The following entities were found in only ONE source and need verification. "
        f"Search the web to confirm which ones are valid according to the task criteria.\n\n"
        f"{diff_md}\n\n"
        f"Output ONLY the verified entities as a Markdown table with the same columns. "
        f"If none can be verified, output an empty table with just the header row."
    )


def _extract_submit_answer(result: dict) -> str:
    """Extract submit_answer content from verifier-agent message history."""
    for msg in reversed(result.get("messages", [])):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            if tc["function"]["name"] == "submit_answer":
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except _json.JSONDecodeError:
                        return args
                return args.get("answer", "")
    return ""


def run_search_verifier(
    task: str,
    diff_md: str,
    *,
    sub_agent_env_config: dict,
) -> tuple[str, dict]:
    """
    Run a search-verifier agent to validate low-confidence rows that appear only once.

    verbs/runtime.run_with_prompt launches the search-verifier agent. ReactRuntimeContext
    supplies its global step budget, so no separate max_steps argument is passed.

    Args:
        task: Original task description
        diff_md: Markdown table containing entities that appear only once
        sub_agent_env_config: Child-agent tool-environment configuration

    Returns:
        (verified_md, log): validated Markdown table and runtime log
    """

    log: dict = {"step": "search_verifier"}

    if not diff_md or not diff_md.strip():
        log["skipped"] = "diff_empty"
        return "", log

    verifier_task = _build_verifier_task(task, diff_md)
    log["diff_md"] = diff_md

    try:
        result = run_with_prompt(
            system_prompt=SEARCH_VERIFIER_SYSTEM_PROMPT,
            task=verifier_task,
            env_config=sub_agent_env_config,
        )
        log["agent_result"] = {
            "terminated": result.get("terminated", False),
            "steps": result.get("steps", 0),
            "error": result.get("error"),
            "result": result,
        }

        # The verifier may return prediction_answer directly or leave it only in tool-call arguments.
        verified_md = result.get("prediction_answer", "")
        if not verified_md:
            verified_md = _extract_submit_answer(result)

        log["verified_md"] = verified_md
        print(f"[SearchVerifier] Got verified answer, length={len(verified_md)}")
        return verified_md, log

    except Exception as e:
        log["error"] = repr(e)
        log["traceback"] = traceback.format_exc()
        print(f"[SearchVerifier] Error: {e}")
        return "", log
