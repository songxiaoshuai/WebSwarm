"""Sampling-path count decision module for entity_collect.

For entity-set enumeration, multi-strategy sampling offers limited value when the task already
defines closed boundaries and an entity count. For open sets, retain multiple sampling paths
to improve recall. This module returns only a sampling-strategy signal and does not alter
entity_collect merge logic.
"""
from __future__ import annotations

from llm_infer.llm_infer import llm_infer


_TASK_DIFFICULTY_SYSTEM_PROMPT = """\
You are a task difficulty classifier for entity enumeration tasks.

Your goal is to determine whether the entity set required by the task \
is already EXPLICITLY or IMPLICITLY CLOSED and COUNTABLE from the task itself.

## Definitions

**DETERMINISTIC task**:
- The task itself explicitly reveals a fixed, closed entity set.
- The expected number of entities can be given directly from the task itself.
- Examples: "List the 7 Harry Potter books", \
  "The 5 permanent members of the UN Security Council".

**OPEN-ENDED task**:
- The task does NOT reveal a precise entity count or closed boundary.
- The expected number of the entity set is not officially fixed or publicly well-known. You can not give the number of entities directly from the task itself.
- Examples: "All restaurants with Michelin stars in Paris 2024". \
  "All Nobel Prize winners in Physics since 1900".

# Important:
## Decision rule

Ask yourself:

"Can I know the exact entity count directly from the task wording alone,
without relying on world knowledge?"

## Output format

Reply with ONLY one of the following two tokens (no explanation, no punctuation):
  DETERMINISTIC
  OPEN-ENDED
"""

_TASK_DIFFICULTY_USER_TEMPLATE = """\
## Task
{task}
"""


def check_is_count_task(
    task: str,
    provider: str,
    model: str,
    max_retries: int = 3,
) -> tuple[bool, dict]:
    """Determine whether a task is simple because it specifies a fixed number of entities.

    Return True  → simple task; one sampling path is recommended.
    Return False → difficult/open-ended enumeration; retain multiple sampling paths.

    Use llm_infer defaults for temperature/enable_thinking. If the LLM call fails or its
    output cannot be parsed, conservatively return False to retain multi-path sampling.
    """
    if "[Reference Context — DO NOT EXPAND SCOPE]" in task:
        task = task.split("[Reference Context — DO NOT EXPAND SCOPE]")[0].strip()
    log: dict = {"original_task": task}
    messages = [
        {"role": "system", "content": _TASK_DIFFICULTY_SYSTEM_PROMPT},
        {"role": "user", "content": _TASK_DIFFICULTY_USER_TEMPLATE.format(task=task)},
    ]

    for attempt in range(max_retries):
        try:
            raw_response = llm_infer(
                provider=provider,
                model=model,
                messages=messages,
                tools=None,
                generation_config={"max_tokens": 32768},
            )
            log["llm_response"] = raw_response
            log["message"] = messages
            content = (raw_response.get("content", "") or "").strip().upper()
            if "DETERMINISTIC" in content.upper():
                print(f"[TaskDifficulty] Classified as DETERMINISTIC (single-sample mode)")
                return True, log
            if "OPEN-ENDED" in content.upper():
                print(f"[TaskDifficulty] Classified as OPEN-ENDED (multi-sample mode)")
                return False, log
            # Retry after parse failures so one abnormal output cannot alter the sampling strategy.
            print(f"[TaskDifficulty] Unexpected response (attempt {attempt+1}): {content!r}")
        except Exception as e:
            print(f"[TaskDifficulty] LLM call failed (attempt {attempt+1}/{max_retries}): {e}")

    # Conservative default: retain multi-path sampling.
    print("[TaskDifficulty] Falling back to OPEN-ENDED (multi-sample mode)")
    return False, log


if __name__ == "__main__":
    task = "Find the National Multifamily Housing Council (NMHC) top 50 property managers list/ranking for 2025. If the exact 2025 list isn't available yet, find the most recent available ranking (likely 2024). Extract the complete list including rank, company name, number of managed units for the previous year, and company founding year for all companies in the ranking."
    print(check_is_count_task(task, "openai", "glm-4.5"))
