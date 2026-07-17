"""subtask_experience: extract transferable experience from scout subtasks.

Before a large fanout, the wide agent can run a small number of scout subtasks. This module
reads their actual execution traces, extracts process-level retrieval experience such as
effective queries, reliable sites, or paths to avoid, and injects it into later sibling agents.

The mechanism affects only later fanout batches of the current wide node. Extraction failure
returns None, after which the caller continues with the default concurrent flow.
"""

from __future__ import annotations

import traceback

from llm_infer.llm_infer import llm_infer

from .extract import _render_trace
from .event_store import GuidanceEventStore, _now_str


# System prompt for extracting subtask_experience.

EXPERIENCE_SYSTEM_PROMPT = """\
You are a skill-extraction assistant for a multi-agent research system.

CONTEXT
-------
A "wide" task has been decomposed into a batch of sibling sub-tasks (the
FANOUT batch). The system first ran a small SCOUT batch
(the first N sub-tasks) and recorded their full execution traces. Now,
**before** dispatching the remaining sub-tasks, you extract transferable
tactical knowledge ("skill") from the scout traces to help the remaining
sub-agents work more efficiently and more effectively.

YOU WILL BE GIVEN
-----------------
  - ROOT_TASK        : the original user-level question
  - WIDE_TASK        : the parent wide task that produced this batch
  - SCOUT_TASKS      : the sub-tasks that already completed (with answers)
  - REMAINING_TASKS  : the sub-tasks that have NOT yet been dispatched
  - SCOUT_TRACES     : full message traces of the scout sub-agents
                       (system stripped, reasoning_content stripped,
                       long contents middle-truncated)

YOUR JOB
--------
Analyze the scout execution traces and extract transferable process
skill such as:
  - which queries / domains worked well / poorly
  - which URLs were authoritative for this kind of question
  - which dead ends to skip
  - any constraint the scout discovered that affects all siblings

Do NOT include facts that are specific to a single scout sub-task's
answer. The skill is process-level, not knowledge-level.

HARD RULES
----------
  1. Skill text must be **process advice**, not factual claims. It will
     be prepended to each sub-agent's task brief as if a sibling were
     whispering "here's what worked for me".
  2. Output ONLY the skill text itself — no JSON, no code fences, no
     preamble, no explanation. Just the plain-text skill paragraph.
  3. Keep it concise: <= 150 words, plain English, no markdown headers.
  4. If no useful tactical skill is observable from the scout traces,
     output exactly the single word: NONE
"""


# Rendering limit for one scout trace. Reuse the middle truncation in extract._render_trace by
# passing messages through for it to handle (_render_trace already applies _MAX_MSG_CHARS and
# _MAX_TRACE_CHARS). This module applies one more aggregate limit after joining multiple scout
# traces so N scouts cannot overflow the prompt.
_MAX_TOTAL_TRACES_CHARS = 300000

# Rendering limit for remaining_tasks, with a line-count fallback.
_MAX_REMAINING_TO_SHOW = 60


def extract_subtask_experience(
    *,
    root_task: str,
    wide_task: str,
    scout_results: list[dict],
    remaining_tasks: list[str],
    guidance_store: "GuidanceEventStore",
    model: str,
    provider: str,
) -> tuple[str | None, dict | None]:
    """Extract execution experience from the scout phase and return text plus an audit event.

    subtask_experience_str:
        Extracted process-level experience forwarded to the fanout batch.
        None indicates failure or no useful experience, so the caller should use the default fanout flow.

    event_dict is always appended to guidance_store.events, retaining an audit record even
    when subtask_experience is None so failed calls can be identified later.

    Args:
      scout_results: Dicts returned by WideNode._dispatch_one_subtask, each containing at
                     least task / answer / messages fields.
      remaining_tasks: Strings for subtasks not yet dispatched during fanout.

    This function writes no external knowledge base; it only appends one audit event with
    kind='subtask_experience'.
    """
    # Keep the default fanout when there are no scouts or no target tasks for injection.
    if not scout_results or not remaining_tasks:
        return None, None

    # root, wide, task, and trace together form the experience-extraction context.
    scout_tasks_text = _render_scout_tasks(scout_results)
    remaining_text = _render_remaining(remaining_tasks)
    traces_text = _render_all_scout_traces(scout_results)

    user_msg = (
        f"ROOT_TASK:\n{root_task}\n\n"
        f"WIDE_TASK:\n{wide_task}\n\n"
        f"SCOUT_TASKS (already completed):\n{scout_tasks_text}\n\n"
        f"REMAINING_TASKS (not yet dispatched, total={len(remaining_tasks)}):\n"
        f"{remaining_text}\n\n"
        f"SCOUT_TRACES:\n{traces_text}\n"
    )

    messages = [
        {"role": "system", "content": EXPERIENCE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # LLM failures must degrade gracefully and never block the wide agent's main task.
    try:
        resp = llm_infer(
            provider=provider,
            model=model,
            messages=messages,
        )
    except Exception as e:  # noqa: BLE001 — failure must degrade gracefully
        print(f"[guidance.subtask_experience] llm_infer failed: {e}")
        traceback.print_exc()
        err_messages = list(messages) + [
            {"role": "assistant", "content": f"<llm_infer exception: {e}>"}
        ]
        err_event = _make_event(
            root_task=root_task,
            wide_task=wide_task,
            remaining_tasks=remaining_tasks,
            scout_traces_rendered=traces_text,
            llm_input_messages=err_messages,
            subtask_experience=None,
            extra={"error": str(e)},
        )
        guidance_store.append_event(err_event)
        return None, err_event

    raw = (resp.get("content") if isinstance(resp, dict) else "") or ""
    subtask_experience = _parse_subtask_experience_response(raw)

    messages_with_resp = list(messages) + [
        {"role": "assistant", "content": raw}
    ]
    event = _make_event(
        root_task=root_task,
        wide_task=wide_task,
        remaining_tasks=remaining_tasks,
        scout_traces_rendered=traces_text,
        llm_input_messages=messages_with_resp,
        subtask_experience=subtask_experience,
    )
    guidance_store.append_event(event)

    return subtask_experience, event


def _render_scout_tasks(scout_results: list[dict]) -> str:
    """Render scout subtasks as a concise (task, answer) list."""
    lines: list[str] = []
    for i, r in enumerate(scout_results, 1):
        task = (r.get("task") or "").strip()
        answer = (r.get("answer") or "").strip()
        if len(answer) > 800:
            answer = answer[:800] + "…"
        lines.append(f"[scout {i}] task: {task}\n[scout {i}] answer: {answer}")
    return "\n\n".join(lines) if lines else "(no scout results)"


def _render_remaining(remaining_tasks: list[str]) -> str:
    shown = remaining_tasks[:_MAX_REMAINING_TO_SHOW]
    lines = [f"  {i}. {t}" for i, t in enumerate(shown, 1)]
    if len(remaining_tasks) > _MAX_REMAINING_TO_SHOW:
        lines.append(
            f"  ... (+{len(remaining_tasks) - _MAX_REMAINING_TO_SHOW} more "
            f"truncated for prompt budget)"
        )
    return "\n".join(lines) if lines else "(none)"


def _render_all_scout_traces(scout_results: list[dict]) -> str:
    """Render each scout trace, then apply an aggregate length safeguard."""
    blocks: list[str] = []
    for i, r in enumerate(scout_results, 1):
        msgs = r.get("messages") or []
        rendered = _render_trace(msgs)
        blocks.append(f"=== Scout {i} trace ===\n{rendered}")
    joined = "\n\n".join(blocks) if blocks else "(no traces)"
    if len(joined) > _MAX_TOTAL_TRACES_CHARS:
        joined = (
            f"<... {len(joined) - _MAX_TOTAL_TRACES_CHARS} chars truncated "
            f"from start ...>\n" + joined[-_MAX_TOTAL_TRACES_CHARS:]
        )
    return joined


def _make_event(
    *,
    root_task: str,
    wide_task: str,
    remaining_tasks: list[str],
    scout_traces_rendered: str,
    llm_input_messages: list[dict],
    subtask_experience: str | None,
    extra: dict | None = None,
) -> dict:
    """Build one audit event with kind='subtask_experience'."""
    ev: dict = {
        "kind": "subtask_experience",
        "root_task": root_task,
        "wide_task": wide_task,
        "remaining_tasks": list(remaining_tasks),
        "scout_traces_rendered": scout_traces_rendered,
        "llm_input_messages": llm_input_messages,
        "subtask_experience": subtask_experience,
        "ts": _now_str(),
    }
    if extra:
        ev.update(extra)
    return ev


def _parse_subtask_experience_response(raw: str) -> str | None:
    """Parse LLM output and extract subtask_experience text, returning nonempty text or None."""
    if not raw:
        return None
    text = raw.strip()
    if not text or text.upper() == "NONE":
        return None
    return text
