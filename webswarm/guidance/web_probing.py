"""web_probing: probe information topology before wide decomposition.

Before a root-level wide agent begins fanout, web_probing runs a lightweight leaf agent
to determine whether the target information resembles a centralized hub, a hub with gaps,
or content distributed across multiple pages.

The raw submit_answer text from web_probing is injected directly into the WideNode LLM user
message as a "# Topology Finding" section to inform later decomposition dimensions and
granularity. web_probing itself neither makes decomposition decisions nor performs structured
parsing in Python.

Topology classification has three values: centralized / centralized_with_gaps / distributed.
The middle state represents a hub that covers most, but not all, information.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .event_store import _now_str

# System prompt for the web_probing leaf agent.

_WEB_PROBING_SYSTEM_PROMPT = """\
You are an **information topology scout**. Before a research task is decomposed \
into parallel sub-tasks, you do a quick search to determine how the required \
information is distributed across web pages.

Current Date: {current_date}

## Your Goal

Determine the **information topology** for a research task:
1. **centralized** — 1-3 hub pages cover ALL required data (rows AND columns).
2. **centralized_with_gaps** — hub pages cover a large portion of the \
required data, but have identifiable gaps (missing columns, incomplete \
rows, missing details, etc.).
3. **distributed** — no hub covers even the majority of the data; \
information is scattered across many independent pages.

## Workflow

1. Read the task. Identify the key entities (rows) and attributes (columns) needed.
2. Search for aggregate / hub sources — **prioritize Wikipedia**:
   - Wikipedia "List of ..." pages are often the best structured hub sources. \
Try queries like `<topic> site:en.wikipedia.org` or `List of <entities>` first.
   - Also try: "[topic] complete list", "[topic] table / database / all"
3. Fetch promising pages and **check coverage column by column**:
   - For each required column, confirm whether the page actually contains it \
(not just whether a table exists).
   - Check row scope: does the page cover the full range the task asks for \
(time range, geographic scope, etc.)?
   - A page having a table does NOT mean it covers everything — verify.
(Repeat steps 2 and 3 until a clear conclusion is reached.)
4. Make your topology judgment and submit.

## Rules

- **Budget**: at most {max_steps} tool calls.
- Do NOT answer the research task. Only assess information topology.
- Report ONLY structural observations (page count, page organization, \
coverage scope).
- Do NOT report content-level observations (specific dates, prices, \
availability, whether data is "current" or "outdated"). These bias \
downstream agents.
- Do NOT give decomposition advice, strategy suggestions, or warnings.
- **Avoid premature conclusions**: Do not judge "centralized" after only \
1-2 searches. If a hub page looks promising, try at least one more \
independent query to cross-check whether the hub truly covers everything \
before committing.
- When in doubt between centralized and centralized_with_gaps, prefer \
centralized_with_gaps — it is safer to flag potential gaps than to \
over-promise hub completeness.

## Output Format — pick ONE of the three topologies:

### If data is fully centralized (1-3 pages cover ALL rows AND columns):

```
## Topology
centralized

## Hub Sources
- <URL 1> — <what rows/columns this page covers>
- <URL 2> — <what rows/columns this page covers>

## Page Structure
<brief factual description: single table / multiple tables / paginated / etc.>
```

### If data is centralized with gaps (hub covers most data, but has identifiable gaps):

```
## Topology
centralized_with_gaps

## Hub Sources
- <URL 1> — <what this page covers>

## Hub Coverage
<what the hub page(s) DO provide: which rows, columns, or details are available>

## Gaps
<what is missing: specific columns not present, rows not covered, details absent, etc.>

## Estimated Coverage
<rough fraction of required data on hub, e.g. "4/5 columns", "~80% of rows">
```

### If data is distributed (no hub covers the majority of rows):

```
## Topology
distributed

## Organization Dimension
<the dimension along which pages are organized: e.g., "tour name", "brand", \
"program", "year", "country">

## Evidence
<factual observation about page structure that supports this dimension>

## Key Source Pattern
- <representative URL pattern or hub page> — <what it links to>
```
"""


_WEB_PROBING_USER_HINT = (
    "Determine the information topology for this task: "
    "is the data fully centralized, centralized with gaps, or distributed? "
    "If centralized with gaps, describe what the hub covers and what is missing. "
    "If distributed, identify the organization dimension. "
    "Do NOT answer the task or give decomposition advice."
)


def run_web_probing(
    *,
    wide_task: str,
    model: str,
    provider: str,
) -> dict | None:
    """Run the web_probing leaf agent.

    Like atom and other leaves, it uses ctx.default_max_steps instead of a separate setting.

    Args:
        wide_task: WideNode task text.
        model / provider: Model/provider used by web_probing, inherited from the main agent.

    Returns:
        raw_agent_result: Complete agent result containing prediction_answer / messages /
        trajectory and related fields, or None when agent execution fails.

        prediction_answer is Markdown text with sections such as ## Topology / ## Hub Sources.
        The caller appends it directly to the wide-agent user message without structured
        parsing in Python.
    """
    from ..verbs.runtime import run_with_prompt, get_runtime_context

    ctx = get_runtime_context()
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt = _WEB_PROBING_SYSTEM_PROMPT.format(
        current_date=current_date,
        max_steps=ctx.default_max_steps,
    )
    task_text = _build_web_probing_task(wide_task=wide_task)

    try:
        return run_with_prompt(
            system_prompt=system_prompt,
            task=task_text,
            model=model,
            provider=provider,
        )
    except Exception as e:
        print(f"[web_probing] agent execution failed: {e}")
        return None


def _build_web_probing_task(*, wide_task: str) -> str:
    """Build the user message for the web_probing agent."""
    return (
        "## RESEARCH TASK\n"
        f"{wide_task.strip()}\n\n"
        f"{_WEB_PROBING_USER_HINT}"
    )


def build_web_probing_event(
    *,
    wide_task: str,
    raw_web_probing_agent: dict | None = None,
    error: str | None = None,
) -> dict:
    """Build one web_probing audit event.

    The event no longer flattens structured topology fields. For later topology analysis,
    read the Markdown text in `event["web_probing_agent"]["prediction_answer"]`.
    """
    ev: dict[str, Any] = {
        "kind": "web_probing",
        "wide_task": wide_task[:300],
        "ts": _now_str(),
    }
    if error:
        ev["error"] = str(error)[:500]
    if raw_web_probing_agent is not None:
        ev["web_probing_agent"] = {
            "prediction_answer": raw_web_probing_agent.get("prediction_answer"),
            "steps": raw_web_probing_agent.get("steps", 0),
            "terminated": raw_web_probing_agent.get("terminated", False),
            "messages": raw_web_probing_agent.get("messages", []),
            "trajectory": raw_web_probing_agent.get("trajectory", []),
            "tool_states": raw_web_probing_agent.get("tool_states", {}),
        }
    return ev
