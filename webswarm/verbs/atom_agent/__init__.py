"""Entry point for the atom verb agent.

atom handles attribute lookup for known entities, short-chain multi-hop tasks, and small-scale
fact verification. It reuses the general BaseReactAgent with an atom-specific system prompt.
"""

from .prompts import get_system_prompt
from ..runtime import run_with_prompt

VERB_NAME = "atom"


def run(
    task: str,
    *,
    guidance_store=None,
    root_task: str | None = None,
    subtask_id: str = "atom",
    **_unused,
) -> dict:
    """Run one atom leaf agent."""
    return run_with_prompt(
        system_prompt=get_system_prompt(),
        task=task,
    )
