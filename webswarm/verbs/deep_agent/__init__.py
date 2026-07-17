"""Entry point for the deep verb agent.

deep handles reasoning under deep constraints and independent verification. DeepNode forms
a propose-then-verify loop through call_search_agent / call_verify_agent / submit_answer,
suitable for tasks with unknown target entities or repeated verification needs.
"""

from __future__ import annotations

VERB_NAME = "deep"


def run(
    task: str,
    env_config: dict,
    model: str,
    provider: str,
    max_steps: int,
    *,
    root_task: str | None = None,
    **_unused,
) -> dict:
    """Build a DeepNode and execute one deep subtask."""
    from .deep_node import DeepNode

    node = DeepNode(
        model=model,
        provider=provider,
        max_steps=max_steps,
        env_config=env_config,
        root_task=root_task,
    )
    return node.run(task_info={"task": task})
