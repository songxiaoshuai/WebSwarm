"""Package entry point for the entity_collect verb agent.

entity_collect enumerates entity sets. Its goal is to retrieve a complete entity set rather
than collect attributes one known dimension at a time like wide. At runtime it samples
multiple strategies in parallel and combines candidate tables through split-verify-merge.

This file exposes only ``VERB_NAME`` and ``run()`` for the registry. ``EntityCollectNode``
implements the four-stage flow.
"""

from .entity_collect_node import EntityCollectNode, VERB_NAME  # noqa: F401

__all__ = ["VERB_NAME", "run", "EntityCollectNode"]


def run(
    task: str,
    env_config: dict,
    model: str,
    provider: str,
    *,
    original_task: str | None = None,
    **_unused,
) -> dict:
    """Execute one entity_collect subtask.

    The registry routes tasks delegated by root/wide here. This function is a thin wrapper
    that builds ``EntityCollectNode`` and lets the node perform schema injection,
    multi-strategy sampling, split-verify-merge, and log assembly.

    ``**_unused`` accepts generic registry arguments. ``ReactRuntimeContext`` supplies
    the step budget for entity_collect sampling agents.
    """
    node = EntityCollectNode(
        task=task,
        env_config=env_config,
        model=model,
        provider=provider,
        original_task=original_task,
    )
    return node.run()
