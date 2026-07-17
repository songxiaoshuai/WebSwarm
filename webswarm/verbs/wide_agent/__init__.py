"""Entry point for the wide verb agent.

wide performs fanout decomposition: it splits homogeneous data-collection work into a batch
of self-contained subtasks and delegates them concurrently to the same verb-agent type. When
necessary, it recursively creates child wide agents for multidimensional expansion.
"""

VERB_NAME = "wide"


def run(
    task: str,
    env_config: dict,
    model: str,
    provider: str,
    max_steps: int,
    depth: int = 0,
    max_parallel_workers: int | None = None,
    *,
    guidance_store=None,
    root_task: str | None = None,
    subtask_id: str = "wide",
    web_probing_enabled: bool = False,
    subtask_experience_enabled: bool = False,
    **_unused,
) -> dict:
    """Build a WideNode and execute one wide subtask.

    Key arguments used only by wide and not accepted by other verbs:
        depth                : Current wide-node depth; defaults to 0 when called from root
        max_parallel_workers : Optional concurrent-subtask limit; defaults to WideNode's value
    """
    # Delay the import to avoid a cycle when registry imports the verb package.
    from .wide_node import WideNode

    node_kwargs = {
        "model": model,
        "provider": provider,
        "max_steps": max_steps,
        "env_config": env_config,
        "depth": depth,
        "guidance_store": guidance_store,
        "root_task": root_task,
        "web_probing_enabled": web_probing_enabled,
        "subtask_experience_enabled": subtask_experience_enabled,
    }
    if max_parallel_workers is not None:
        node_kwargs["max_parallel_workers"] = max_parallel_workers

    node = WideNode(**node_kwargs)
    return node.run(task_info={"task": task})
