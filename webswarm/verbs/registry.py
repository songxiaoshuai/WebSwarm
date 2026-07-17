"""Verb-agent registry and unified dispatch entry point.

Both root and wide agents route natural-language subtasks to atom/deep/wide/entity_collect
through run_verb(). The registry performs only table-driven dispatch and does not participate
in search, verification, or merge logic.

Verb-module contract:
  - A module-level `VERB_NAME: str` constant exactly matching the LLM-selected string.
  - A module-level `run(task: str, **kwargs) -> dict` function.
  - The registry only dispatches; each verb consumes its required env_config / model /
    provider / max_steps / guidance arguments.
  - llm_infer model-layer defaults control temperature / enable_thinking.

Adding a verb agent:
  1. Create a `<verb>_agent/` package or `<verb>_agent.py` module in this package and
     export `VERB_NAME` and `run(...)`.
  2. Register it in `_VERB_MODULES` below.
  3. Describe the verb's scope and selection guidance in the root prompt.

Current entity_collect implementation:
  - Sampling stage: parallel multi-strategy sampling through run_with_prompt.
  - Merge stage: fixed LLM Split + Search Verifier validation flow.
"""

from . import atom_agent, deep_agent, wide_agent, entity_collect_agent

# Registered verb agents. Ordering affects only enum presentation in the tool schema.
_VERB_MODULES = [atom_agent, deep_agent, wide_agent, entity_collect_agent]

# Verb name → run function for that verb.
VERB_REGISTRY: dict = {m.VERB_NAME: m.run for m in _VERB_MODULES}
# Root/wide tool schemas use this list to constrain the verbs available to the LLM.
VALID_VERBS: list[str] = list(VERB_REGISTRY.keys())


def run_verb(verb: str, task: str, **kwargs) -> dict:
    """Dispatch to and execute the verb agent selected by name.

    This function is the sole call point between root/wide and verb agents.

    Args:
        verb: Must be one of VALID_VERBS.
        task: Self-contained natural-language subtask text.
        **kwargs: Execution arguments forwarded to the verb agent. Each verb consumes its
            required env_config / model / provider / max_steps / guidance arguments.
            llm_infer defaults control temperature / enable_thinking.

    Returns:
        Verb-agent result dict, equivalent to BaseReactAgent.run() output.
    """
    # The caller already validates LLM input; this assertion exposes only internal code inconsistencies.
    assert verb in VERB_REGISTRY, (
        f"Internal error: run_verb called with verb={verb!r}, "
        f"but caller should have validated against VALID_VERBS={VALID_VERBS}."
    )

    return VERB_REGISTRY[verb](task=task, **kwargs)
