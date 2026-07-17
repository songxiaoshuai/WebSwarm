"""Shared runtime context and leaf runner for verb agents.

Multiple leaf agents share one execution mechanism: given a system prompt and task text,
create BaseReactAgent and run it with the model, tool environment, and step budget derived
from runtime configuration. This module centralizes those parameters so each verb does not
repeatedly create ToolEnv and BaseReactAgent.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from tool_env.tool_env import ToolEnv
from ..base_react_agent import BaseReactAgent


# Leaf-agent runtime context

@dataclass(frozen=True)
class ReactRuntimeContext:
    """Execution context required by run_with_prompt to create BaseReactAgent.

    Fields come from runtime configuration and ToolEnv and represent the minimal context
    shared by leaf agents. temperature / enable_thinking are not configured here; llm_infer
    model-layer defaults control them.
    """
    env_config: dict
    model: str
    provider: str
    default_max_steps: int


_RUNTIME: Optional[ReactRuntimeContext] = None


def set_runtime_context(ctx: ReactRuntimeContext) -> None:
    """Runtime context injected by WebSwarmAgent before constructing the root agent."""
    global _RUNTIME
    _RUNTIME = ctx


def get_runtime_context() -> ReactRuntimeContext:
    if _RUNTIME is None:
        raise RuntimeError(
            "ReactRuntimeContext not initialized. "
            "Did you forget to call set_runtime_context() in "
            "WebSwarmAgent.__init__?"
        )
    return _RUNTIME


def run_with_prompt(
    system_prompt: str,
    task: str,
    *,
    max_steps: Optional[int] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    env_config: Optional[dict] = None,
) -> dict:
    """Create and run one leaf agent.

    Args:
        system_prompt: System prompt defining the leaf-agent role.
        task: User task text.
        max_steps: Optional override; defaults to ctx.default_max_steps.
        model / provider: Optional overrides; default to ctx.model / ctx.provider.
        env_config: Optional override; defaults to ctx.env_config. Used only when an EC sampler
            or similar component must adjust tool_config dynamically per sample, such as keep_links.

    Return the raw BaseReactAgent.run() result dict, which can be embedded directly in the WebSwarm log tree.
    """
    ctx = get_runtime_context()

    # Callers may override the model for selected helper agents; otherwise inherit the WebSwarm primary model.
    if model and provider:
        eff_model, eff_provider = model, provider
    else:
        eff_model, eff_provider = ctx.model, ctx.provider

    # Root, search nodes, and helper leaf agents share one step budget by default.
    eff_max_steps = max_steps if max_steps is not None else ctx.default_max_steps

    # Different entity_collect sampling strategies may supply local env_config overrides.
    eff_env_config = env_config if env_config is not None else ctx.env_config

    tool_env = ToolEnv(config=deepcopy(eff_env_config))
    agent = BaseReactAgent(
        tool_env=tool_env,
        model=eff_model,
        provider=eff_provider,
        max_steps=eff_max_steps,
        system_prompt=system_prompt,
    )
    return agent.run(task_info={"task": task})
