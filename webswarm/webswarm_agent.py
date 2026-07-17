"""Top-level WebSwarm execution entry point.

WebSwarmAgent connects the task_manager, ToolEnv, and runtime-configuration dict supplied by
the runner, creates the root agent, and invokes the benchmark evaluator to compute a reward
after the root finishes. It does not perform search or page reading directly; the root delegates
work to verb agents such as atom, deep, wide, and entity_collect.
"""

from __future__ import annotations

from .guidance import GuidanceEventStore
from .root_node import RootNode


class WebSwarmAgent:
    """WebSwarm runner for one task."""

    def __init__(
        self,
        tool_env,
        task_manager,
        webswarm_config: dict,
        **_unused,
    ):
        if _unused:
            raise TypeError(
                f"Unexpected WebSwarmAgent argument(s): {sorted(_unused)}"
            )
        self.task_manager = task_manager
        self.tool_env = tool_env

        if not isinstance(webswarm_config, dict):
            raise TypeError(
                "webswarm_config must be a dict built by "
                "experiment_config.build_webswarm_config."
            )
        cfg = webswarm_config
        guidance_enabled = cfg["web_probing"] or cfg["subtask_experience"]

        env_config = {
            "enable_tools": tool_env.enable_tools,
            **tool_env.tool_config_map,
        }

        # Prepare shared context for leaf-agent runners: model, tool configuration, and step budget.
        from .verbs.runtime import ReactRuntimeContext, set_runtime_context
        set_runtime_context(ReactRuntimeContext(
            env_config=env_config,
            model=cfg["model"],
            provider=cfg["provider"],
            default_max_steps=cfg["max_steps"],
        ))

        # GuidanceEventStore records audit events for web_probing and subtask_experience.
        self._guidance_store = GuidanceEventStore() if guidance_enabled else None

        self.root_node = RootNode(
            model=cfg["model"],
            provider=cfg["provider"],
            max_steps=cfg["max_steps"],
            env_config=env_config,
            guidance_store=self._guidance_store,
            web_probing_enabled=cfg["web_probing"],
            subtask_experience_enabled=cfg["subtask_experience"],
            prompt_version=cfg["prompt_version"],
        )

        self._webswarm_config_dict = dict(webswarm_config)

    def run(self) -> dict:
        """Execute task_manager's current task and return a WebSwarm log containing the reward."""
        task_info = self.task_manager.get_task_info()

        print(f"[WebSwarm] Starting task: {task_info.get('task', '')[:120]}")
        cfg = self._webswarm_config_dict
        print(f"[WebSwarm] Config: max_steps={cfg['max_steps']}")

        result = self.root_node.run(task_info=task_info)

        prediction_answer = result.get("prediction_answer")
        total_reward, reward_info = None, {}
        if prediction_answer is not None:
            total_reward, reward_info = self.task_manager.calculate_reward(prediction_answer)

        result["total_reward"] = total_reward
        result["reward_info"] = reward_info

        sub_results = result.get("subtask_results", [])
        verbs_used = [r.get("verb") for r in sub_results]
        print(f"[WebSwarm] Done. subtasks={len(sub_results)} verbs={verbs_used}")
        if self._guidance_store is not None:
            print(f"[WebSwarm] guidance stats: {self._guidance_store.stats()}")

        return result

    def __repr__(self) -> str:
        return (
            f"WebSwarmAgent("
            f"max_steps={self._webswarm_config_dict['max_steps']}, "
            f"model={self._webswarm_config_dict['model']!r})"
        )
