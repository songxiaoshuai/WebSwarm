"""Fanout-decomposition node for the wide verb.

WideNode receives a task with an iteration dimension, delegates homogeneous subtasks in
concurrent batches through create_sub_agents, and recursively creates child wide nodes for
multidimensional expansion when needed. The node itself only decomposes, aggregates, and submits.
"""

import json
from copy import deepcopy
from typing import Optional

from llm_infer.llm_infer import llm_infer
from tool_env.tool_env import ToolEnv

from ...guidance import GuidanceEventStore
from ..registry import VALID_VERBS
from .prompts import build_wide_system_prompt
from .dispatch import DispatchMixin
from .experience import ExperienceMixin


DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_PARALLEL_WORKERS = 10


# Batch-delegation tool schema exposed to the LLM by the wide agent.
_create_sub_agents_tool_info = {
    "type": "function",
    "function": {
        "name": "create_sub_agents",
        "description": (
            "Decompose the current wide task into a batch of self-contained sub-tasks "
            "and dispatch them concurrently to verb agents. All sub-tasks in the same "
            "call use the same verb `type`. Returns aggregated text of all sub-task "
            "answers. After this call, you may issue another `create_sub_agents` call "
            "(e.g., to follow up after a discovery batch) or `submit_answer` to finalize."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of sub-task descriptions. Each must be fully self-contained "
                        "(include time scope, entity names, exclusions, expected output). "
                        "Sub-agents have no access to your prior reasoning."
                    ),
                },
                "type": {
                    "type": "string",
                    "enum": list(VALID_VERBS),
                    "description": (
                        "Verb type for ALL sub-tasks in this batch. "
                        "atom: single-entity attribute lookup or short multi-hop chain. "
                        "deep: constraint-intersection reasoning (target entity unknown). "
                        "wide: further fan-out (will recurse into a child WideNode). "
                        "entity_collect: enumerate a complete entity set with high P/R."
                    ),
                },
            },
            "required": ["tasks", "type"],
        },
    },
}


class WideNode(DispatchMixin, ExperienceMixin):
    """Main-loop node for the wide verb.

    Responsibilities:
      - Receive a wide task for fanout or nested-table collection.
      - Maintain its own multi-turn LLM loop, choosing create_sub_agents or submit_answer each turn.
      - Execute subtasks concurrently through ThreadPoolExecutor; the LLM selects their type.
      - Safeguard: nested wide cannot exceed max_depth and is forcibly downgraded to atom beyond it.

    Method organization:
      - Dispatch and concurrency → dispatch.py (DispatchMixin)
      - Experience mechanisms → experience.py (ExperienceMixin)
      - Core loop in this file: __init__, reset, step, run
    """

    def __init__(
        self,
        model: str,
        provider: str,
        max_steps: int,
        env_config: dict,
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_parallel_workers: int = DEFAULT_MAX_PARALLEL_WORKERS,
        guidance_store: Optional[GuidanceEventStore] = None,
        root_task: Optional[str] = None,
        web_probing_enabled: bool = False,
        subtask_experience_enabled: bool = False,
    ):
        # LLM and loop budget.
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.env_config = env_config
        self.max_parallel_workers = max_parallel_workers

        # Wide-recursion safeguard.
        self.depth = depth
        self.max_depth = max_depth

        # Guidance event stream and switches are forwarded from the root.
        self.guidance_store = guidance_store
        self.root_task = root_task
        self.web_probing_enabled = web_probing_enabled
        self.subtask_experience_enabled = subtask_experience_enabled

        # Temporarily store scout-derived subtask_experience during fanout.
        self._current_subtask_experience: Optional[str] = None

        # Include depth state in the prompt so the LLM can account for the recursion safeguard.
        self.system_prompt = build_wide_system_prompt(
            depth=depth, max_depth=max_depth,
            web_probing_enabled=web_probing_enabled,
        )
        self._build_tools()

        # reset() clears the following per-run state.
        self.messages: list[dict] = []
        self.current_observation = None
        self.current_info: dict = {}
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.trajectory: list[dict] = []
        self.all_child_results: list[dict] = []

    def _build_tools(self):
        """Build the tool list visible to the wide agent."""
        tmp_env = ToolEnv(config=deepcopy(self.env_config))
        env_tools = {t["function"]["name"]: t for t in tmp_env.get_tools_info()}
        self.tools = [_create_sub_agents_tool_info]
        if "submit_answer" in env_tools:
            self.tools.append(env_tools["submit_answer"])

    def reset(self, task_info: dict):
        """Initialize one wide-subtask run."""
        assert "task" in task_info, "task_info missing 'task'"
        self.task_info = deepcopy(task_info)
        task_observation = task_info["task"]

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_observation},
        ]
        self.current_observation = task_observation
        self.current_info = deepcopy(task_info)
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.trajectory = [{"step": 0, "observation": task_observation}]
        self.all_child_results = []
        # Forward original_task to entity_collect to restore field constraints omitted by higher levels.
        self._original_task: str = task_info.get("original_task") or task_observation

    def step(self):
        """Execute one create_sub_agents/submit_answer decision by the wide agent."""
        if self.terminated or self.truncated:
            raise RuntimeError("WideNode already finished; call reset first.")

        # 1. Based on history, wide chooses the next subtask batch or submits an answer.
        raw_response = llm_infer(
            provider=self.provider,
            model=self.model,
            messages=self.messages,
            tools=self.tools,
        )

        # 2. wide executes one tool call per turn; create_sub_agents expresses batches through tasks.
        message: dict = {"role": "assistant", "content": raw_response["content"]}
        if raw_response["tool_calls"]:
            if len(raw_response["tool_calls"]) > 1:
                print(f"[Wide-d{self.depth}] Warning: multiple tool calls; keeping the first.")
            raw_response["tool_calls"] = [raw_response["tool_calls"][0]]
            message["tool_calls"] = [raw_response["tool_calls"][0]]
        if raw_response.get("reasoning_content"):
            message["reasoning_content"] = raw_response["reasoning_content"]
        self.messages.append(message)

        # 3. Route to batch delegation or submit_answer.
        observation, reward, terminated, truncated, info = None, 0.0, False, False, {}

        if not raw_response["tool_calls"] and not raw_response["content"]:
            observation = "action is empty, please check the model"
            reward, terminated, truncated = 0, True, True
            info = {"action": "", "role": "user"}

        elif not raw_response["tool_calls"]:
            observation = (
                f"No tools called. You must call one of: "
                f"{[t['function']['name'] for t in self.tools]}"
            )
            info = {"action": raw_response["content"], "role": "user"}

        else:
            tool_call = raw_response["tool_calls"][0]
            tool_name = tool_call["function"]["name"]

            try:
                tool_params = tool_call["function"]["arguments"]
                if isinstance(tool_params, str):
                    tool_params = json.loads(tool_params)
            except json.JSONDecodeError as e:
                observation = f"Failed to parse tool parameters: {e}"
                info = {"action": f"{tool_name}(<parse error>)", "role": "tool"}
                self._append_tool_observation(raw_response, tool_name, observation)
                self.step_count += 1
                self.trajectory.append({
                    "step": self.step_count, "action": raw_response,
                    "observation": observation, "reward": 0,
                    "terminated": False, "truncated": False,
                })
                return observation, 0, False, False, info, info["action"]

            # A: Delegate homogeneous subtasks concurrently in a batch.
            if tool_name == "create_sub_agents":
                tasks = tool_params.get("tasks", [])
                verb = tool_params.get("type", "")
                observation, child_results = self._execute_create_sub_agents(
                    tasks=tasks, verb=verb
                )
                self.all_child_results.extend(child_results)
                info = {
                    "action": (
                        f"create_sub_agents(type={verb!r}, "
                        f"tasks={json.dumps(tasks, ensure_ascii=False)})"
                    ),
                    "role": "tool",
                }

            # B: submit_answer terminates the current wide node.
            elif tool_name == "submit_answer":
                tmp_env = ToolEnv(config=deepcopy(self.env_config))
                observation, reward, terminated, truncated, info = tmp_env.step(
                    action=raw_response
                )
                info["role"] = "tool"

            else:
                observation = (
                    f"Unknown tool: {tool_name}. "
                    f"Available: {[t['function']['name'] for t in self.tools]}"
                )
                info = {"action": f"{tool_name}(?)", "role": "tool"}

        # 4. Near the step limit, remind wide to synthesize subtask results.
        if not terminated and not truncated and self.step_count == self.max_steps - 2:
            observation += (
                "\nSystem warning: You have reached the maximum interaction rounds. "
                "Please synthesize current sub-task results and call `submit_answer` now."
            )

        # 5. Update wide-node runtime state.
        action = info.get("action", "")
        self.step_count += 1
        self.current_observation = observation
        self.current_info = info
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        # 6. Add the aggregated observation to LLM history.
        role = info.get("role", "user")
        if role == "tool" and raw_response.get("tool_calls"):
            self._append_tool_observation(
                raw_response,
                raw_response["tool_calls"][0]["function"]["name"],
                observation,
            )
        else:
            self.messages.append({"role": "user", "content": observation})

        # 7. Record the wide main-loop trajectory; subtask trajectories remain in all_child_results.
        self.trajectory.append({
            "step": self.step_count,
            "action": raw_response,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        return observation, reward, terminated, truncated, info, action

    def run(self, task_info: dict) -> dict:
        """Run a complete wide subtask and return a result embeddable in the log tree."""
        self.reset(task_info=task_info)

        # Run web_probing only on first-level wide nodes delegated directly by the root.
        if (
            self.web_probing_enabled
            and self.depth == 0
        ):
            self._maybe_run_web_probing()

        while not self.terminated and not self.truncated and self.step_count < self.max_steps:
            self.step()

        prediction_answer = self.current_info.get("prediction_answer", None)

        return {
            "task_info":          self.task_info,
            "prediction_answer":  prediction_answer,
            "tools":              self.tools,
            "messages":           self.messages,
            "trajectory":         self.trajectory,
            "total_reward":       0.0,
            "terminated":         self.terminated,
            "truncated":          self.truncated,
            "final_observation":  self.current_observation,
            "final_info":         self.current_info,
            "steps":              self.step_count,
            "all_child_results":  self.all_child_results,
            "depth":              self.depth,
            "max_depth":          self.max_depth,
        }

    def _append_tool_observation(self, raw_response: dict, tool_name: str, observation: str):
        tool_call_id = raw_response["tool_calls"][0]["id"]
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": observation,
        })
