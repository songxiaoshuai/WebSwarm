"""WebSwarm root-agent node.

RootNode is WebSwarm's top-level control loop. It divides the original task into self-contained
subtasks and delegates them to specialized verb agents. It does not perform web searches,
page reads, or table filling itself; it maintains conversation history, tool-call trajectories,
and the final answer from the root perspective.

Execution flow:
    1. reset(task_info) writes the original task to root messages.
    2. Each step calls the LLM with the current messages.
    3. The LLM delegates one subtask through solve_subtask(task, verb), or submits the final
       answer through submit_answer(answer).
    4. solve_subtask calls run_verb and compresses the verb agent's complete result into a
       [Subtask Result] tool message written back to root messages.
    5. Based on existing subtask results, the root continues delegating, revises its plan,
       or submits an answer until submit_answer succeeds or max_steps is reached.

Each delegation runs the target verb agent with an independent ToolEnv copy, so subtasks do
not share submit_answer state. Complete child results remain in the log tree for reviewing
the multi-agent delegation process.
"""

import json
import uuid
from copy import deepcopy
from typing import Optional

from llm_infer.llm_infer import llm_infer
from tool_env.tool_env import ToolEnv

from .guidance import GuidanceEventStore
from .system_prompt import build_system_prompt
from .verbs import VALID_VERBS, run_verb


# ---------------------------------------------------------------------------- #
# solve_subtask is the root agent's only delegation tool.
# - task : a fully self-contained subtask description including all dates, scopes, and exclusions,
#          because verb agents cannot see the root context.
# - verb : an enum that determines which verb agent receives the task.
_solve_subtask_tool_info = {
    "type": "function",
    "function": {
        "name": "solve_subtask",
        "description": (
            "Delegate a single self-contained subtask to a specialized verb agent. "
            "Exactly one subtask per call. After this call returns, you will see the "
            "subtask's result and decide the next action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Self-contained subtask description. Include all relevant context "
                        "(dates, scope, named entities, exclusions, expected output format). "
                        "The verb agent has no access to the original user task or your prior reasoning."
                    ),
                },
                "verb": {
                    "type": "string",
                    "enum": list(VALID_VERBS),
                    "description": (
                        "The verb agent that should handle this subtask. "
                        "atom: single-entity / short-chain factual lookup. "
                        "deep: constraint-intersection reasoning where the target entity is unknown. "
                        "wide: fan-out collection with iteration dimensions and per-item attributes. "
                        "entity_collect: high-precision/high-recall enumeration of an entity set."
                    ),
                },
            },
            "required": ["task", "verb"],
        },
    },
}


class RootNode:
    """WebSwarm root delegation agent.

    Responsibilities:
      - This class handles only the outer loop: calling the LLM, routing tool calls, and maintaining history.
      - It produces prediction_answer through the submit_answer tool.
      - It does **not** perform search, parallel scheduling, or task decomposition; verb agents do that work.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        max_steps: int,
        env_config: dict,
        guidance_store: Optional[GuidanceEventStore] = None,
        web_probing_enabled: bool = False,
        subtask_experience_enabled: bool = False,
        prompt_version: str = "general",
    ):
        # LLM and loop budget for the root agent.
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.env_config = env_config
        self.prompt_version = prompt_version

        # guidance records audit events only; the root does not consume experience content directly.
        self.guidance_store = guidance_store
        self.web_probing_enabled = web_probing_enabled
        self.subtask_experience_enabled = subtask_experience_enabled
        self.root_task: Optional[str] = None  # Populated by reset()

        # The root prompt teaches the LLM to choose a verb and decide when to call submit_answer.
        self.system_prompt = build_system_prompt(version=prompt_version)
        self._build_tools()

        # The following fields hold state for one task run; reset() clears all of them.
        self.messages: list[dict] = []          # Complete conversation history passed to the LLM
        self.current_observation = None         # Most recent observation text, for debugging
        self.current_info: dict = {}            # Most recent step info; prediction_answer appears only for submit_answer
        self.terminated = False                 # Normal-termination flag set after successful submit_answer
        self.truncated = False                  # Truncation flag for exceeding max_steps or similar limits
        self.step_count = 0
        self.trajectory: list[dict] = []        # Turn-by-turn trajectory for later analysis
        self.subtask_results: list[dict] = []   # Raw result dicts from all verb agents

    # ------------------------------------------------------------------ #
    #                         Tool construction                            #
    # ------------------------------------------------------------------ #
    def _build_tools(self):
        """Build the tool list visible to the root agent.

        solve_subtask is the root's custom delegation entry point; submit_answer reuses the
        ToolEnv tool schema. ToolEnv is used here only to read schemas and holds no runtime state.
        """
        tmp_env = ToolEnv(config=deepcopy(self.env_config))
        env_tools = {t["function"]["name"]: t for t in tmp_env.get_tools_info()}
        self.tools = [_solve_subtask_tool_info]
        if "submit_answer" in env_tools:
            self.tools.append(env_tools["submit_answer"])

    # ------------------------------------------------------------------ #
    #                       solve_subtask execution                        #
    # ------------------------------------------------------------------ #
    def _execute_solve_subtask(self, task: str, verb: str) -> tuple[str, dict]:
        """Delegate one self-contained subtask to the selected verb agent.

        Returns two values:
          - observation: Human-readable [Subtask Result] text written to root messages as a
            tool message; the root LLM sees only this text on the next turn.
          - sub_result: Raw result dict containing complete messages / trajectory / tool_states,
            appended to self.subtask_results for later analysis.

        Error strategy: return the error to the root LLM as an observation instead of raising,
        allowing it to retry, switch verbs, or submit based on available information.
        """
        if verb not in VALID_VERBS:
            return (
                f"[Error] Unknown verb {verb!r}. Valid verbs: {VALID_VERBS}.",
                {"error": "unknown_verb", "verb": verb, "task": task},
            )
        if not isinstance(task, str) or not task.strip():
            return (
                f"[Error] 'task' must be a non-empty string. "
                f"You passed: task={task!r} (type={type(task).__name__}), verb={verb!r}.",
                {"error": "empty_task", "verb": verb, "task": task},
            )

        print(f"[Root] solve_subtask(verb={verb}) — task={task[:120]}")

        # subtask_id is used for guidance auditing and is not included in the child-agent prompt.
        subtask_id = f"root.{verb}#{uuid.uuid4().hex[:6]}"

        try:
            raw = run_verb(
                verb=verb,
                task=task,
                env_config=deepcopy(self.env_config),
                model=self.model,
                provider=self.provider,
                max_steps=self.max_steps,
                # Pass guidance switches downward; currently they are used mainly by the wide agent.
                guidance_store=self.guidance_store,
                root_task=self.root_task,
                subtask_id=subtask_id,
                web_probing_enabled=self.web_probing_enabled,
                subtask_experience_enabled=self.subtask_experience_enabled,
            )
        except Exception as e:
            err = f"[Error] verb agent {verb!r} raised: {e}"
            print(f"[Root] {err}")
            return err, {
                "error": str(e),
                "verb": verb,
                "task": task,
            }

        result = {
            "task": task,
            "verb": verb,
            "answer": raw.get("prediction_answer") or "",
            "status": "completed" if raw.get("terminated") else "truncated",
            "steps": raw.get("steps", 0),
            "messages": raw.get("messages", []),
            "trajectory": raw.get("trajectory", []),
            "tool_states": raw.get("tool_states", {}),
            # wide returns a nested subtree; forwarding child_results preserves the full delegation log.
            "child_results": raw.get("all_child_results", []),
        }

        observation = (
            f"[Subtask Result]\n"
            f"verb: {verb}\n"
            f"status: {result['status']} (steps: {result['steps']})\n"
            f"task: {task}\n"
            f"answer:\n{result['answer'] or '(empty)'}"
        )
        print(f"[Root] verb={verb} done — status={result['status']} steps={result['steps']}")
        return observation, result

    # ------------------------------------------------------------------ #
    #                              Reset                                   #
    # ------------------------------------------------------------------ #
    def reset(self, task_info: dict):
        """Initialize root-agent state from task_info."""
        assert "task" in task_info, "task_info missing 'task'"
        self.task_info = deepcopy(task_info)
        task_observation = task_info["task"]
        # root_task supports guidance auditing and experience extraction; it is not added to verb-agent prompts.
        self.root_task = task_observation

        # Start with the root prompt and original user task, then append tool interactions turn by turn.
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
        self.subtask_results = []

    # ------------------------------------------------------------------ #
    #                         Single-step execution                         #
    # ------------------------------------------------------------------ #
    def step(self):
        """Run one root-agent delegation/submission decision.

        Seven phases:
          1. Call the LLM.
          2. Append the complete assistant message, including tool_call, to messages.
          3. Route to solve_subtask / submit_answer / error handling.
          4. Near max_steps, append a hint to the observation so the LLM submits promptly.
          5. Update runtime state: step_count / terminated / truncated.
          6. Write the observation to messages with a tool or user role for the next LLM turn.
          7. Record the trajectory.
        """
        if self.terminated or self.truncated:
            raise RuntimeError("Node already finished; call reset first.")

        # 1. The root chooses solve_subtask or submit_answer from the complete history.
        raw_response = llm_infer(
            provider=self.provider,
            model=self.model,
            messages=self.messages,
            tools=self.tools,
        )

        # 2. Root semantics allow one action per turn; use wide/entity_collect for concurrent batches.
        message: dict = {"role": "assistant", "content": raw_response["content"]}
        if raw_response["tool_calls"]:
            if len(raw_response["tool_calls"]) > 1:
                print("[Root] Warning: multiple tool calls; keeping the first.")
            raw_response["tool_calls"] = [raw_response["tool_calls"][0]]
            message["tool_calls"] = [raw_response["tool_calls"][0]]
        if raw_response.get("reasoning_content"):
            message["reasoning_content"] = raw_response["reasoning_content"]
        self.messages.append(message)

        # 3. Route tool calls across empty responses, no-tool responses, valid tools, and JSON parse failures.
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

            # Branch A: solve_subtask → delegate to the selected verb agent.
            if tool_name == "solve_subtask":
                task = tool_params.get("task", "")
                verb = tool_params.get("verb", "")
                observation, sub_result = self._execute_solve_subtask(task=task, verb=verb)
                self.subtask_results.append(sub_result)
                info = {
                    "action": (
                        f"solve_subtask(verb={verb!r}, "
                        f"task={json.dumps(task, ensure_ascii=False)})"
                    ),
                    "role": "tool",
                }

            # Branch B: submit_answer → reuse ToolEnv's termination tool to write prediction_answer.
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

        # 4. Near the step limit, remind the root to synthesize existing child results.
        if not terminated and not truncated and self.step_count == self.max_steps - 2:
            observation += (
                "\nSystem warning: You have reached the maximum interaction rounds. "
                "Please synthesize current results and call `submit_answer` now."
            )

        # 5. Update root runtime state.
        action = info.get("action", "")
        self.step_count += 1
        self.current_observation = observation
        self.current_info = info
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        # 6. Add the observation to conversation history as context for the next root decision.
        role = info.get("role", "user")
        if role == "tool" and raw_response.get("tool_calls"):
            self._append_tool_observation(
                raw_response,
                raw_response["tool_calls"][0]["function"]["name"],
                observation,
            )
        else:
            self.messages.append({"role": "user", "content": observation})

        # 7. Record the root trajectory; child-agent trajectories remain in subtask_results.
        self.trajectory.append({
            "step": self.step_count,
            "action": raw_response,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        return observation, reward, terminated, truncated, info, action

    # ------------------------------------------------------------------ #
    #                           Complete run                                #
    # ------------------------------------------------------------------ #
    def run(self, task_info: dict) -> dict:
        """Execute the complete root delegation loop and return the task log.

        Exit on either condition:
          - terminated: submit_answer was called (normal completion)
          - truncated: step_count reached max_steps or the model returned an empty response
        """
        self.reset(task_info=task_info)
        while not self.terminated and not self.truncated and self.step_count < self.max_steps:
            self.step()

        # prediction_answer appears in current_info only after submit_answer succeeds.
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
            "subtask_results":    self.subtask_results,
            # None when guidance is disabled; otherwise export the complete audit-event stream.
            "guidance":           (
                self.guidance_store.export()
                if self.guidance_store is not None
                else None
            ),
        }

    # ------------------------------------------------------------------ #
    #                          Helper methods                               #
    # ------------------------------------------------------------------ #
    def _append_tool_observation(self, raw_response: dict, tool_name: str, observation: str):
        tool_call_id = raw_response["tool_calls"][0]["id"]
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": observation,
        })
