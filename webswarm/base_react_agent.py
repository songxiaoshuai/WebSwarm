"""General ReAct leaf-agent runner.

Leaf nodes such as atom, deep search/verifier, and entity_collect sampling agents all
reuse this loop: the LLM generates a tool call, ToolEnv executes it, and the observation
is appended to conversation history until submit_answer or the step limit is reached.
"""

from copy import deepcopy
from llm_infer.llm_infer import llm_infer


class BaseReactAgent:
    """ReAct runner shared by all leaf-agent types."""

    def __init__(
        self,
        tool_env,
        model,
        provider,
        max_steps,
        system_prompt: str,
    ):
        # Each leaf agent has an independent ToolEnv to prevent submit_answer state leakage.
        self.tool_env = tool_env

        # The run configuration supplies LLM settings uniformly; leaf agents have no separate experiment switches.
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt

        # The root and search nodes share the same max_steps budget.
        self.max_steps = max_steps

        # reset() clears the following runtime state.
        self.messages = []              # Conversation history
        self.current_observation = None
        self.current_info = None
        self.total_reward = 0.0
        self.terminated = False         # Environment terminated normally
        self.truncated = False          # Environment truncated, for example after exceeding a limit
        self.step_count = 0
        self.trajectory = []            # Per-step execution details

    def reset(self, task_info: dict):
        """Initialize one leaf-agent run."""

        assert "task" in task_info, "task_info is missing the 'task' field"
        self.task_info = deepcopy(task_info)
        task_observation = task_info["task"]

        # Tool schemas come from the current ToolEnv, keeping search/fetch/submit_answer consistent.
        self.tool_env.reset()
        self.tools = self.tool_env.get_tools_info()

        # fetch_url page summaries need the original question to focus the summary.
        if self.tool_env.fetch_url_tool is not None:
            self.tool_env.fetch_url_tool.update_original_question(task_observation)

        # The system prompt defines the leaf agent's role, such as atom, search, or verifier.
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = task_observation
        self.current_info = deepcopy(task_info)
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.trajectory = []

        # Start with the task as a user message, then append assistant/tool turns or user feedback.
        self.messages += [{"role": "user", "content": task_observation}]

        # trajectory is the basic unit of the runtime log tree.
        self.trajectory.append({
            "step": self.step_count,
            "observation": self.current_observation,
        })

    def step(self):
        """Execute one ReAct turn and return observation / reward / status information."""

        if self.terminated or self.truncated:
            raise RuntimeError("The environment has terminated; call reset before step.")

        # In function-calling mode, llm_infer returns content, reasoning_content, and tool_calls.
        raw_response = llm_infer(
            provider=self.provider,
            model=self.model,
            messages=self.messages,
            tools=self.tools,
        )

        # The framework permits one tool call per turn; retain only the first if several are returned.
        message = {"role": "assistant", "content": raw_response["content"]}
        if raw_response["tool_calls"]:
            if len(raw_response["tool_calls"]) > 1:
                print("[WARNING] Multiple tool calls detected; only the first will be executed.")
            raw_response["tool_calls"] = [raw_response["tool_calls"][0]]
            message["tool_calls"] = [raw_response["tool_calls"][0]]
        if raw_response["reasoning_content"]:
            message["reasoning_content"] = raw_response["reasoning_content"]
        self.messages.append(message)

        # ToolEnv validates tool arguments, executes the tool, and produces the observation.
        if not raw_response['tool_calls'] and not raw_response['content']:
            # Treat a response with neither a tool call nor content as abnormal termination.
            print("raw_response is empty, please check the model")
            action = ""
            observation, reward, terminated, truncated, info = (
                "action is empty, please check the model", 0, True, True, {"action": "", "role": "user"}
            )
        else:
            observation, reward, terminated, truncated, info = self.tool_env.step(action=raw_response)
            action = info["action"]

        # Near the step limit, remind the leaf agent to submit an answer promptly.
        if self.step_count == self.max_steps - 2:
            observation += "\nSystem warning: You have reached the maximum tool usage rounds. You now need to organize and use the `submit_answer` tool to submit your answer."

        # Synchronize runtime state so run() can return the final log.
        self.step_count += 1
        self.total_reward += float(reward or 0.0)
        self.current_observation = observation
        self.current_info = info
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        # Tool observations require tool_call_id; feedback without a tool call becomes a user message.
        assert "role" in info
        if info["role"] == "tool":
            tool_call_id = raw_response["tool_calls"][0]['id']
            tool_call_name = raw_response["tool_calls"][0]['function']['name']
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_call_name,
                "content": observation,
            })

        elif info["role"] == "user":
            self.messages.append({"role": "user", "content": observation})
        else:
            raise ValueError(f"Unsupported observation role: {info['role']}")

        # The action field records this turn's tool-call summary for log analysis and review.
        self.trajectory.append({
            "step": self.step_count,
            "action": action,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        return observation, reward, terminated, truncated, info, action

    def run(self, task_info: dict):
        """Run a complete leaf agent and return a result dict embeddable in the WebSwarm log tree."""

        self.reset(task_info=task_info)

        while (not self.terminated) and (not self.truncated) and (self.step_count < self.max_steps):
            self.step()

        # prediction_answer is written to current_info only after submit_answer succeeds.
        prediction_answer = self.current_info.get("prediction_answer", None)
        tool_states = self.tool_env.get_tools_state()

        result = {
            "task_info":         self.task_info,
            "prediction_answer": prediction_answer,
            "tools":             self.tools,
            "messages":          self.messages,
            "trajectory":        self.trajectory,
            "total_reward":      self.total_reward,
            "terminated":        self.terminated,
            "truncated":         self.truncated,
            "final_observation": self.current_observation,
            "final_info":        self.current_info,
            "steps":             self.step_count,
            "tool_states":       tool_states,
        }

        return result
