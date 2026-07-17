"""Tool-environment entry point that loads a fixed tool set and dispatches model tool_calls."""

import json
from copy import deepcopy

from .tools import WebSearchTool, FetchURLTool, TerminateTool


class ToolEnv:
    """Pure tool environment managing tool loading/execution without task or evaluation logic."""

    def __init__(self, config: dict | None = None):
        """Initialize the tool environment.

        Args:
            config: Tool-configuration dict supporting keys such as enable_tools / search /
                    fetch_url / submit_answer.
        """
        if config is None:
            config = {}
        self.enable_tools = config["enable_tools"]
        self.tool_config_map = {}
        for tool_name in self.enable_tools:
            self.tool_config_map[tool_name] = config[tool_name]
        self.tools = {}
        self._load_tools()

    # ── Tool loading ─────────────────────────────────────────────────────

    def _load_tools(self):
        """Load tool instances according to config['enable_tools'].

        The tool set is fixed:
            search        -> WebSearchTool (single-query interface)
            fetch_url     -> FetchURLTool (single-URL interface)
            submit_answer -> TerminateTool
        """
        # Keep instance attributes so higher-level agents can access specific tool state directly.
        self.search_tool = None
        self.fetch_url_tool = None
        self.terminate_tool = None

        if "search" in self.enable_tools:
            cfg = self.tool_config_map["search"] or {}
            self.search_tool = WebSearchTool(tool_config=cfg)
            self.tools["search"] = self.search_tool

        if "fetch_url" in self.enable_tools:
            cfg = self.tool_config_map["fetch_url"] or {}
            self.fetch_url_tool = FetchURLTool(tool_config=cfg)
            self.tools["fetch_url"] = self.fetch_url_tool

        if "submit_answer" in self.enable_tools:
            cfg = self.tool_config_map["submit_answer"] or {}
            self.terminate_tool = TerminateTool(tool_config=cfg)
            self.tools["submit_answer"] = self.terminate_tool

    # ── Tool execution ───────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_params: dict) -> dict:
        """Execute the specified tool and return {type, content, is_error}."""
        if tool_name not in self.tools:
            error_msg = f"Unknown tool: {tool_name}. Available tools: {list(self.tools.keys())}"
            return {"type": "tool_result", "content": error_msg, "is_error": True}

        tool = self.tools[tool_name]
        try:
            return tool.execute(tool_params)
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            print(f"[ENV] {error_msg}")
            return {"type": "tool_result", "content": error_msg, "is_error": True}

    # ── Main interaction interface ───────────────────────────────────────

    def step(self, action: dict):
        """Execute one tool-environment interaction and return observation/reward/terminated/truncated/info."""
        action = deepcopy(action)
        observation, reward, terminated, truncated, info = None, None, False, False, {"action": action}

        # No tool call
        if not action.get("tool_calls"):
            observation = f"No tools called. Use one of the following tools: {list(self.tools.keys())}"
            info["role"] = "user"
            return observation, reward, terminated, truncated, info

        assert len(action["tool_calls"]) == 1, (
            f"Expected exactly one tool call per action, but got {len(action['tool_calls'])}"
        )

        tool_name = action["tool_calls"][0]["function"]["name"]
        try:
            tool_params = action["tool_calls"][0]["function"]["arguments"]
            if isinstance(tool_params, str):
                tool_params = json.loads(tool_params)
        except json.JSONDecodeError as e:
            observation = (
                f"Failed to parse tool parameters as JSON: {str(e)}. "
                f"Original parameters: {action['tool_calls'][0]['function']['arguments']}"
            )
            info["role"] = "tool"
            print(f"[ENV] {observation}")
            truncated = True
            return observation, reward, terminated, truncated, info

        # submit_answer termination tool.
        if self.is_action_terminated(action):
            prediction_answer = tool_params.get("answer", "")
            # Reject an empty answer without terminating, and return feedback so the LLM can retry.
            if not prediction_answer or not str(prediction_answer).strip():
                observation = (
                    "[Error] Your submit_answer call had an empty or missing `answer` field. "
                    "The task is NOT complete. Please collect the required information and "
                    "call submit_answer again with a non-empty answer."
                )
                info["role"] = "tool"
                print(f"[ENV] {observation}")
                return observation, reward, terminated, truncated, info

            print("[ENV] Observing the terminate tool call, environment will enter terminated state")
            terminated = True
            observation = f"Task terminated. Prediction Answer: {prediction_answer}"
            info["role"] = "tool"
            info["prediction_answer"] = prediction_answer
            return observation, reward, terminated, truncated, info

        # Route ordinary tool calls uniformly through the tool instance's execute method.
        tool_result = self._execute_tool(tool_name, tool_params)
        observation = tool_result["content"]
        info["role"] = "tool"
        return observation, reward, terminated, truncated, info

    # ── Helper methods ───────────────────────────────────────────────────

    def get_tools_info(self) -> list[dict]:
        """Return metadata for all tools."""
        return [tool.get_info() for tool in self.tools.values()]

    def reset(self):
        """Reset all tool state without recreating tool instances."""
        for tool in self.tools.values():
            tool.clear_tool_state()

    def get_tools_state(self) -> dict:
        """Return the current state of all tools."""
        return {name: tool.return_tool_state() for name, tool in self.tools.items()}

    def is_action_terminated(self, action: dict) -> bool:
        """Determine whether an action contains a terminate tool call."""
        for tool_call in action.get("tool_calls", []):
            if tool_call["function"]["name"] == "submit_answer":
                return True
        return False
