"""Answer-submission tool providing the submit_answer function-call schema and basic validation."""

from .base_tool import BaseTool


class TerminateTool(BaseTool):
    """Stateless termination tool; ToolEnv.step handles the actual termination logic."""

    def __init__(self, tool_config=None) -> None:
        super().__init__()

    def get_info(self) -> dict:
        """Return metadata for the final-answer submission tool."""
        return {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Complete the task, pass the completion status and answer to the user. The answer can only be text or table, other formats such as html webpage should not be passed here.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The final answer to the task. If the answer is text, keep it concise. If the answer is a table, follow the format specified in your system prompt."
                        }
                    },
                    "required": ["answer"]
                }
            }
        }

    def execute(self, tool_params: dict) -> dict:
        """Return the submitted answer directly; ToolEnv normally intercepts submit_answer at runtime."""
        # Validate the input-parameter format.
        if not isinstance(tool_params, dict) or "answer" not in tool_params:
            error_msg = f"Invalid tool_params format for terminate tool. Expected dict with 'answer' key, got {type(tool_params)}"
            return {"type": "tool_result", "content": error_msg, "is_error": True}

        # Return the predicted answer.
        prediction_answer = tool_params["answer"]
        return {"type": "tool_result", "content": prediction_answer, "is_error": False}
    
    def clear_tool_state(self):
        """Clear tool state; the termination tool is stateless."""
        pass

    def return_tool_state(self):
        """The termination tool has no additional runtime state."""
        return {}
