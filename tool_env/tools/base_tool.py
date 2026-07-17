"""Abstract tool base class defining the minimal interface required by ToolEnv."""


class BaseTool:
    """Common interface for all tool implementations."""

    def __init__(self):
        self.name: str = ""

    def get_info(self) -> dict:
        """Return function-call tool metadata."""
        raise NotImplementedError

    def execute(self, tool_params: dict) -> dict:
        """Execute tool logic and return a result ToolEnv can convert directly to an observation."""
        raise NotImplementedError
    
    def clear_tool_state(self):
        """Clear tool state accumulated during one task run."""
        raise NotImplementedError
    
    def return_tool_state(self) -> dict:
        """Return tool state for log statistics and review."""
        raise NotImplementedError 
