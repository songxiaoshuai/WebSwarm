"""LLM-response and tool-call data structures from the original DeepWideSearch evaluation package.

These structures preserve compatibility with the original evaluation code's representation
of model output, tool calls, and run results. They do not directly determine the final log
structure of the current main runner.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from typing_extensions import TypedDict


class ErrorMarker(TypedDict):
    """Generic error marker for recording errors during a call."""

    message: str


@dataclass
class ToolCall:
    """Data structure for one tool call."""

    tool_name: str
    """Name of the tool to invoke."""

    arguments: str | dict[str, Any]
    """Arguments passed to the tool in either of two forms:
    
    - Dict: already parsed key-value arguments.
    - String: argument text in JSON form.
    """

    tool_call_id: str
    """Tool-call ID generated in the LLM response."""


class ToolCallResultDict(TypedDict):
    """Dictionary-form tool-call result."""

    data: str | None
    error: str | None
    system_error: str | None


@dataclass
class ToolCallResult:
    """Data structure for a tool-call result."""

    tool_call_id: str
    """Tool-call ID generated in the LLM response."""

    content: str | None = None
    """Content returned by the tool call; convert it to a string before assignment."""

    error_marker: ErrorMarker | None = None
    """Error marker set when the tool call fails."""

    system_error_marker: ErrorMarker | None = None
    """Error marker for a system-level exception."""

    extra: dict[str, Any] = field(default_factory=dict)

    def get_content_or_error(self) -> str:
        """Return tool-call content, or the error message when content is absent."""
        if self.content is not None:
            return self.content
        elif self.error_marker is not None:
            return self.error_marker["message"]
        else:
            raise ValueError(
                "[ToolCallResult] Must have one of content or error marker."
            )


@dataclass
class LLMOutputItem:
    """One LLM output item."""

    role: Literal["assistant"] = "assistant"
    """Role of the output message."""

    content: str | None = None
    """Output body."""

    reasoning_content: str | None = None
    """Reasoning content returned by the model."""

    signature: str | None = None
    """Step signature that some Claude thinking models may return."""

    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ModelResponse:
    """Non-streaming model response retaining the original evaluation package's output-container shape."""

    outputs: list[LLMOutputItem] = field(default_factory=list)

    session_id: str | None = None
    """Session ID."""

    error_marker: ErrorMarker | None = None
    """Error marker set when an LLM call fails."""


@dataclass
class RunResult:
    """Result after one agent execution round."""

    stop_reason: Literal["finished", "reach_max_steps", "error"] = "finished"

    content: str | None = None
    """Output body."""

    reasoning_content: str | None = None
    """Reasoning content returned by the model."""
