"""LLM-judge call adapter used by the original DeepWideSearch evaluation flow.

This file connects the original evaluation package's llm_completion interface to the
project's unified llm_infer and reads service configuration from judge environment variables.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union

from loguru import logger

from .schema import LLMOutputItem, ModelResponse, ToolCall
from llm_infer.llm_infer import llm_infer


@dataclass
class APIResponse:
    """Minimal LLM response object expected by the original evaluation flow."""

    content: str


def llm_completion(
    messages: Union[str, List[dict]],
    tools: Optional[List[dict]] = None,
    model_config_name: str = "default_eval_config",
    *,
    judge_model_name: str,
    judge_model_provider: str,
) -> Optional[APIResponse]:
    """Use the project's unified llm_infer as the LLM-judge backend."""

    if isinstance(messages, str):
        # The original evaluation package sometimes passes a string; wrap it as chat messages here.
        messages = [{"role": "user", "content": messages}]

    result = llm_infer(
        provider=judge_model_provider,
        model=judge_model_name,
        messages=messages,
        tools=tools,
        temperature=0.0,
        enable_thinking=False,
        generation_config={"max_tokens": 10240},
        extra_params={
            "api_key_env": "JUDGE_MODEL_API_KEY",
            "base_url_env": "JUDGE_MODEL_BASE_URL",
        },
    )
    content = result.get("content")
    if content is None:
        logger.warning("[llm_completion] llm_infer returned no content")
        return None
    return APIResponse(content=content)


def transform_model_response(response: Any | None) -> ModelResponse:
    """Convert a richer backend response to the original evaluation package's ModelResponse structure."""
    out = ModelResponse()
    if response is None:
        out.error_marker = {"message": "Calling LLM failed."}
        return out

    # Store the simplified response in the project's standard ModelResponse structure.
    item = LLMOutputItem(content=response.content)
    # Read optional fields through a dict to support response objects from different backends.
    resp_dict = response.model_dump()
    if resp_dict.get("reasoning_content"):
        item.reasoning_content = resp_dict["reasoning_content"]
    if resp_dict.get("signature"):
        item.signature = resp_dict["signature"]

    if response.tool_calls:
        item.tool_calls = []
        for tool_call in response.tool_calls:
            item.tool_calls.append(
                ToolCall(
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    # Add ID generation later if a backend does not provide one and it becomes necessary.
                    tool_call_id=tool_call.id,
                )
            )
    out.outputs.append(item)
    return out
