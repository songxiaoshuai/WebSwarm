"""
Summary of differences between the Claude and OpenAI inference interfaces
=========================================================================

1. System Prompt
   - OpenAI:    system is the first entry in the messages list,
                {"role": "system", "content": "..."}
   - Anthropic: system is a separate top-level `system="..."` argument and is not included in messages

2. Tool Schema Definitions
   - OpenAI:    {"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON Schema>}}
   - Anthropic: {"name": ..., "description": ..., "input_schema": <JSON Schema>}
   Difference: the outer function wrapper and the JSON Schema field name parameters → input_schema

3. Assistant Tool-Call Output (tool_calls)
   - OpenAI:    tool_calls is a top-level field and arguments is a JSON string
                {"role": "assistant", "tool_calls": [{"id": ..., "type": "function",
                    "function": {"name": ..., "arguments": "{\"city\": \"Beijing\"}"}}]}
   - Anthropic: tool_use is a block in the content array and input is an already parsed dict
                {"role": "assistant", "content": [{"type": "tool_use", "id": ...,
                    "name": ..., "input": {"city": "Beijing"}}]}

4. Returning Tool Results
   - OpenAI:    each result is a separate role="tool" message with a tool_call_id field;
                multiple results are sent as multiple messages
                {"role": "tool", "tool_call_id": "call_xxx", "content": "..."}
   - Anthropic: results are embedded in the content array of a role="user" message under tool_use_id;
                multiple results become multiple blocks in the same user message
                {"role": "user", "content": [{"type": "tool_result",
                    "tool_use_id": "toolu_xxx", "content": "..."}]}

5. Thinking / Reasoning
   - OpenAI:    some models output reasoning through reasoning_content or <think> tags (nonstandard)
   - Anthropic: enabled through the thinking configuration and returned in a separate thinking block
                (type="thinking"); temperature must be set to 1.0
"""
import json
import time
from typing import Dict, Any, Optional

from anthropic import Anthropic, NOT_GIVEN, RateLimitError

def _extract_system_and_messages(messages: list[dict]):
    """Extract the system prompt and remaining messages from a message list."""
    if messages and messages[0].get("role") == "system":
        system = messages[0].get("content", "")
        if isinstance(system, list):
            # content may be a list of blocks
            texts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
            system = "\n".join(texts)
        return system, messages[1:]
    return None, messages


def _convert_tools_to_anthropic_format(tools: list[dict]) -> list[dict]:
    """Convert an OpenAI function-calling tool list to Anthropic format.

    OpenAI format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    Anthropic format: {"name": ..., "description": ..., "input_schema": ...}
    """
    result = []
    for tool in tools:
        fn = tool["function"]
        anthropic_tool: Dict[str, Any] = {"name": fn["name"]}
        if "description" in fn:
            anthropic_tool["description"] = fn["description"]
        anthropic_tool["input_schema"] = fn.get("parameters", {"type": "object", "properties": {}})
        result.append(anthropic_tool)
    return result


def _convert_messages_to_anthropic_format(messages: list[dict]) -> list[dict]:
    """Convert an OpenAI-format message history to Anthropic format.

    Handle two types of differences:
    1. tool_calls in assistant messages:
       OpenAI:    {"role": "assistant", "tool_calls": [{"id": ..., "type": "function",
                      "function": {"name": ..., "arguments": "<JSON string>"}}]}
       Anthropic: {"role": "assistant", "content": [{"type": "tool_use", "id": ...,
                      "name": ..., "input": <dict>}]}

    2. Tool-result messages:
       OpenAI:    {"role": "tool", "tool_call_id": ..., "content": "..."}
       Anthropic: {"role": "user", "content": [{"type": "tool_result",
                      "tool_use_id": ..., "content": "..."}]}
       Note: consecutive role=tool messages are merged into the content array of one user message.
    """
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        # ── assistant with tool_calls ──
        if role == "assistant" and msg.get("tool_calls"):
            content: list = []
            # Retain the text parts
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn["name"],
                    "input": json.loads(fn["arguments"]),
                })
            result.append({"role": "assistant", "content": content})
            i += 1
            continue

        # ── tool result(s) → merge into a single user message ──
        if role == "tool":
            tool_result_blocks = []
            while i < len(messages) and messages[i].get("role") == "tool":
                t = messages[i]
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": t["tool_call_id"],
                    "content": t["content"],
                })
                i += 1
            result.append({"role": "user", "content": tool_result_blocks})
            continue

        # ── Preserve ordinary messages as-is ──
        result.append(msg)
        i += 1

    return result


def _parse_stop_message(stop_message) -> Dict[str, Any]:
    """Parse an Anthropic EndpointMessage into an OpenAI-compatible format.

    Return fields aligned with the OpenAI ChatCompletion format:
    - content:           text output (str | None)
    - reasoning_content: thinking-block content (str | None)
    - tool_calls:        list of OpenAI-format tool calls, each containing:
                           id / type="function" / function.name / function.arguments (JSON string)
    - token_usage:       {"input_tokens": ..., "output_tokens": ...}
    """
    content_text: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list] = None

    for block in stop_message.content:
        if block.type == "thinking":
            reasoning_content = block.thinking
        elif block.type == "text":
            content_text = block.text
        elif block.type == "tool_use":
            if tool_calls is None:
                tool_calls = []
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input, ensure_ascii=False),
                },
            })

    usage = stop_message.usage
    token_usage = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    # Cache-related fields returned by some APIs
    for attr in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        val = getattr(usage, attr, None)
        if val is not None:
            token_usage[attr] = val

    return {
        "content": content_text,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "token_usage": token_usage,
    }


def claude_llm_infer(
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.0,
        enable_thinking: bool = True,
        api_key: str = None,
        base_url: str = None,
        generation_config: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
    """Unified Claude LLM inference function.

    Args:
        model:             Model name, such as "claude-3-7-sonnet-20250219"
        messages:          List of dicts containing "role" / "content"; the first message is
                           extracted automatically as the system prompt when role=="system"
        tools:             Tool definitions in OpenAI function-calling format, converted internally
                           to Anthropic format; None disables tools
        temperature:       Sampling temperature; forced to 1.0 when thinking is enabled
        enable_thinking:   Whether to enable extended thinking
        api_key:           Key used for direct API access
        base_url:          Custom API endpoint
        generation_config: Additional generation parameters; supported fields:
                           - max_tokens (int, default 8192)
                           - stop (list[str])
                           - budget_tokens (int, thinking budget, default max_tokens // 2)
        extra_params:      Provider-specific parameters reserved for extensions
        stream:            Whether to obtain results through the streaming interface
                           (calls are always streamed internally to improve stability)

    Returns:
        A dict containing:
            - content (str | None):           Primary model text output
            - reasoning_content (str | None): Thinking/reasoning content (thinking block)
            - tool_calls (list | None):       Tool-call list; each item contains id / name / input
            - token_usage (dict):             Token statistics with input_tokens / output_tokens
    """
    generation_config = generation_config or {}
    extra_params = extra_params or {}

    max_tokens: int = generation_config.get("max_tokens", 8192)
    stop_sequences = generation_config.get("stop", NOT_GIVEN)
    if not stop_sequences:
        stop_sequences = NOT_GIVEN

    # ---- Build the thinking configuration ----
    thinking_config = NOT_GIVEN
    actual_temperature = temperature
    if enable_thinking:
        budget_tokens: int = generation_config.get("budget_tokens", max_tokens // 2)
        thinking_config = {"type": "enabled", "budget_tokens": budget_tokens}
        actual_temperature = 1.0  # Temperature must be 1.0 when thinking is enabled

    # ---- Build the client ----
    client = Anthropic(api_key=api_key,base_url=base_url)

    # ---- Extract the system prompt ----
    system_prompt, user_messages = _extract_system_and_messages(messages)
    user_messages = _convert_messages_to_anthropic_format(user_messages)

    # ---- Convert tool formats and configure the call ----
    anthropic_tools = _convert_tools_to_anthropic_format(tools) if tools else None
    tool_choice = NOT_GIVEN
    if anthropic_tools:
        tool_choice = {"type": "auto", "disable_parallel_tool_use": False} 

    # ---- Call with RateLimitError retries ----
    cur_try = 0
    max_try = 5
    while cur_try < max_try:
        try:
            if stream:
                with client.messages.stream(
                    model=model,
                    messages=user_messages,
                    system=system_prompt if system_prompt else NOT_GIVEN,
                    temperature=actual_temperature,
                    max_tokens=max_tokens,
                    stop_sequences=stop_sequences,
                    thinking=thinking_config,
                    tools=anthropic_tools if anthropic_tools else NOT_GIVEN,
                    tool_choice=tool_choice,
                ) as s:
                    # Wait for the stream to finish and retrieve the complete message
                    for _ in s:
                        pass
                    stop_message = s.get_final_message()
            else:
                stop_message = client.messages.create(
                    model=model,
                    messages=user_messages,
                    system=system_prompt if system_prompt else NOT_GIVEN,
                    temperature=actual_temperature,
                    max_tokens=max_tokens,
                    stop_sequences=stop_sequences,
                    thinking=thinking_config,
                    tools=anthropic_tools if anthropic_tools else NOT_GIVEN,
                    tool_choice=tool_choice,
                )
            break
        except Exception as e:
            print(f"Error: {e}, retrying... ({cur_try + 1}/{max_try})")
            time.sleep(10)
            cur_try += 1
            
    return _parse_stop_message(stop_message)
