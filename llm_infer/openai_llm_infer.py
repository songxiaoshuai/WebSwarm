import time
from typing import Dict, Any, Optional, Callable

import httpx
from openai import OpenAI, NOT_GIVEN


# ============================================================================
# Module overview: standard OpenAI ChatCompletion inference wrapper.
#   - Supports both non-streaming and streaming (SSE) calls with identical output structures.
#   - Includes exception retries and anti-repetition retries for finish_reason=length.
# ============================================================================


# ============================================================================
# Request parameters / client
# ============================================================================

def _build_request_params(
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        temperature: float,
        enable_thinking: bool,
        generation_config: Dict[str, Any],
        extra_params: Dict[str, Any],
        retry_for_length: bool = False,
) -> Dict[str, Any]:
    """Build OpenAI ChatCompletion request parameters.

    Consolidate parameter-processing logic in one place, including:
    - Anti-repetition parameters (initial call vs. length retry)
    - thinking-mode extra_body
    - User-supplied extra_params
    - Tool-call configuration
    - Generation parameters such as max_tokens / stop

    Returns:
        A dict that can be unpacked directly into client.chat.completions.create(**params).
    """
    # ---- Anti-repetition parameters: use base values initially and stronger values for length retries ----
    # Qwen models can degrade into repetition during long outputs, so use a stronger anti-repetition strategy;
    # base values suffice for other models and avoid introducing unnecessary bias.
    _base_extra = {"top_k": 20, "repetition_penalty": 1.0}
    _base_freq  = 0.0
    if "qwen" in model.lower():
        _anti_extra = {"top_k": 20, "repetition_penalty": 1.1, "no_repeat_ngram_size": 10}
        _anti_freq  = 0.3
    else:
        _anti_extra = _base_extra
        _anti_freq  = 0.0

    # Switch to anti-repetition values only for length-triggered retries; keep base values for the initial call.
    sampling_extra = _anti_extra if retry_for_length else _base_extra
    frequency_penalty = _anti_freq if retry_for_length else _base_freq

    # ---- thinking mode ----
    # Backends that require different fields can override extra_body through extra_params.
    thinking_extra: Dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking}
    }

    # ---- Merge extra_body: sampling + thinking + user values (user values have highest priority) ----
    # Apply user-supplied extra_params last so they can override framework defaults (such as a custom top_k).
    extra_body = {**sampling_extra, **thinking_extra, **extra_params}

    # ---- Generation parameters ----
    max_tokens: int = generation_config.get("max_tokens", 32 * 1024)
    stop = generation_config.get("stop", NOT_GIVEN)

    # ---- Tool calls ----
    # Enable tool_choice="auto" only when tools are provided explicitly; otherwise let the server use its default.
    tool_choice = "auto" if tools else NOT_GIVEN

    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "frequency_penalty": frequency_penalty,
        "max_tokens": max_tokens,
        "stop": stop,
        "tools": tools if tools else NOT_GIVEN,
        "tool_choice": tool_choice,
        "stream": False,
        "extra_body": extra_body if extra_body else NOT_GIVEN,
    }


def _create_client(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        is_local: bool = False,
) -> OpenAI:
    """Build an OpenAI client."""
    client_kwargs: Dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    if is_local:
        client_kwargs["http_client"] = httpx.Client(trust_env=False)
    return OpenAI(**client_kwargs)


# ============================================================================
# Response parsing
# ============================================================================

def _parse_response(response) -> Dict[str, Any]:
    """Extract a normalized result from a ChatCompletion response.

    Returns:
        A dict containing:
            - content (str | None)
            - reasoning_content (str | None):  Nonstandard extension field
            - tool_calls (list | None):        OpenAI-format tool calls
            - token_usage (dict):              prompt_tokens / completion_tokens / total_tokens
            - finish_reason (str | None)
    """
    choice = response.choices[0]
    msg = choice.message

    # reasoning_content is nonstandard (Seed, DeepSeek, etc.); use getattr to avoid AttributeError.
    reasoning_content = getattr(msg, "reasoning_content", None)

    # tool_calls is already aggregated for non-streaming responses; convert it to plain dicts for serialization.
    tool_calls = None
    if msg.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    # Token usage: some gateways omit usage on fallback error paths, so guard against None.
    usage = response.usage
    token_usage = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    } if usage else {}

    return {
        "content": msg.content or None,
        "reasoning_content": reasoning_content or None,
        "tool_calls": tool_calls,
        "token_usage": token_usage,
        "finish_reason": choice.finish_reason,
    }


def _parse_stream(response) -> Dict[str, Any]:
    """Accumulate and extract a normalized result from a streaming ChatCompletion response.

    Key streaming-protocol details:
    - Each chunk.choices[0].delta carries incremental fields (content / reasoning_content / tool_calls)
    - tool_calls uses index to identify a call; name/arguments arrive in fragments and must be accumulated by index
    - usage appears only in the final chunk (requires stream_options={"include_usage": True})
    - finish_reason appears in the final chunk that contains content

    Returns:
        A dict matching _parse_response, plus finish_reason.
    """
    # Accumulate each delta type separately; store tool calls by index to support out-of-order and parallel calls.
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: Dict[int, Dict[str, Any]] = {}
    finish_reason: Optional[str] = None
    token_usage: Dict[str, int] = {}

    for chunk in response:
        # usage may appear in the final chunk without choices (include_usage must be enabled).
        usage = getattr(chunk, "usage", None)
        if usage:
            token_usage = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

        # Skip subsequent delta handling for a trailing packet that contains usage but no choices.
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        # Main text delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)

        # Reasoning-content delta (a nonstandard extension supported by only some providers)
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            reasoning_parts.append(rc)

        # tool_calls fragments: name and arguments may span chunks, so accumulate them by index.
        if getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                idx = tc.index
                slot = tool_calls_acc.setdefault(idx, {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                # id/type usually appear only in the first fragment; write only nonempty values to preserve them.
                if tc.id:
                    slot["id"] = tc.id
                if tc.type:
                    slot["type"] = tc.type
                fn = getattr(tc, "function", None)
                if fn:
                    if getattr(fn, "name", None):
                        slot["function"]["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["function"]["arguments"] += fn.arguments

        # finish_reason appears in the final chunk that contains content.
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    # Emit entries by ascending index so dict insertion order cannot affect the result.
    tool_calls = (
        [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]
        if tool_calls_acc else None
    )
    content = "".join(content_parts) or None
    reasoning_content = "".join(reasoning_parts) or None

    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "token_usage": token_usage,
        "finish_reason": finish_reason,
    }


# ============================================================================
# Retry driver
# ============================================================================

_EMPTY_RESULT: Dict[str, Any] = {
    "content": "Nothing generated after max retries.",
    "reasoning_content": None,
    "tool_calls": None,
    "token_usage": {},
    # finish_reason is not part of the public structure; _run_with_retry removes it before returning.
    # It is retained only for special cases such as context_length_exceeded so callers can identify the failure.
}


def _run_with_retry(
        func_name: str,
        client: OpenAI,
        build_params: Callable[[bool], Dict[str, Any]],
        parser: Callable[[Any], Dict[str, Any]],
        max_try: int = 10,
) -> Dict[str, Any]:
    """Unified retry loop for exceptions and anti-repetition retries on finish_reason=length.

    Args:
        func_name:    Function name displayed in logs
        client:       OpenAI client
        build_params: (retry_for_length) -> request-parameter dict
        parser:       (response) -> normalized result dict containing finish_reason
        max_try:      Maximum attempts shared by exception and length retries

    Returns:
        A normalized result dict with finish_reason removed; returns _EMPTY_RESULT after all retries.
    """
    cur_try = 0
    retry_for_length = False
    while cur_try < max_try:
        try:
            # Rebuild parameters each round because length-triggered retries switch to anti-repetition values.
            params = build_params(retry_for_length)
            response = client.chat.completions.create(**params)
            result = parser(response)

            # ---- Handle finish_reason=length ----
            # Distinguish between two cases:
            #   1) completion_tokens < max_tokens: the server truncated early because the prompt filled the context window.
            #      A retry cannot produce more content, so stop retrying and return the current result.
            #   2) completion_tokens >= max_tokens: the model output reached the limit, possibly due to repetition.
            #      Retry once with anti-repetition parameters enabled.
            # ---- Both content and tool_calls are empty: the model returned an empty response ----
            if not result.get("content") and not result.get("tool_calls"):
                print(f"[Warning] Empty response (content=None, tool_calls=None), "
                      f"retrying... ({cur_try + 1}/{max_try})")
                cur_try += 1
                continue

            if result["finish_reason"] == "length":
                comp_tokens = result["token_usage"].get("completion_tokens", 0)
                req_max = params.get("max_tokens", 0)
                if comp_tokens < req_max:
                    print(f"[Warning] finish_reason=length, completion_tokens({comp_tokens}) < max_tokens({req_max}), "
                          f"context window exhausted, skipping retry.")
                else:
                    print(f"[Warning] finish_reason=length, completion_tokens({comp_tokens}) >= max_tokens({req_max}), "
                          f"retrying with anti-repeat params... ({cur_try + 1}/{max_try})")
                    retry_for_length = True
                    cur_try += 1
                    continue

            # Remove the internal finish_reason field and expose only the stable public structure.
            result.pop("finish_reason", None)
            return result

        except Exception as e:
            # ---- Context too long: return immediately without retrying ----
            # Common context-length error terms across OpenAI, vLLM, Volcengine, and other backends
            _err_lower = str(e).lower()
            _ctx_keywords = (
                "context_length_exceeded",
                "context length exceeded",
                "maximum context length",
                "reduce the length",
                "too many tokens",
                "input is too long",
                "prompt is too long",
            )
            if any(kw in _err_lower for kw in _ctx_keywords):
                print(
                    f"[Error] {func_name}: context window exceeded, NOT retrying. "
                    f"Raw error: {e}"
                )
                result = dict(_EMPTY_RESULT)
                result["finish_reason"] = "context_length_exceeded"
                return result

            # Network, rate-limit, or server error: retry with linear backoff.
            wait_time = 10 * (cur_try + 1)
            print(f"[Error] {e}, retrying... ({cur_try + 1}/{max_try}), waiting {wait_time}s")
            time.sleep(wait_time)
            cur_try += 1

    # All retries exhausted: return a placeholder dict matching the normal structure to prevent downstream KeyError.
    print(f"[Error] {func_name} failed after {max_try} retries, returning empty result.")
    return dict(_EMPTY_RESULT)


# ============================================================================
# Public API
# ============================================================================

def openai_llm_infer_stream(
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.6,
        enable_thinking: bool = False,
        api_key: str = None,
        base_url: str = None,
        generation_config: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """Streaming inference function.

    Inputs and outputs match `openai_llm_infer`; the only difference is that the underlying
    implementation receives deltas through an SSE stream and accumulates them here.

    Returns:
        The same structure as the non-streaming version (content / reasoning_content / tool_calls / token_usage).
    """
    generation_config = generation_config or {}
    extra_params = extra_params or {}

    is_local = bool(model and ("qwen" in model.lower() or "glm" in model.lower()  or "kimi" in model.lower()))
    client = _create_client(api_key=api_key, base_url=base_url, is_local=is_local)

    # Closure invoked for every retry; retry_for_length=True switches to anti-repetition parameters.
    def _build(retry_for_length: bool) -> Dict[str, Any]:
        params = _build_request_params(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            enable_thinking=enable_thinking,
            generation_config=generation_config,
            extra_params=extra_params,
            retry_for_length=retry_for_length,
        )
        # Streaming only: enable SSE and request usage in the final packet (unsupported by some backends).
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        return params

    return _run_with_retry(
        func_name="openai_llm_infer_stream",
        client=client,
        build_params=_build,
        parser=_parse_stream,
    )


def openai_llm_infer(
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.6,
        enable_thinking: bool = False,
        api_key: str = None,
        base_url: str = None,
        stream: bool = False,
        generation_config: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """Standard OpenAI-format LLM inference function (non-streaming).

    Input and output formats align with OpenAI ChatCompletion and require no conversion.

    Args:
        model:             Model name, such as "gpt-4o" / "deepseek-chat"
        messages:          OpenAI-format message list supporting system / user / assistant / tool roles
        tools:             Tool list in OpenAI function-calling format; None disables tools
        temperature:       Sampling temperature
        enable_thinking:   Whether to enable thinking mode through extra_body; supported by some providers only
        api_key:           API key
        base_url:          Custom API endpoint for an OpenAI-compatible third-party service
        generation_config: Additional generation parameters; supported fields:
                           - max_tokens (int, default)
                           - stop (list[str])
        extra_params:      Additional parameters forwarded to extra_body, such as
                           {"thinking": {"type": "enabled"}}

    Returns:
        A dict containing:
            - content (str | None):           Primary model text output
            - reasoning_content (str | None): Reasoning content (a nonstandard extension supported by some providers)
            - tool_calls (list | None):       OpenAI-format tool-call list; each item contains
                                              id / type="function" / function.name / function.arguments
            - token_usage (dict):             Token statistics containing
                                              prompt_tokens / completion_tokens / total_tokens
    """
    generation_config = generation_config or {}
    extra_params = extra_params or {}

    # With stream=True, reuse the streaming implementation; its output matches the non-streaming structure.
    if stream:
        return openai_llm_infer_stream(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            enable_thinking=enable_thinking,
            api_key=api_key,
            base_url=base_url,
            generation_config=generation_config,
            extra_params=extra_params,
        )

    is_local = bool(model and ("qwen" in model.lower() or "glm" in model.lower() or "kimi" in model.lower()))
    client = _create_client(api_key=api_key, base_url=base_url, is_local=is_local)

    # Closure invoked for every retry; retry_for_length=True switches to anti-repetition parameters.
    def _build(retry_for_length: bool) -> Dict[str, Any]:
        return _build_request_params(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            enable_thinking=enable_thinking,
            generation_config=generation_config,
            extra_params=extra_params,
            retry_for_length=retry_for_length,
        )

    return _run_with_retry(
        func_name="openai_llm_infer",
        client=client,
        build_params=_build,
        parser=_parse_response,
    )
