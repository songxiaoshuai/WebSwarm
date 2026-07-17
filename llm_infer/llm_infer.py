"""Unified LLM inference entry point."""

import os
from typing import Dict, Any, Optional

from .claude_llm_infer import claude_llm_infer
from .openai_llm_infer import openai_llm_infer


LLM_BASE_URL_ENV = "LLM_BASE_URL"
LLM_API_KEY_ENV = "LLM_API_KEY"


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Set it in the environment before running.")
    return value


def _consume_env_overrides(
    extra_params: Optional[Dict[str, Any]],
) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Consume environment-variable overrides at the llm_infer layer without forwarding them to the backend."""
    backend_extra_params = dict(extra_params or {})
    api_key_env = backend_extra_params.pop("api_key_env", LLM_API_KEY_ENV)
    base_url_env = backend_extra_params.pop("base_url_env", LLM_BASE_URL_ENV)
    return (
        _get_required_env(api_key_env),
        _get_required_env(base_url_env),
        backend_extra_params,
    )


def llm_infer(
        provider: str,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.6,
        enable_thinking: bool = True,
        generation_config: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
) -> dict:
    api_key, base_url, backend_extra_params = _consume_env_overrides(
        extra_params
    )
    provider = provider.lower()

    if provider == "openai":
        return openai_llm_infer(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            enable_thinking=enable_thinking,
            api_key=api_key,
            base_url=base_url,
            generation_config=generation_config,
            extra_params=backend_extra_params,
            stream=stream,
        )
    elif provider == "claude":
        return claude_llm_infer(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            enable_thinking=enable_thinking,
            api_key=api_key,
            base_url=base_url,
            generation_config=generation_config,
            extra_params=backend_extra_params,
            stream=stream,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
