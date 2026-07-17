"""LLM-judge call adapter for WideSearch table evaluation.

table_evaluate depends only on a `(template, kwargs) -> str` callback. This file connects
that callback to the unified llm_infer and uses judge environment variables prepared by the runner.
"""

from llm_infer.llm_infer import llm_infer


def call_claude(
    template: str,
    kwargs: dict,
    *,
    judge_model_name: str,
    judge_model_provider: str,
) -> str:
    """Call a Claude Sonnet model for tasks such as LLM judging that require high-quality reasoning.

    The implementation uses llm_infer and explicitly disables thinking during evaluation
    to keep judge behavior stable.
    """
    prompt = template.format(**kwargs)
    response = llm_infer(
        provider=judge_model_provider,
        model=judge_model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        enable_thinking=False,
        generation_config={"max_tokens": 32768},
        extra_params={
            "api_key_env": "JUDGE_MODEL_API_KEY",
            "base_url_env": "JUDGE_MODEL_BASE_URL",
        },
        stream=True,
    )
    return response.get("content") or ""
