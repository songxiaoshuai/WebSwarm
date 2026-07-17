"""Experiment-configuration entry point.

This is the only researcher-facing configuration module and centralizes:

1. Experiment-selection constants: primary model, judge model, benchmark/version, and task_ids.
2. `load_env_file`: load the project-root .env (service endpoints and keys) into os.environ.
3. `apply_experiment_env`: inject the selected model/judge names into os.environ so
   experiment_config is their single source of truth.
4. `build_webswarm_config`: build and validate top-level WebSwarm runtime parameters
   (method hyperparameters are fixed, type/value validation happens here, and the
   webswarm package receives only a valid dict).
5. `build_webswarm_env_config`: build tool configuration that can be passed directly
   to `ToolEnv` (`search_engine` determines tool_engine, while `enable_cache` is an
   independent hyperparameter).
6. `build_experiment`: package the above into the complete configuration for one run
   (webswarm/env/task/task_ids) and validate benchmark/prompt_version consistency.
   The runner only needs to call this function instead of importing each selection constant.

Service endpoints and keys still come from .env; individual tool and LLM modules read
them directly from os.environ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameters (edit here for experiments)
# ═══════════════════════════════════════════════════════════════════════════

# ── Experiment selection ─────────────────────────────────────────────────────

# Primary model shared by the WebSwarm main agent, verb agents, and page summaries.
MODEL = "ep-20260122194732-jvvqq"
PROVIDER = "openai"

# Benchmark judge model (used during evaluation; endpoint and key remain in .env).
JUDGE_MODEL_NAME = "pa/claude-sonnet-4-5-20250929"
JUDGE_MODEL_PROVIDER = "claude"

# Dataset and version. See the comments below for supported combinations.
BENCHMARK = "browsecomp_plus"
BENCHMARK_VERSION = "bc_all"
# Supported combinations:
# BENCHMARK, BENCHMARK_VERSION = "browsecomp_plus", "all"          # 830
# BENCHMARK, BENCHMARK_VERSION = "browsecomp_plus", "plus_subset"  # 200
# BENCHMARK, BENCHMARK_VERSION = "widesearch", "all"
# BENCHMARK, BENCHMARK_VERSION = "widesearch", "en_subset"
# BENCHMARK, BENCHMARK_VERSION = "deepwidesearch", "all"
# BENCHMARK, BENCHMARK_VERSION = "deepwidesearch", "en_subset"
# BENCHMARK, BENCHMARK_VERSION = "gisa", "all"

# Task ID list; None runs all cases for the selected benchmark/version.
TASK_IDS = ["bc_en_1"]
# Common examples:
# None runs all cases for the selected benchmark/version.
# TASK_IDS = None
# TASK_IDS = [f"bc_en_{i}" for i in range(1, 6)]
# TASK_IDS = [f"bc_plus_subset_{i}" for i in range(151, 201)]
# TASK_IDS = [f"ws_en_{i:03d}" for i in range(1, 101)]
# TASK_IDS = ["gisa_1"]
# DeepWideSearch uses raw instance_id values, for example:
# TASK_IDS = ['wide2deep_ws_en_009', 'deep2wide_result_83_Cameroon']


# ── WebSwarm method hyperparameters ──────────────────────────────────────────
# Method hyperparameters are fixed by default; set PROMPT_VERSION to "gisa" to run GISA.
# Type and value validation is performed in build_webswarm_config.

MAX_STEPS = 200            # Maximum steps per task for every agent
PROMPT_VERSION = "general"  # Prompt version (GISA uses TSV output, so use "gisa" for the GISA dataset)
WEB_PROBING = True          # Whether to enable web probing
SUBTASK_EXPERIENCE = True   # Whether to enable subtask experience reuse


# ── ToolEnv tool configuration ────────────────────────────────────────────────
# Search engine and cache switches (experiment settings):
#   - web: uses the SERPER + JINA online services; caching is usually enabled to reuse prior requests.
#   - local: uses a self-hosted retrieval service (URL from LOCAL_ENGINE_BASE_URL in .env); caching is usually disabled.
# enable_cache is now controlled independently and is no longer derived from search_engine.

SEARCH_ENGINE = "web"  # "web" or "local"
ENABLE_CACHE = True    # Whether to enable search/fetch caching

# When switching to the local self-hosted retrieval service, use these settings (cache disabled):
# SEARCH_ENGINE = "local"
# ENABLE_CACHE = False

DEFAULT_SEARCH_MAX_RESULTS = 5           # Results returned per search round
DEFAULT_SEARCH_MAX_ROUND = 200           # Maximum number of search rounds
DEFAULT_SERPER_CACHE_DIR = "cache/serper_cache"   # Search cache directory (created automatically when caching is enabled)

DEFAULT_FETCH_MAX_TOKENS = 80000         # Maximum tokens retained when truncating webpage content
DEFAULT_JINA_CACHE_DIR = "cache/jina_cache"       # fetch_url cache directory (created automatically when caching is enabled)


# ═══════════════════════════════════════════════════════════════════════════
# Processing functions
# ═══════════════════════════════════════════════════════════════════════════


def apply_experiment_env() -> None:
    """Write the selected model/judge names to os.environ.

    Downstream judge evaluation reads JUDGE_MODEL_NAME / JUDGE_MODEL_PROVIDER from
    environment variables. Injecting them here keeps these selections in one file.
    """
    os.environ["WEBSWARM_MODEL"] = MODEL
    os.environ["WEBSWARM_PROVIDER"] = PROVIDER
    os.environ["JUDGE_MODEL_NAME"] = JUDGE_MODEL_NAME
    os.environ["JUDGE_MODEL_PROVIDER"] = JUDGE_MODEL_PROVIDER


def load_env_file(path: str | Path = ".env", override: bool = False) -> None:
    """Load a .env file into os.environ.

    By default, read .env from the project root. With override=False, existing shell
    environment variables take precedence and are not overwritten by .env.
    """
    env_path = Path(path)
    if not env_path.is_absolute():
        env_path = Path(__file__).resolve().parent / env_path
    load_dotenv(dotenv_path=env_path, override=override)


def build_webswarm_config(
    model: str,
    provider: str,
    *,
    max_steps: int = MAX_STEPS,
    prompt_version: str = PROMPT_VERSION,
    web_probing: bool = WEB_PROBING,
    subtask_experience: bool = SUBTASK_EXPERIENCE,
) -> dict:
    """Build and validate top-level WebSwarm runtime parameters.

    Method hyperparameters default to the constants above. Type/value validation occurs
    here, and the returned dict can be passed directly to WebSwarmAgent.
    """
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string.")
    if not isinstance(provider, str) or not provider:
        raise ValueError("provider must be a non-empty string.")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive int.")
    if not isinstance(prompt_version, str) or not prompt_version:
        raise ValueError("prompt_version must be a non-empty string.")
    if not isinstance(web_probing, bool):
        raise TypeError("web_probing must be a bool.")
    if not isinstance(subtask_experience, bool):
        raise TypeError("subtask_experience must be a bool.")
    return {
        "model": model,
        "provider": provider,
        "max_steps": max_steps,
        "prompt_version": prompt_version,
        "web_probing": web_probing,
        "subtask_experience": subtask_experience,
    }


def build_webswarm_env_config(
    *,
    model: str,
    provider: str,
    search_engine: str = SEARCH_ENGINE,
    enable_cache: bool = ENABLE_CACHE,
    local_engine_base_url: str | None = None,
) -> dict:
    """Build WebSwarm tool configuration that can be passed directly to ToolEnv.

    search_engine determines tool_engine (web -> default, local -> local), while
    enable_cache is supplied independently and is no longer derived from search_engine.
    """
    if search_engine not in {"web", "local"}:
        raise ValueError(
            "search_engine must be either 'web' or 'local', "
            f"got {search_engine!r}."
        )
    tool_engine = "default" if search_engine == "web" else "local"

    # search tool configuration: corresponds to single-query search in ToolEnv.
    # cache_dir need not exist beforehand; the cache layer calls mkdir(parents=True) when enabled.
    search_cfg = {
        "engine": tool_engine,               # "default" = Serper; "local" = self-hosted /search
        "max_results": DEFAULT_SEARCH_MAX_RESULTS,     # Results returned per round
        "max_search_round": DEFAULT_SEARCH_MAX_ROUND,  # Maximum number of search rounds
        "ignored_urls": None,                # URLs to block (None means no blocking)
        "enable_cache": enable_cache,        # Whether to read and write the search cache
        "cache_dir": DEFAULT_SERPER_CACHE_DIR,
    }
    # fetch_url tool configuration: single-URL fetching plus goal-based page summarization.
    fetch_cfg = {
        "max_tokens": DEFAULT_FETCH_MAX_TOKENS,  # Page-content truncation limit
        "engine": tool_engine,               # "default" = Jina; "local" = self-hosted /document
        "enable_cache": enable_cache,        # Shares the same switch as search
        "cache_dir": DEFAULT_JINA_CACHE_DIR,
        "web_summary": {                     # LLM for page summaries (same as the primary model)
            "model_provider": provider,
            "model_name": model,
        },
    }

    # Local mode: route search/fetch to the self-hosted /search and /document endpoints.
    if search_engine == "local":
        if not local_engine_base_url:
            raise ValueError("local_engine_base_url is required when search_engine='local'.")
        local_engine_base_url = local_engine_base_url.rstrip("/")
        search_cfg["local_url"] = f"{local_engine_base_url}/search"
        fetch_cfg["local_url"] = f"{local_engine_base_url}/document"

    # Return a dict that can be passed directly to ToolEnv; enable_tools limits the exposed tool set.
    return {
        "enable_tools": ["search", "fetch_url", "submit_answer"],
        "search": search_cfg,
        "fetch_url": fetch_cfg,
        "submit_answer": None,
    }


# ── Complete configuration for one experiment run ────────────────────────────


def check_prompt_version(benchmark: str, prompt_version: str) -> None:
    """Validate that prompt_version matches the benchmark.

    GISA uses a dedicated TSV output format and requires prompt_version="gisa";
    all other benchmarks permit only "general".
    """
    allowed = {"gisa"} if benchmark == "gisa" else {"general"}
    if prompt_version not in allowed:
        raise ValueError(
            f"benchmark={benchmark!r} requires prompt_version in {sorted(allowed)!r}, "
            f"got {prompt_version!r}."
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration required for one experiment run."""

    webswarm_config: dict
    env_config: dict
    task_config: dict
    task_ids: list | None


def build_experiment() -> ExperimentConfig:
    """Build a complete experiment configuration from the selection constants above.

    search_engine comes from this file's SEARCH_ENGINE constant. The local service endpoint
    still comes from LOCAL_ENGINE_BASE_URL in .env (call load_env_file first). Validate
    benchmark/prompt_version consistency before returning. The runner consumes the result
    directly without importing constants such as MODEL / PROVIDER / BENCHMARK individually.
    """
    search_engine = SEARCH_ENGINE.strip().lower()
    local_engine_base_url = (
        os.environ.get("LOCAL_ENGINE_BASE_URL") if search_engine == "local" else None
    )

    webswarm_config = build_webswarm_config(MODEL, PROVIDER)
    env_config = build_webswarm_env_config(
        model=MODEL,
        provider=PROVIDER,
        search_engine=search_engine,
        enable_cache=ENABLE_CACHE,
        local_engine_base_url=local_engine_base_url,
    )
    task_config = {"benchmark": BENCHMARK, "benchmark_version": BENCHMARK_VERSION}

    check_prompt_version(task_config["benchmark"], webswarm_config["prompt_version"])

    return ExperimentConfig(
        webswarm_config=webswarm_config,
        env_config=env_config,
        task_config=task_config,
        task_ids=TASK_IDS,
    )
