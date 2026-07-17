"""URL-fetching tool that retrieves one webpage and uses WebSummaryTool to extract target information."""

import os
import tiktoken
from copy import deepcopy
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor

from .base_tool import BaseTool

# URL-fetching implementation
from .fetch_url_utils.jina_engine import JinaEngine
from .fetch_url_utils.local_fetch_engine import LocalFetchEngine
from .fetch_url_utils.summary_page import WebSummaryTool

JINA_BASE_URL_ENV = "JINA_BASE_URL"
JINA_API_KEY_ENV = "JINA_API_KEY"


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Set it in the environment before running.")
    return value

# Cache the tiktoken encoder to avoid reinitializing it on every call.
_tiktoken_encoding = tiktoken.get_encoding("cl100k_base")

def truncate_to_tokens(text: str, max_tokens) -> str:
    """Truncate raw webpage text by approximate token count to keep summary input bounded."""
    tokens = _tiktoken_encoding.encode(text)
    if len(tokens) <= max_tokens:
        print(f"[INFO] [FETCH URL] Estimated page tokens: {len(tokens)}")
        return text
    print(f"[INFO] [FETCH URL] Truncated page to {max_tokens} tokens (estimated)")
    return _tiktoken_encoding.decode(tokens[:max_tokens]) + "...(truncated)"


class FetchURLTool(BaseTool):
    """Single-URL fetching tool for retrieving and summarizing webpage content.

    Supported configuration fields:
        engine: str, fetching engine: "default" (Jina API) or "local" (local fetching service)
        local_url: str, local fetching-service endpoint, required when engine="local"
        enable_cache: bool, whether to enable the disk cache; applies only to engine="default"
        cache_dir: str, cache directory; applies only to engine="default"
        web_summary: dict, webpage-summarization tool configuration
    """

    def __init__(self, tool_config: dict) -> None:
        super().__init__()
        self.max_tokens = tool_config.get("max_tokens", 95000)

        enable_cache = tool_config.get("enable_cache", False)
        engine_type = tool_config.get("engine", "default")

        if engine_type == "local":
            # Use the local webpage-fetching service.
            local_url = tool_config.get("local_url")
            assert local_url is not None, "engine='local' requires 'local_url' in tool_config"
            self.jina_engine = LocalFetchEngine(
                base_url=local_url,
                timeout=tool_config.get("timeout", 600),
                max_retries=tool_config.get("max_retries", 3),
            )
            print(f"[INFO] [FETCH URL] Using local fetch engine, url={local_url}")
        else:
            # Use the Jina Reader API engine (default).
            jina_url = _get_required_env(JINA_BASE_URL_ENV)
            jina_api_key = _get_required_env(JINA_API_KEY_ENV)
            cache_dir = tool_config.get("cache_dir", "cache/jina_cache")
            self.jina_engine = JinaEngine(
                base_url=jina_url,
                api_key=jina_api_key,
                timeout=60,
                max_retries=3,
                enable_cache=enable_cache,
                cache_dir=cache_dir,
            )
            print(f"[INFO] [FETCH URL] Using default Jina engine")
        
        # Initialize the summarization tool.
        self.web_summary_tool = WebSummaryTool(tool_config=deepcopy(tool_config["web_summary"]))

        # keep_links: read from tool_config; controls whether summaries append a list of relevant links.
        # The caller sets it through env_config["fetch_url"]["keep_links"] when constructing ToolEnv;
        # the agent does not need to pass it in tool_call parameters.
        self.keep_links: bool = bool(tool_config.get("keep_links", False))

        # State tracking
        self.fetch_url_round = 0
        self.original_question = None

    # ── Backward-compatible properties delegated to the engine ───────────

    @property
    def jina_api_cnt(self) -> int:
        return self.jina_engine.api_cnt

    @property
    def jina_api_cnt_ignore_cache(self) -> int:
        return self.jina_engine.api_cnt_ignore_cache

    @property
    def jina_api_token(self) -> int:
        return self.jina_engine.api_token

    def get_info(self) -> dict:
        """Return tool metadata."""
        return {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": (
                    "Fetch content from a single URL, and extract the content you want "
                    "by an intelligent agent."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "A single URL to fetch.",
                        },
                        "goal": {
                            "type": "string",
                            "description": "The goal of the content you want to extract from the web page",
                        },
                    },
                    "required": ["url", "goal"],
                },
            },
        }
    
    def execute(self, tool_params: object) -> dict:
        """Fetch and summarize a URL."""
        self.fetch_url_round += 1

        if (
            not isinstance(tool_params, dict)
            or "url" not in tool_params
            or "goal" not in tool_params
        ):
            return {
                "type": "tool_result",
                "content": (
                    "Invalid input for fetch_url. Expected dict with 'url' and "
                    "'goal' keys. Only single url is allowed."
                ),
                "is_error": True,
            }

        url = tool_params["url"]
        goal = tool_params["goal"]
        if not isinstance(url, str) or not url.strip():
            return {
                "type": "tool_result",
                "content": "Invalid 'url'. Expected a non-empty string. Only single url is allowed.",
                "is_error": True,
            }
        if not isinstance(goal, str) or not goal.strip():
            return {
                "type": "tool_result",
                "content": "Invalid 'goal'. Expected a non-empty string.",
                "is_error": True,
            }

        # The public tool accepts one URL; internally retain a list to reuse the engine interface.
        urls = [url.strip()]
        fetch_results = self.jina_engine.fetch_many(urls)
        contents = [truncate_to_tokens(r, max_tokens=self.max_tokens) for r in fetch_results]

        assert self.original_question is not None, "Original question is not set in tool state"

        valid_indices = [i for i, r in enumerate(fetch_results) if not r.startswith("[ERROR]")]
        if not valid_indices:
            return {
                "type": "tool_result",
                "content": "URL failed to fetch. Please try a different URL.",
                "is_error": True,
            }

        valid_urls = [urls[i] for i in valid_indices]
        valid_contents = [contents[i] for i in valid_indices]

        # Summarize successfully fetched pages; explicitly mark failed URLs in the result.
        summary_results_str = ""
        with ThreadPoolExecutor(max_workers=len(valid_urls)) as executor:
            summary_results = list(
                executor.map(
                    self.web_summary_tool.summary_one_page,
                    valid_urls,
                    repeat(goal),
                    valid_contents,
                    repeat(self.original_question),
                    repeat(self.keep_links),
                )
            )
            for u, summary_result in zip(valid_urls, summary_results):
                summary_results_str += f"[URL]: {u}\n[INFO FOUND]: {summary_result}\n\n"

        for i in range(len(urls)):
            if i not in valid_indices:
                summary_results_str += f"[URL]: {urls[i]}\n[INFO FOUND]: [FETCH FAILED]\n\n"

        return {"type": "tool_result", "content": summary_results_str, "is_error": False}
    
    def update_original_question(self, original_question: str):
        """Update the original question in tool state."""
        self.original_question = original_question
    
    def clear_tool_state(self):
        """Clear tool state, including API counts and token statistics."""
        self.fetch_url_round = 0
        self.jina_engine.reset_stats()
        self.original_question = None

    def return_tool_state(self) -> dict:
        """Return tool state, including API counts and token statistics."""
        return {
            "fetch_url_round": self.fetch_url_round,
            "jina_api_cnt": self.jina_engine.api_cnt,
            "jina_api_cnt_ignore_cache": self.jina_engine.api_cnt_ignore_cache,
            "jina_api_token": self.jina_engine.api_token,
            "original_question": self.original_question
        }
