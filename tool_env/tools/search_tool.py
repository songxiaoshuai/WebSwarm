"""Web-search tool encapsulating single-query search, call limits, and search-state statistics."""

import os
import threading

from .base_tool import BaseTool
from .search_utils.search_util import format_search_results
from .search_utils.serper_engine import SerperSearchEngine
from .search_utils.local_search_engine import LocalSearchEngine
from .search_utils.url_filter_utils import filter_url


SERPER_BASE_URL_ENV = "SERPER_BASE_URL"
SERPER_API_KEY_ENV = "SERPER_API_KEY"


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Set it in the environment before running.")
    return value


class WebSearchTool(BaseTool):
    """
    Search tool with a single-query interface.

    Required config:
        max_results: int, number of results returned per search
        max_search_round: int, maximum search rounds
        max_search_num: int, maximum searches (API calls)
        ignored_urls: str, URLs to ignore
        engine: str, search engine: "default" (Serper API) or "local" (local FAISS index)
    """

    def __init__(self, tool_config) -> None:
        super().__init__()
        
        # Configuration and basic parameters
        self.max_results = tool_config["max_results"]
        assert "max_search_round" in tool_config or "max_search_num" in tool_config, "must set at least one of max_search_round or max_search_num"
        self.max_search_round = tool_config.get("max_search_round", None)
        self.max_search_num = tool_config.get("max_search_num", None)
        self.tool_timeout = 600
        
        # Initialize the search engine selected by the engine field.
        engine_type = tool_config.get("engine", "default")
        if engine_type == "local":
            local_url = tool_config.get("local_url")
            assert local_url is not None, "engine='local' requires 'local_url' in tool_config"
            self.engine = LocalSearchEngine(
                url=local_url,
                max_results=self.max_results,
                ignored_urls=tool_config.get("ignored_urls") or [],
            )
            print(f"[INFO] [SEARCH TOOL] Using local search engine, url={local_url}")
        else:
            # engine="default" uses the Serper API.
            serper_url = _get_required_env(SERPER_BASE_URL_ENV)
            serper_api_key = _get_required_env(SERPER_API_KEY_ENV)
            enable_cache = tool_config.get("enable_cache", False)
            cache_dir = tool_config.get("cache_dir", "cache/serper_cache")
            self.engine = SerperSearchEngine(
                url=serper_url,
                api_key=serper_api_key,
                max_results=self.max_results,
                ignored_urls=tool_config.get("ignored_urls") or [],
                enable_cache=enable_cache,
                cache_dir=cache_dir,
            )
            print(f"[INFO] [SEARCH TOOL] Using default Serper engine")

        # Search-state tracking
        self._stats_lock = threading.Lock()
        self.cur_search_round = 0
        self.search_api_cnt = 0
        self.search_query_history: list[str] = []
        self.table_url = None


    def get_info(self) -> dict:
        """Return tool information."""
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search for relevant information on the web with a single query. "
                    "Important note: Choose query language based on information source type."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A single search query string. "
                                "Language selection guide: choose the language that can obtain "
                                "the most relevant results from the primary information sources."
                            ),
                        },
                        "date_range": {
                            "type": "string",
                            "description": "Time range (optional), default any time.",
                            "enum": ["qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y"],
                        },
                    },
                    "required": ["query"],
                },
            },
        }


    def _search_one_query(self, query: str, timeout: int, start_date: str | None = None, end_date: str | None = None) -> tuple[str, list[dict], str | None]:
        """Execute a single-query search."""
        with self._stats_lock:
            # Repeated queries are currently allowed so the model can revisit the same keywords deliberately.
            self.search_api_cnt += 1
            self.search_query_history.append(query)

        # Execute the search with network I/O outside the lock.
        response = self.engine.search(query, timeout, start_date, end_date)
        if response is None:
            print(f"[ERROR] [SEARCH TOOL] Search failed: {query}")
            return query, [], f"Search failed for query: {query}"

        # Extract and filter information.
        relevant_infos = self.engine.extract_relevant_info(search_results=response)
        if self.table_url:
            relevant_infos = filter_url(self.table_url, relevant_infos)

        return query, relevant_infos, None

    def execute(self, tool_params: dict) -> dict:
        """Execute the search tool."""
        if not isinstance(tool_params, dict) or "query" not in tool_params:
            return {
                "type": "tool_result",
                "content": (
                    "Invalid input for search. Expected dict with 'query' key "
                    "(string). Only single query is allowed."
                ),
                "is_error": True,
            }
        query = tool_params["query"]
        if not isinstance(query, str) or not query.strip():
            return {
                "type": "tool_result",
                "content": "Invalid 'query'. Expected a non-empty string. Only single query is allowed.",
                "is_error": True,
            }

        start_date = tool_params.get("date_range")
        if start_date is not None and start_date not in [
            "qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y",
        ]:
            return {
                "type": "tool_result",
                "content": (
                    f"Invalid date_range: {start_date}. "
                    f"Expected qdr:h, qdr:d, qdr:w, qdr:m, qdr:y."
                ),
                "is_error": True,
            }
        end_date = None

        self.cur_search_round += 1
        
        # Check the search-count limit.
        if (self.max_search_round is not None and self.cur_search_round > self.max_search_round) or (self.max_search_num is not None and self.search_api_cnt > self.max_search_num):
            error_text = "The maximum search limit is exceeded. You are not allowed to search. Please directly organize the existing information and produce the final answer."
            return {"type": "tool_result", "content": error_text, "is_error": True}

        query_text = query.strip()
        _, search_results, error_msg = self._search_one_query(
            query_text, self.tool_timeout, start_date, end_date
        )
        
        # Check for valid results.
        if len(search_results) == 0:
            error_text = error_msg or "Search completed, but no valid results were obtained."
            return {"type": "tool_result", "content": error_text, "is_error": True}
        
        # Format and return the results.
        result = format_search_results([query_text], {query_text: search_results})
        return {"type": "tool_result", "content": result, "is_error": False}


    def clear_tool_state(self):
        """Clear tool state: search history, counters, and engine API statistics."""
        self.cur_search_round = 0
        self.search_api_cnt = 0
        self.search_query_history = []
        self.table_url = None
        self.engine.reset_stats()
        print("[INFO] [SEARCH TOOL] Tool state cleared.")

    def return_tool_state(self) -> dict:
        """Return tool state, including search history and counters."""
        return {
            "cur_search_round": self.cur_search_round,
            "search_api_cnt": self.search_api_cnt,
            "search_api_cnt_ignore_cache": self.engine.api_cnt_ignore_cache,
            "search_query_history": self.search_query_history,
            "table_url": self.table_url
        }
