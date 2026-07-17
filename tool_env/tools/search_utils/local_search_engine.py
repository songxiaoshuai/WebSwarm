"""
Local search engine that accesses a local search service over HTTP.

This class targets debugging and local deployments with engine="local". Its interface
matches SerperSearchEngine so WebSearchTool can switch between default/local backends.
"""

import json
import time
import threading
import requests
from typing import Any

from .search_util import SearchResultInfo


class LocalSearchEngine:
    """HTTP client for the local search engine with a SerperSearchEngine-compatible interface.

    Sends HTTP POST requests to the local search service and returns Serper-compatible results.
    """

    def __init__(
        self,
        url: str,
        max_results: int,
        ignored_urls: list[str] | None = None,
        max_retries: int = 3,
    ) -> None:
        """Store local search-service configuration and initialize API-usage statistics.

        Args:
            url: Local search-service endpoint.
            max_results: Maximum results returned per search.
            ignored_urls: URL keywords to filter.
            max_retries: Number of retries after failure.
        """
        self.engine_name = "LOCAL_SEARCH"
        self.url = url
        self.max_results = max_results
        self.ignored_urls = ignored_urls or []
        self.max_retries = max_retries

        self.headers = {
            "Content-Type": "application/json",
        }

        # API usage statistics (thread-safe)
        self._stats_lock = threading.Lock()
        self.api_cnt = 0
        self.api_cnt_ignore_cache = 0

    def _call_request(self, query: str, params: str, timeout: int) -> Any:
        """Call the local search service with retries."""
        error_cnt = 0
        while error_cnt < self.max_retries:
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    data=params,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                print(
                    f"[WARNING] [LOCAL_SEARCH] Attempt {error_cnt+1} failed: timeout ({timeout}s), "
                    f"query: {query} error: {e}"
                )
            except requests.exceptions.RequestException as e:
                print(
                    f"[WARNING] [LOCAL_SEARCH] Attempt {error_cnt+1} failed: error={e}, params={params}"
                )
            error_cnt += 1
            time.sleep(5)
        print(f"[ERROR] [LOCAL_SEARCH] query: {query} failed after {error_cnt} retries, skipping.")
        return None

    def _get_params(self, query: str) -> str:
        """Build JSON request parameters for the local search service."""
        params: dict = {"q": query, "k": self.max_results}
        return json.dumps(params)

    def search(
        self,
        query: str,
        timeout: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict | None:
        """Execute a local search and return a Serper-compatible result dict.

        Args:
            query: Search-query string
            timeout: Request timeout in seconds
            start_date: Optional time-range filter
            end_date: Unsupported time-range filter; ignored

        Returns:
            A dict containing an "organic" field, or None on failure.
        """
        params = self._get_params(query)
        print(f"[DEBUG] [LOCAL_SEARCH] Search params: {params}")

        with self._stats_lock:
            self.api_cnt += 1
            self.api_cnt_ignore_cache += 1

        result = self._call_request(query, params, timeout)
        print(f"[INFO] [LOCAL_SEARCH] Search completed: {query}")
        return result

    def extract_relevant_info(self, search_results: Any) -> list[SearchResultInfo]:
        """Extract a uniform SearchResultInfo list from local search results."""
        useful_info: list[SearchResultInfo] = []
        if search_results is None:
            return useful_info

        if "results" not in search_results:
            print(f"[WARNING] [LOCAL_SEARCH] [{self.engine_name}] No results found.")
            return useful_info

        results: list[dict[str, Any]] = search_results["results"][: self.max_results]
        for id, result in enumerate(results):
            url = result.get("url", result.get("link", ""))
            if any(keyword in url.lower() for keyword in self.ignored_urls):
                print(f"[DEBUG] [LOCAL_SEARCH] [{self.engine_name}] Filtered URL: {url}")
                continue
            useful_info.append(
                SearchResultInfo(
                    id=id + 1,
                    title=result.get("title", ""),
                    url=url,
                    site_name=result.get("source", ""),
                    date=str(result.get("date", "")),
                    snippet=result.get("snippet", ""),
                    context="",
                )
            )
        print(f"[INFO] [LOCAL_SEARCH] [{self.engine_name}] Total results extracted: {len(useful_info)}")
        return useful_info

    # ── State management ─────────────────────────────────────────────────

    def reset_stats(self):
        """Reset API-usage statistics."""
        with self._stats_lock:
            self.api_cnt = 0
            self.api_cnt_ignore_cache = 0
