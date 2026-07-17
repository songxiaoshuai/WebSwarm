"""Serper search-engine wrapper for requests, result caching, and display-item extraction."""

import json
import time
import threading
import langid
import requests
from typing import Any

from .search_util import SearchResultInfo
from .serper_cache import SerperCache


class SerperSearchEngine:
    """Serper HTTP client with search/extract interfaces aligned with LocalSearchEngine."""

    def __init__(
        self,
        url: str,
        api_key: str,
        max_results: int,
        ignored_urls: list[str] | None = None,
        max_retries: int = 3,
        enable_cache: bool = False,
        cache_dir: str = "cache/serper_cache",
    ) -> None:
        """Store Serper request configuration and initialize the cache and API-usage statistics."""
        self.engine_name = "SERPER"
        self.url = url
        self.api_key = api_key
        self.max_results = max_results
        self.ignored_urls = ignored_urls or []
        self.max_retries = max_retries

        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        # Cache
        self.enable_cache = enable_cache
        self.cache: SerperCache | None = (
            SerperCache(cache_dir) if enable_cache else None
        )

        # API usage statistics (thread-safe)
        self._stats_lock = threading.Lock()
        self.api_cnt = 0               # Total calls, including cache hits
        self.api_cnt_ignore_cache = 0  # Actual Serper HTTP requests, excluding cache hits

    def _call_request(self, query: str, params: str, timeout: int) -> Any:
        """Call the Serper HTTP interface with retries.

        Args:
            query (str): Query to search
            params (str): Search parameters
            timeout (int): Timeout

        Returns:
            Any: Search result
        """
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
                search_results = response.json()
                return search_results
            except requests.exceptions.Timeout as e:
                print(
                    f"[WARNING] [SERPER] [{self.engine_name}] Attempt {error_cnt+1} failed: timeout ({timeout}s), query: {query} error: {e}"
                )
            except requests.exceptions.RequestException as e:
                print(
                    f"[WARNING] [SERPER] [{self.engine_name}] Attempt {error_cnt+1} failed: error={e}, params={params}"
                )
            error_cnt += 1
            time.sleep(5)
        print(
            f"[ERROR] [SERPER] [{self.engine_name}] query: {query} failed after {error_cnt} retries, skipping."
        )
        return None

    def _get_params(self, query: str, date_range: str | None = None) -> str:
        """Build Serper request parameters from query language and time range.

        Args:
            query (str): Content to search
            date_range (str | None): Time range

        Returns:
            str: Search parameters
        """
        if langid.classify(query)[0] == "zh":
            params = {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
        else:
            params = {
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
            }
        if date_range:
            if date_range not in ["qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y"]:
                raise ValueError(f"Invalid date_range: {date_range}")
            params["tbs"] = date_range
        return json.dumps(params)

    def search(
        self,
        query: str,
        timeout: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict | None:
        """Execute one Serper search, reading/writing the disk cache when needed.

        Args:
            query: Search-query string.
            timeout: Request timeout in seconds.
            start_date: Time-range filter (qdr:h/d/w/m/y).
            end_date: Currently unsupported and ignored.

        Returns:
            Search-result dict, or None on failure.
        """
        params = self._get_params(query, date_range=start_date)
        print(f"[DEBUG] [SERPER] Search params: {params}")

        with self._stats_lock:
            self.api_cnt += 1

        # 1. Check the cache.
        if self.enable_cache and self.cache is not None:
            cached = self.cache.get(params)
            if cached is not None:
                print(f"[INFO] [SERPER] Cache hit for query: {query}")
                return cached

        # 2. Request Serper (actual HTTP call).
        with self._stats_lock:
            self.api_cnt_ignore_cache += 1
        result = self._call_request(query, params, timeout)

        # 3. Cache successful responses; do not cache None.
        if (
            self.enable_cache
            and self.cache is not None
            and result is not None
        ):
            self.cache.set(params, result)

        print(f"[INFO] [SERPER] Search completed: {query}")
        return result

    # ── State management ─────────────────────────────────────────────────

    def reset_stats(self):
        """Reset API-usage statistics without clearing the cache."""
        with self._stats_lock:
            self.api_cnt = 0
            self.api_cnt_ignore_cache = 0

    def extract_relevant_info(self, search_results: Any) -> list[SearchResultInfo]:
        """Extract relevant information from search results.
        
        Args:
            search_results: Raw results returned by the search engine
            
        Returns:
            A list of SearchResultInfo objects
        """
        useful_info: list[SearchResultInfo] = []
        if search_results is None:
            return useful_info

        if "organic" not in search_results:
            print(f"[WARNING] [SERPER] [{self.engine_name}] No organic results found.")
            return useful_info

        # Also filter benchmark/data-source results so search results do not expose the source questions directly.
        blocked_keywords = ["widesearch", "huggingface"]
        results: list[dict[str, Any]] = search_results["organic"][: self.max_results]
        for id, result in enumerate(results):
            url = result.get("url", result.get("link", ""))
            if any(keyword in url.lower() for keyword in self.ignored_urls):
                print(f"[DEBUG] [SERPER] [{self.engine_name}] Filtered URL: {url}")
                continue
            # Filter results containing blocked terms in the URL, title, or snippet.
            combined = (url + result.get("title", "") + result.get("snippet", "")).lower()
            if any(kw in combined for kw in blocked_keywords):
                print(f"[DEBUG] [SERPER] [{self.engine_name}] Filtered result with blocked keyword: {url}")
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
        print(f"[INFO] [SERPER] [{self.engine_name}] Total results extracted: {len(useful_info)}")
        return useful_info
