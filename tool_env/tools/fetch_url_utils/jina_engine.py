"""
Jina Reader API engine for fetching webpage content through Jina and parsing it as text.

Supports an optional disk cache (JinaCache) controlled by enable_cache.
"""
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

from .jina_cache import JinaCache


ERROR_PREFIX = "[ERROR]"


class JinaEngine:
    """Jina Reader API client responsible for webpage fetching and parsing."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 60,
        max_retries: int = 3,
        enable_cache: bool = False,
        cache_dir: str = "cache/jina_cache",
    ) -> None:
        """Store request parameters and initialize the cache and API-usage statistics."""
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Cache
        self.enable_cache = enable_cache
        self.cache: JinaCache | None = JinaCache(cache_dir) if enable_cache else None

        # API usage statistics (thread-safe)
        self._stats_lock = threading.Lock()
        self.api_cnt = 0               # Total calls, including cache hits
        self.api_cnt_ignore_cache = 0  # Actual Jina HTTP requests, excluding cache hits
        self.api_token = 0

    # ── Single-URL fetch ─────────────────────────────────────────────────

    def fetch_one(self, url: str) -> str:
        """Fetch one URL and return processed webpage text.

        With caching enabled, check the cache first and return hits directly; on a miss,
        request Jina and write the response to the cache. On failure, return the error string
        '[ERROR] Failed to fetch page.'
        """
        with self._stats_lock:
            self.api_cnt += 1

        # 1. Check the cache.
        if self.enable_cache and self.cache is not None:
            cached = self.cache.get(url)
            if cached is not None:
                return cached

        # 2. Request Jina.
        content = self._request_jina(url)

        # 3. Cache successful responses; do not cache errors.
        if self.enable_cache and self.cache is not None and not content.startswith(ERROR_PREFIX):
            self.cache.set(url, content)

        return content

    def _request_jina(self, url: str) -> str:
        """Fetch webpage content through the Jina Reader API with retries."""
        for attempt in range(self.max_retries):
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            try:
                response = requests.get(
                    f"{self.base_url}/{url}",
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    return self._post_process_response(response.json())
                print(f"[WARNING] [JINA] Fetch failed, url: {url} | status_code: {response.status_code}")
            except requests.exceptions.ReadTimeout:
                print(f"[WARNING] [JINA] Read timeout: {url}")
            except requests.exceptions.ConnectionError:
                print(f"[WARNING] [JINA] Connection error: {url}")
            except Exception as e:
                print(f"[WARNING] [JINA] Fetch exception, url: {url} | error: {e}")

            # Wait before another retry, but not after the final attempt.
            if attempt < self.max_retries - 1:
                time.sleep(1)

        print(f"[ERROR] [JINA] Fetch failed after {self.max_retries} retries: {url}")
        return "[ERROR] Failed to fetch page."

    # ── Parallel batch fetching ──────────────────────────────────────────

    def fetch_many(self, urls: list[str]) -> list[str]:
        """Fetch multiple URLs in parallel while preserving input order.

        Args:
            urls: URLs to fetch

        Returns:
            A content list with the same length as urls
        """
        if not urls:
            return []

        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            results = list(executor.map(self.fetch_one, urls))
        return results

    # ── Response postprocessing ──────────────────────────────────────────

    def _post_process_response(self, response: dict) -> str:
        """Parse a Jina Reader API JSON response and extract webpage text."""
        # Record API usage (thread-safe).
        if "meta" in response and "usage" in response["meta"] and "tokens" in response["meta"]["usage"]:
            with self._stats_lock:
                self.api_cnt_ignore_cache += 1
                self.api_token += response["meta"]["usage"]["tokens"]

        # Defensive check
        data = response.get("data")
        if not data or not isinstance(data, dict):
            print(f"[WARNING] [JINA] Response missing 'data' field: {list(response.keys())}")
            return "[ERROR] Invalid Jina response format."

        # Build a uniform webpage-text format for subsequent truncation and summarization by FetchURLTool.
        text = ""
        if title := data.get("title", ""):
            text += f"Title: {title}\n\n"
        if url := data.get("url", ""):
            text += f"URL Source: {url}\n\n"
        if published_time := data.get("publishedTime", ""):
            text += f"Published Time: {published_time}\n\n"
        if warning := data.get("warning", ""):
            text += f"Warning: {warning}\n\n"
        if content := data.get("content", ""):
            text += f"Markdown Content:\n{content}\n\n"
        elif description := data.get("description", ""):
            text += f"Description:\n{description}\n\n"
        return text

    # ── State management ─────────────────────────────────────────────────

    def reset_stats(self):
        """Reset API-usage statistics without clearing the cache."""
        with self._stats_lock:
            self.api_cnt = 0
            self.api_cnt_ignore_cache = 0
            self.api_token = 0

    def flush_cache(self):
        """Print cache statistics; index.json no longer needs to be flushed."""
        if self.cache is not None:
            print(f"[INFO] [JINA ENGINE] Cache stats: {self.cache.stats}")

    @property
    def stats(self) -> dict:
        """Return API-usage statistics, including cache statistics."""
        s = {
            "api_cnt": self.api_cnt,
            "api_cnt_ignore_cache": self.api_cnt_ignore_cache,
            "api_token": self.api_token,
        }
        if self.cache is not None:
            s["cache"] = self.cache.stats
        return s
