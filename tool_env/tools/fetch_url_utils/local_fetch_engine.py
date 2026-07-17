"""
Local fetch engine that accesses a local webpage-fetching service over HTTP.

This class targets debugging and local deployments with engine="local". Its interface
matches JinaEngine so FetchURLTool can switch between default/local backends without
changing call logic.
"""

import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor


ERROR_PREFIX = "[ERROR]"


class LocalFetchEngine:
    """HTTP client for the local webpage-fetching engine with a JinaEngine-compatible interface.

    Sends HTTP GET requests to the local fetching service, whose response format is
    compatible with the Jina Reader API.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Store the local service endpoint, timeout, and retry parameters.

        Args:
            base_url: Local fetching-service endpoint.
            timeout: Timeout in seconds for one HTTP request.
            max_retries: Number of retries after failure.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # API usage statistics (thread-safe)
        self._stats_lock = threading.Lock()
        self.api_cnt = 0
        self.api_cnt_ignore_cache = 0
        self.api_token = 0  # The local service has no token billing; retain the field for compatibility.

    # ── Single-URL fetch ─────────────────────────────────────────────────

    def fetch_one(self, url: str) -> str:
        """Fetch one URL and return processed webpage text.

        Return the error string '[ERROR] Failed to fetch page.' on failure.
        """
        with self._stats_lock:
            self.api_cnt += 1
            self.api_cnt_ignore_cache += 1

        return self._request_local(url)

    def _request_local(self, url: str) -> str:
        """Fetch webpage content through the local service with retries."""
        for attempt in range(self.max_retries):
            headers = {
                "Accept": "application/json",
            }
            try:
                response = requests.post(
                    f"{self.base_url}",
                    headers=headers,
                    json={"url": url},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    return self._post_process_response(response.json())
                print(
                    f"[WARNING] [LOCAL_FETCH] Fetch failed, url: {url} | "
                    f"status_code: {response.status_code}"
                )
            except requests.exceptions.ReadTimeout:
                print(f"[WARNING] [LOCAL_FETCH] Read timeout: {url}")
            except requests.exceptions.ConnectionError:
                print(f"[WARNING] [LOCAL_FETCH] Connection error: {url}")
            except Exception as e:
                print(f"[WARNING] [LOCAL_FETCH] Fetch exception, url: {url} | error: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(1)

        print(f"[ERROR] [LOCAL_FETCH] Fetch failed after {self.max_retries} retries: {url}")
        return "[ERROR] Failed to fetch page."

    # ── Parallel batch fetching ──────────────────────────────────────────

    def fetch_many(self, urls: list[str]) -> list[str]:
        """Fetch multiple URLs in parallel while preserving input order."""
        if not urls:
            return []

        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            results = list(executor.map(self.fetch_one, urls))
        return results

    # ── Response postprocessing ──────────────────────────────────────────

    def _post_process_response(self, response: dict) -> str:
        """Parse a local-service JSON response and return webpage text matching JinaEngine's format."""
        return response.get("text", "")

    # ── State management ─────────────────────────────────────────────────

    def reset_stats(self):
        """Reset API-usage statistics."""
        with self._stats_lock:
            self.api_cnt = 0
            self.api_cnt_ignore_cache = 0
            self.api_token = 0

    @property
    def stats(self) -> dict:
        """Return API-usage statistics."""
        return {
            "api_cnt": self.api_cnt,
            "api_cnt_ignore_cache": self.api_cnt_ignore_cache,
            "api_token": self.api_token,
        }
