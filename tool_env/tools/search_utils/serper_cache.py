"""
Serper search cache that stores request-parameter → search-response JSON mappings on disk.

get / set rely entirely on the filesystem (filename = md5(params_str).json), making them
naturally safe for concurrent use. Multiple SerperCache instances can share one cache_dir
without interference. index.json no longer participates in get/set decisions.

The cache key is determined by the JSON string for Serper request params, so different
queries, languages, or date_range values produce independent entries.

Usage:
    cache = SerperCache(cache_dir="cache/serper_cache")
    resp = cache.get(params_str)          # HIT → dict, MISS → None
    cache.set(params_str, response_dict)  # Write the response file
    print(cache.stats)                    # View hit-rate statistics
"""
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Any


class SerperCache:
    """Concurrent-safe disk cache mapping params to separate md5(key).json response files."""

    def __init__(self, cache_dir: str) -> None:
        """Create the cache directory and initialize hit statistics."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self._hit_count = 0
        self._miss_count = 0

        print(
            f"[INFO] [SERPER CACHE] Ready "
            f"(cache_dir={self.cache_dir})"
        )

    # ── Read/write interface ─────────────────────────────────────────────

    def get(self, key: str) -> dict | None:
        """Query the cache, returning a response dict on HIT and None on MISS."""
        filepath = self.cache_dir / self._key_to_filename(key)
        if not filepath.exists():
            self._miss_count += 1
            print(f"[INFO] [SERPER CACHE] MISS: {key}")
            return None

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARNING] [SERPER CACHE] Cache file corrupted ({key}): {e}")
            self._miss_count += 1
            return None

        self._hit_count += 1
        print(f"[DEBUG] [SERPER CACHE] HIT: {key}")
        return data

    def set(self, key: str, response: Any) -> None:
        """Write to the cache by atomically writing a response JSON file."""
        filepath = self.cache_dir / self._key_to_filename(key)
        content = json.dumps(response, ensure_ascii=False)

        # Atomic write: write a temporary file, then rename it.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(content)
            Path(tmp_path).replace(filepath)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    # ── Utility methods ──────────────────────────────────────────────────

    @staticmethod
    def _key_to_filename(key: str) -> str:
        return hashlib.md5(key.encode("utf-8")).hexdigest() + ".json"

    @property
    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0
        return {
            "cache_size": sum(1 for f in self.cache_dir.iterdir() if f.suffix == ".json" and f.name != "index.json"),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": f"{hit_rate:.2%}",
        }
