"""
Jina webpage cache that stores URL → webpage-content mappings on disk.

get / set rely entirely on the filesystem (filename = md5(url).txt), making them naturally
safe for concurrent use. Multiple JinaCache instances can share one cache_dir without
interference. index.json is only an optional persistent snapshot for external inspection
and does not participate in get/set decisions.

Usage:
    cache = JinaCache(cache_dir="cache/jina_cache")
    content = cache.get(url)          # HIT → str, MISS → None
    cache.set(url, content)           # Write the content file
    print(cache.stats)                # View hit-rate statistics
"""
import hashlib
import tempfile
from pathlib import Path


class JinaCache:
    """Concurrent-safe disk cache mapping each URL to a separate md5(url).txt content file."""

    def __init__(self, cache_dir: str) -> None:
        """Create the cache directory and initialize hit statistics."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "index.json"

        # Statistics
        self._hit_count = 0
        self._miss_count = 0

        print(f"[INFO] [JINA CACHE] Ready (cache_dir={self.cache_dir})")

    # ── Read/write interface ─────────────────────────────────────────────

    def get(self, url: str) -> str | None:
        """Query the cache, returning a content string on HIT and None on MISS."""
        filepath = self.cache_dir / self._url_to_filename(url)
        if not filepath.exists():
            self._miss_count += 1
            print(f"[INFO] [JINA CACHE] MISS: {url}")
            return None

        self._hit_count += 1
        print(f"[DEBUG] [JINA CACHE] HIT: {url}")
        return filepath.read_text(encoding="utf-8")

    def set(self, url: str, content: str) -> None:
        """Write to the cache by atomically writing a content file."""
        filepath = self.cache_dir / self._url_to_filename(url)

        # Atomic write: write a temporary file, then rename it to avoid partial files after a crash.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(content)
            Path(tmp_path).replace(filepath)  # Atomic rename
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    # ── Utility methods ──────────────────────────────────────────────────

    @staticmethod
    def _url_to_filename(url: str) -> str:
        """Map a URL to a unique filename using an MD5 hash."""
        return hashlib.md5(url.encode("utf-8")).hexdigest() + ".txt"

    @property
    def stats(self) -> dict:
        """Return cache-hit statistics."""
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0
        return {
            "cache_size": sum(1 for f in self.cache_dir.iterdir() if f.suffix == ".txt"),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": f"{hit_rate:.2%}",
        }
