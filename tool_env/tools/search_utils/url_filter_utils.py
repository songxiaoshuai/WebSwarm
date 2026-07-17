"""URL leakage-prevention filter for results that are too similar to table_url."""
from urllib.parse import unquote, urlparse
import unicodedata
from difflib import SequenceMatcher


def _normalize_url_for_matching(url: str) -> str:
    """Normalize a URL for matching through decoding, Unicode normalization, lowercasing, and www removal."""
    parsed = urlparse(url)
    normalized = unicodedata.normalize(
        "NFC",
        unquote(parsed.netloc.lower().replace("www.", "")) + unquote(parsed.path) + unquote(parsed.query),
    ).strip("/")
    return normalized


def _extract_url_keywords(url: str) -> set:
    """Extract URL keywords after removing common words, punctuation, underscores, and similar noise."""
    # Remove the scheme, domain, and similar components, retaining only path keywords.
    parsed = urlparse(url)
    path = unquote(parsed.path).strip("/")
    # Replace underscores, hyphens, slashes, and similar separators with spaces.
    path = path.replace("_", " ").replace("-", " ").replace("/", " ")
    # Convert to lowercase and split into words.
    words = path.lower().split()
    # Filter common insignificant words.
    stop_words = {
        "wiki",
        "www",
        "en",
        "org",
        "com",
        "net",
        "the",
        "a",
        "an",
    }
    keywords = {w for w in words if w not in stop_words and len(w) > 1}
    return keywords


def _calculate_url_similarity(url1: str, url2: str) -> float:
    """Compute URL similarity from both character sequence and keyword overlap."""
    similarity1 = SequenceMatcher(None, url1, url2).ratio()
    keywords1 = _extract_url_keywords(url1)
    keywords2 = _extract_url_keywords(url2)
    if keywords1 or keywords2:
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        similarity2 = len(intersection) / len(union) if union else 0.0
    else:
        similarity2 = 0.0

    return max(similarity1, similarity2)


def filter_url(table_url: str, relevant_infos: list) -> list:
    """Filter search results too similar to table_url to prevent source-question leakage."""
    table_url_normalized = _normalize_url_for_matching(table_url)

    filtered_infos = []
    for info in relevant_infos:
        info_url_normalized = _normalize_url_for_matching(info.url)

        if table_url_normalized in info_url_normalized:
            continue
        if info_url_normalized in table_url_normalized:
            continue

        similarity = _calculate_url_similarity(table_url_normalized, info_url_normalized)
        if similarity >= 0.75:
            print(
                f"[INFO] [SEARCH TOOL] Filtered result matching table_url: {info.url} (similarity: {similarity:.4f})"
            )
            continue
        filtered_infos.append(info)
    relevant_infos = filtered_infos
    return relevant_infos
