"""General string-normalization utilities from the original DeepWideSearch evaluation package."""


def norm_column(col: str) -> str:
    """Normalize a column name by stripping, lowercasing, and removing internal spaces."""
    return col.strip().lower().replace(" ", "")
