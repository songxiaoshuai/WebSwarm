"""Search-result structures and formatting utilities that convert engine results to observation text."""

from pydantic import BaseModel, Field


class SearchResultInfo(BaseModel):
    """Structure representing one summarized search result."""

    id: int = Field(..., description="Search result index")
    title: str = Field(..., description="Search result title")
    url: str = Field(..., description="Search result URL")
    site_name: str = Field(..., description="Search result site name")
    date: str = Field(..., description="Search result date")
    snippet: str = Field(..., description="Search result snippet")
    context: str | None = Field(None, description="Full search result content (not fetched by default)")


def format_search_results(queries: list[str], query2search_results: dict[str, list["SearchResultInfo"]]) -> str:
    """Format search results for direct return to the agent without further summarization.

    Args:
        queries (list[str]): Query list
        query2search_results (dict[str, list["SearchResultInfo"]]): Search results grouped by query

    Returns:
        str: Formatted search results
    """
    ret = ""
    for query in queries:
        search_results = query2search_results[query]
        ret += f"### Search for query '{query}' got {len(search_results)} results: \n"
        for i, result in enumerate(search_results):
            ret += f"#### Page {i+1}\n"
            ret += f"[Title] {result.title}\n"
            ret += f"[URL] {result.url}\n"
            if result.site_name:
                ret += f"[Source] {result.site_name}\n"
            ret += f"[Snippet] {result.snippet}\n"
            ret += "---\n"
        ret += "============================\n\n"
    return ret
