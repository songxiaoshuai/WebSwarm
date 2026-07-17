"""
Manual tool-testing entry point.

This file is not part of production execution. It is used only for temporary manual checks
of tool schemas and execute return formats for search / fetch_url / submit_answer.
"""
import json
import os

from .search_tool import WebSearchTool
from .terminate_tool import TerminateTool
from .fetch_url_tool import FetchURLTool


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Set it in the environment before running.")
    return value



def test_web_search_tool():
    """Test the search tool."""
    # Configure search parameters.
    config = {
        "engine": "default",
        "max_results": 5,
        "max_search_round": 20,
        "max_search_num": 100,
        "ignored_urls": None,
    }
    
    # Initialize the tool and retrieve its information.
    tool = WebSearchTool(config)
    print(json.dumps(tool.get_info(), indent=2))
    
    # Run the search test.
    result = tool.execute({"query": "history of artificial intelligence", "date_range": "qdr:y"})
    print(result)


def test_fetch_url_tool():
    """Test the URL-fetching tool."""
    # Configure URL-fetching parameters.
    config = {
        "max_tokens": 95000,
        "web_summary": {
            "model_provider": _get_required_env("WEBSWARM_PROVIDER"),
            "model_name": _get_required_env("WEBSWARM_MODEL"),
        }
    }
    
    # Initialize the tool and retrieve its information.
    tool = FetchURLTool(tool_config=config)
    print(json.dumps(tool.get_info(), indent=2))
    
    # Run the URL-fetching test.
    tool.update_original_question("test question")
    result = tool.execute({
        "url": _get_required_env("TOOL_TEST_FETCH_URL"),
        "goal": "Extract information related to artificial intelligence from the web page",
    })
    print(result)
    print(f"JINA API calls: {tool.jina_api_cnt}, Tokens: {tool.jina_api_token}")


def test_terminate_tool():
    """Test the termination tool."""
    # Initialize the tool and retrieve its information.
    tool = TerminateTool({})
    print(json.dumps(tool.get_info(), indent=2))
    
    # Run the termination test.
    result = tool.execute({"answer": "The task is completed successfully."})
    print(result)


# Uncomment the functions below to test each tool.
if __name__ == "__main__":
    # test_web_search_tool()
    # test_fetch_url_tool()
    # test_terminate_tool()
    pass
