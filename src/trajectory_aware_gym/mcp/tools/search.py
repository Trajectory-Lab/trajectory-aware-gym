from typing import Any

from ddgs import DDGS

from trajectory_aware_gym.mcp.server import mcp


@mcp.tool()
def search(query: str) -> dict[str, Any]:
    """Search the web and return up to 5 results with titles, URLs, and snippets.

    Returns {"status": "success", "results": [{"title": ..., "url": ...,
    "snippet": ...}, ...]} or {"status": "error", "error": "..."}.
    Use specific, keyword-rich queries for best results. Issue multiple
    searches to triangulate facts from different sources.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        formatted = [
            {
                "title": result["title"],
                "url": result["href"],
                "snippet": result["body"],
            }
            for result in results
        ]

        return {
            "status": "success",
            "results": formatted,
        }

    except Exception as error:
        return {
            "status": "error",
            "error": str(error),
        }
