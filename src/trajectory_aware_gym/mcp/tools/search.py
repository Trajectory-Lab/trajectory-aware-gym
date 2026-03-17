from typing import Any

from ddgs import DDGS

from trajectory_aware_gym.mcp.server import mcp


@mcp.tool()
def search(query: str) -> dict[str, Any]:
    """Search the web for information."""
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
