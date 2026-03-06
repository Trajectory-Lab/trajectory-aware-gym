from typing import Any

from ddgs import DDGS

from trajectory_aware_gym.tools.base import MCPTool


class SearchTool(MCPTool):
    name = "search"

    description = "Search the web for information."

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            }
        },
        "required": ["query"],
    }

    def run(self, **kwargs) -> dict[str, Any]:
        query = kwargs.get("query")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))

            formatted = [
                {
                    "title": r["title"],
                    "url": r["href"],
                    "snippet": r["body"],
                }
                for r in results
            ]

            return {
                "status": "success",
                "results": formatted,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
