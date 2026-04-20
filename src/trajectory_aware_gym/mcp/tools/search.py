import logging
from typing import Any

from ddgs import DDGS

from trajectory_aware_gym.mcp.server import mcp

# Yahoo consistently fails with `RequestError` under parallel load and Mojeek
# returns `403 Forbidden`, together producing dozens of INFO-level tracebacks
# per minute without contributing results. `ddgs` has no exclusion syntax, so
# we pass an explicit allow-list of the remaining backends. Keep in sync with
# `ddgs.engines.ENGINES["text"]`.
_SEARCH_BACKENDS = "brave,duckduckgo,google,grokipedia,wikipedia,yandex"

# `ddgs` uses `primp` for HTTP, which logs every response at INFO. `ddgs.ddgs`
# itself logs per-engine errors at INFO too — both are expected chatter in a
# meta-search and drown real failures. Raise both to WARNING at import time so
# the noise is suppressed regardless of entry point (runner, tests, notebooks).
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("ddgs.ddgs").setLevel(logging.WARNING)


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
            results = list(ddgs.text(query, max_results=5, backend=_SEARCH_BACKENDS))

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
