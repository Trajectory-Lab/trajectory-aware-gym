from trajectory_aware_gym.mcp.tools.search import search


class _FakeDDGS:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query: str, max_results: int, backend: str):
        assert query == "OpenAI"
        assert max_results == 5
        assert "yahoo" not in backend
        assert "mojeek" not in backend
        return [
            {
                "title": "OpenAI",
                "href": "https://openai.com",
                "body": "AI research and products.",
            }
        ]


def test_search_tool(monkeypatch):
    monkeypatch.setattr("trajectory_aware_gym.mcp.tools.search.DDGS", _FakeDDGS)

    result = search.fn(query="OpenAI")

    assert result["status"] == "success"
    assert len(result["results"]) > 0


class _FailingDDGS:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query: str, max_results: int, backend: str):
        raise RuntimeError("network unavailable")


def test_search_tool_returns_error_on_exception(monkeypatch):
    monkeypatch.setattr("trajectory_aware_gym.mcp.tools.search.DDGS", _FailingDDGS)

    result = search.fn(query="anything")

    assert result["status"] == "error"
    assert "network unavailable" in result["error"]
