from trajectory_aware_gym.tools.search import SearchTool


def test_search_tool():
    tool = SearchTool()

    result = tool.run(query="OpenAI")

    assert result["status"] == "success"
    assert len(result["results"]) > 0
