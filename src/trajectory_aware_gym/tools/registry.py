from trajectory_aware_gym.tools.base import MCPTool


class ToolRegistry:
    """
    Stores available tools and exposes schemas for the agent.
    """

    def __init__(self):
        self._tools: dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> MCPTool:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not registered")
        return self._tools[name]

    def list_schemas(self):
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            for t in self._tools.values()
        ]
