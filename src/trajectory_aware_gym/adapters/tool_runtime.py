from typing import Any

from trajectory_aware_gym.tools.registry import ToolRegistry
from trajectory_aware_gym.utils.tool_logging import log_tool_call


class ToolRuntime:
    """
    Executes tool calls emitted by the agent.
    """

    def __init__(self, registry: ToolRegistry, log_path: str = "logs/tool_calls.jsonl"):
        self.registry = registry
        self.log_path = log_path

    def execute(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = tool_call["tool"]
        args = tool_call.get("arguments", {})

        tool = self.registry.get(name)

        result = tool.run(**args)

        log_tool_call(self.log_path, name, args, result)

        return result
