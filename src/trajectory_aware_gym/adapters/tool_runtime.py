import asyncio
from typing import Any

import trajectory_aware_gym.mcp.tools  # noqa: F401
from trajectory_aware_gym.config import settings
from trajectory_aware_gym.mcp.server import mcp
from trajectory_aware_gym.utils.tool_logging import log_tool_call


class ToolRuntime:
    """
    Executes tool calls emitted by the agent via FastMCP.
    """

    def __init__(self, log_path: str = "logs/tool_calls.jsonl"):
        self.log_path = log_path
        self._server = mcp

    def _normalize_result(self, raw_result: Any) -> dict[str, Any]:
        if isinstance(raw_result, dict):
            return raw_result

        if hasattr(raw_result, "structured_content") and isinstance(
            raw_result.structured_content,
            dict,
        ):
            return raw_result.structured_content

        if hasattr(raw_result, "data") and isinstance(raw_result.data, dict):
            return raw_result.data

        if hasattr(raw_result, "model_dump"):
            dumped = raw_result.model_dump()
            if isinstance(dumped, dict):
                return dumped

        return {"status": "success", "result": raw_result}

    async def execute(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = tool_call["tool"]
        args = tool_call.get("arguments", {})

        try:
            tool = await self._server.get_tool(name)
        except (KeyError, ValueError) as exc:
            raw_result = {"status": "error", "error": f"Tool '{name}' not found: {exc}"}
            result = self._normalize_result(raw_result)
            log_tool_call(self.log_path, name, args, result)
            return result
        if tool is None:  # pragma: no cover - defensive guard for MCP typing surface
            raw_result = {"status": "error", "error": f"Tool '{name}' not found"}
            result = self._normalize_result(raw_result)
            log_tool_call(self.log_path, name, args, result)
            return result

        raw_result = await asyncio.wait_for(tool.run(args), timeout=settings.gem.tool_timeout)

        result = self._normalize_result(raw_result)

        log_tool_call(self.log_path, name, args, result)

        return result

    async def list_schemas(self) -> list[dict[str, Any]]:
        tools = await self._server.get_tools()  # pyright: ignore[reportAttributeAccessIssue]

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools.values()
        ]
