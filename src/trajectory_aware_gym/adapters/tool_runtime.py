import asyncio
from collections.abc import Coroutine
from threading import Thread
from typing import Any

import trajectory_aware_gym.mcp.tools  # noqa: F401
from trajectory_aware_gym.config import settings
from trajectory_aware_gym.mcp.server import mcp
from trajectory_aware_gym.utils.tool_logging import log_tool_call


class ToolRuntime:
    """
    Executes tool calls emitted by the agent via FastMCP.
    """

    def __init__(self, log_path: str = "logs/trajectories.db"):
        self.log_path = log_path
        self._server = mcp

    def _run_sync(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        result: Any = None
        error: BaseException | None = None

        def runner() -> None:
            nonlocal result, error
            try:
                result = asyncio.run(coroutine)
            except BaseException as exc:  # pragma: no cover - defensive path
                error = exc

        thread = Thread(target=runner)
        thread.start()
        timeout = settings.gem.tool_timeout
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(f"Tool execution timed out after {timeout}s")

        if error is not None:
            raise error

        return result

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

    def execute(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = tool_call["tool"]
        args = tool_call.get("arguments", {})

        try:
            tool = self._run_sync(self._server.get_tool(name))
        except (KeyError, ValueError) as exc:
            raw_result = {"status": "error", "error": f"Tool '{name}' not found: {exc}"}
            result = self._normalize_result(raw_result)
            log_tool_call(self.log_path, name, args, result)
            return result

        raw_result = self._run_sync(tool.run(args))

        result = self._normalize_result(raw_result)

        log_tool_call(self.log_path, name, args, result)

        return result

    def list_schemas(self) -> list[dict[str, Any]]:
        tools = self._run_sync(self._server.get_tools())  # pyright: ignore[reportAttributeAccessIssue]

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools.values()
        ]
