import asyncio
from collections.abc import Coroutine
from threading import Thread
from typing import Any

import trajectory_aware_gym.mcp.tools  # noqa: F401
from trajectory_aware_gym.mcp.server import mcp
from trajectory_aware_gym.utils.tool_logging import log_tool_call


class ToolRuntime:
    """
    Executes tool calls emitted by the agent via FastMCP.
    """

    def __init__(self, log_path: str = "logs/tool_calls.jsonl"):
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
        thread.join()

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

        raw_result = self._run_sync(self._server.call_tool(name, args))
        result = self._normalize_result(raw_result)

        log_tool_call(self.log_path, name, args, result)

        return result

    def list_schemas(self) -> list[dict[str, Any]]:
        raw_schemas = self._run_sync(self._server.list_tools())

        normalized_schemas: list[dict[str, Any]] = []
        for schema in raw_schemas:
            if all(hasattr(schema, field) for field in ("name", "description", "parameters")):
                normalized_schemas.append(
                    {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": schema.parameters,
                    }
                )
                continue

            if isinstance(schema, dict):
                normalized_schemas.append(schema)
                continue

            if hasattr(schema, "model_dump"):
                dumped = schema.model_dump()
                if isinstance(dumped, dict):
                    normalized_schemas.append(
                        {
                            "name": dumped.get("name"),
                            "description": dumped.get("description"),
                            "parameters": dumped.get("parameters"),
                        }
                    )
                    continue

            normalized_schemas.append({"tool": str(schema)})

        return normalized_schemas
