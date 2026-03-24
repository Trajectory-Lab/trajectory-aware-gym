"""Unit tests for ToolRuntime adapter."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trajectory_aware_gym.adapters.tool_runtime import ToolRuntime


@pytest.fixture
def runtime(tmp_path):
    return ToolRuntime(log_path=str(tmp_path / "tool_calls.jsonl"))


# ---------------------------------------------------------------------------
# _normalize_result
# ---------------------------------------------------------------------------


class TestNormalizeResult:
    """Cover all branches of _normalize_result."""

    @pytest.mark.parametrize(
        ("raw_input", "expected"),
        [
            ({"a": 1}, {"a": 1}),
            ({}, {}),
            ("raw string", {"status": "success", "result": "raw string"}),
            (None, {"status": "success", "result": None}),
            (42, {"status": "success", "result": 42}),
        ],
    )
    def test_direct_values(self, runtime, raw_input, expected):
        assert runtime._normalize_result(raw_input) == expected

    def test_structured_content_attribute(self, runtime):
        class Result:
            structured_content = {"key": "value"}

        assert runtime._normalize_result(Result()) == {"key": "value"}

    def test_structured_content_non_dict_skipped(self, runtime):
        class Result:
            structured_content = "not a dict"

        result = runtime._normalize_result(Result())
        assert result["status"] == "success"

    def test_data_attribute(self, runtime):
        class Result:
            data = {"output": "hello"}

        assert runtime._normalize_result(Result()) == {"output": "hello"}

    def test_data_attribute_non_dict_skipped(self, runtime):
        class Result:
            data = 42

        result = runtime._normalize_result(Result())
        assert result["status"] == "success"

    def test_model_dump_pydantic_like(self, runtime):
        class Result:
            def model_dump(self) -> dict[str, Any]:
                return {"dumped": True}

        assert runtime._normalize_result(Result()) == {"dumped": True}

    def test_model_dump_non_dict_falls_through(self, runtime):
        class Result:
            def model_dump(self) -> list:
                return [1, 2, 3]

        result = runtime._normalize_result(Result())
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_execute_calls_tool_and_logs(self, runtime, tmp_path):
        mock_tool = MagicMock()
        mock_tool.run = AsyncMock()
        mock_result = {"status": "success", "output": "42"}
        mock_tool.run.return_value = mock_result

        with patch.object(runtime._server, "get_tool", new=AsyncMock(return_value=mock_tool)):
            result = await runtime.execute(
                {"tool": "python_exec", "arguments": {"code": "print(42)"}}
            )

        assert result == mock_result
        mock_tool.run.assert_awaited_once_with({"code": "print(42)"})

        log_file = tmp_path / "tool_calls.jsonl"
        assert log_file.exists()
        entry = json.loads(log_file.read_text().strip())
        assert entry["tool"] == "python_exec"
        assert entry["args"] == {"code": "print(42)"}

    async def test_execute_defaults_arguments_to_empty_dict(self, runtime):
        mock_tool = MagicMock()
        mock_tool.run = AsyncMock(return_value={"ok": True})

        with patch.object(runtime._server, "get_tool", new=AsyncMock(return_value=mock_tool)):
            result = await runtime.execute({"tool": "some_tool"})

        assert result == {"ok": True}

    async def test_execute_normalizes_non_dict_result(self, runtime):
        mock_tool = MagicMock()
        mock_tool.run = AsyncMock(return_value="raw")

        with patch.object(runtime._server, "get_tool", new=AsyncMock(return_value=mock_tool)):
            result = await runtime.execute({"tool": "t", "arguments": {}})

        assert result == {"status": "success", "result": "raw"}

    async def test_execute_returns_error_dict_on_unknown_tool(self, runtime, tmp_path):
        with patch.object(runtime._server, "get_tool", new=AsyncMock(side_effect=KeyError("x"))):
            result = await runtime.execute({"tool": "no_such_tool", "arguments": {}})

        assert result["status"] == "error"
        assert "no_such_tool" in result["error"]

        log_file = tmp_path / "tool_calls.jsonl"
        assert log_file.exists()

    async def test_execute_returns_error_on_value_error(self, runtime):
        with patch.object(
            runtime._server,
            "get_tool",
            new=AsyncMock(side_effect=ValueError("invalid tool")),
        ):
            result = await runtime.execute({"tool": "bad_tool", "arguments": {}})

        assert result["status"] == "error"
        assert "bad_tool" in result["error"]

    async def test_execute_times_out(self, runtime, monkeypatch):
        mock_tool = MagicMock()

        async def slow_run(_: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(1)
            return {"status": "success"}

        mock_tool.run = AsyncMock(side_effect=slow_run)
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.tool_runtime.settings.gem.tool_timeout", 0.01
        )

        with patch.object(runtime._server, "get_tool", new=AsyncMock(return_value=mock_tool)):
            with pytest.raises(TimeoutError):
                await runtime.execute({"tool": "python_exec", "arguments": {}})


# ---------------------------------------------------------------------------
# list_schemas
# ---------------------------------------------------------------------------


class TestListSchemas:
    def _make_tool(self, name: str, description: str, parameters: dict) -> MagicMock:
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.parameters = parameters
        return tool

    async def test_returns_tool_schemas(self, runtime):
        tool_a = self._make_tool("tool_a", "Does A", {"type": "object"})
        tool_b = self._make_tool("tool_b", "Does B", {})

        with patch.object(
            runtime._server,
            "get_tools",
            new=AsyncMock(return_value={"tool_a": tool_a, "tool_b": tool_b}),
        ):
            schemas = await runtime.list_schemas()

        assert schemas == [
            {"name": "tool_a", "description": "Does A", "parameters": {"type": "object"}},
            {"name": "tool_b", "description": "Does B", "parameters": {}},
        ]

    async def test_empty_tools(self, runtime):
        with patch.object(runtime._server, "get_tools", new=AsyncMock(return_value={})):
            assert await runtime.list_schemas() == []

    async def test_single_tool(self, runtime):
        tool = self._make_tool("only", "The only tool", {"x": "int"})

        with patch.object(
            runtime._server,
            "get_tools",
            new=AsyncMock(return_value={"only": tool}),
        ):
            schemas = await runtime.list_schemas()

        assert len(schemas) == 1
        assert schemas[0]["name"] == "only"
