"""Unit tests for ToolRuntime adapter."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

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

    def test_dict_passthrough(self, runtime):
        assert runtime._normalize_result({"a": 1}) == {"a": 1}

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

    def test_fallback_wraps_primitive(self, runtime):
        result = runtime._normalize_result("raw string")
        assert result == {"status": "success", "result": "raw string"}

    def test_fallback_wraps_none(self, runtime):
        result = runtime._normalize_result(None)
        assert result == {"status": "success", "result": None}


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_execute_calls_tool_and_logs(self, runtime, tmp_path):
        mock_tool = MagicMock()
        mock_result = {"status": "success", "output": "42"}

        with (
            patch.object(runtime._server, "get_tool", new=MagicMock()),
            patch.object(runtime, "_run_sync", side_effect=[mock_tool, mock_result]) as mock_sync,
        ):
            result = runtime.execute({"tool": "python_exec", "arguments": {"code": "print(42)"}})

        assert result == mock_result
        assert mock_sync.call_count == 2

        log_file = tmp_path / "tool_calls.jsonl"
        assert log_file.exists()
        entry = json.loads(log_file.read_text().strip())
        assert entry["tool"] == "python_exec"
        assert entry["args"] == {"code": "print(42)"}

    def test_execute_defaults_arguments_to_empty_dict(self, runtime):
        mock_tool = MagicMock()

        with (
            patch.object(runtime._server, "get_tool", new=MagicMock()),
            patch.object(runtime, "_run_sync", side_effect=[mock_tool, {"ok": True}]),
        ):
            result = runtime.execute({"tool": "some_tool"})

        assert result == {"ok": True}

    def test_execute_normalizes_non_dict_result(self, runtime):
        mock_tool = MagicMock()

        with (
            patch.object(runtime._server, "get_tool", new=MagicMock()),
            patch.object(runtime, "_run_sync", side_effect=[mock_tool, "raw"]),
        ):
            result = runtime.execute({"tool": "t", "arguments": {}})

        assert result == {"status": "success", "result": "raw"}

    def test_execute_returns_error_dict_on_unknown_tool(self, runtime, tmp_path):
        with (
            patch.object(runtime._server, "get_tool", new=MagicMock()),
            patch.object(runtime, "_run_sync", side_effect=KeyError("no_such_tool")),
        ):
            result = runtime.execute({"tool": "no_such_tool", "arguments": {}})

        assert result["status"] == "error"
        assert "no_such_tool" in result["error"]

        log_file = tmp_path / "tool_calls.jsonl"
        assert log_file.exists()

    def test_execute_returns_error_on_value_error(self, runtime):
        with (
            patch.object(runtime._server, "get_tool", new=MagicMock()),
            patch.object(runtime, "_run_sync", side_effect=ValueError("invalid tool")),
        ):
            result = runtime.execute({"tool": "bad_tool", "arguments": {}})

        assert result["status"] == "error"
        assert "bad_tool" in result["error"]

    def test_execute_catches_tool_run_exception(self, runtime, tmp_path):
        """tool.run() raising is caught and returned as an error dict."""
        mock_tool = MagicMock()
        call_count = 0

        def side_effect_fn(coro):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_tool
            raise RuntimeError("Pydantic validation failed")

        with (
            patch.object(runtime._server, "get_tool", new=MagicMock()),
            patch.object(runtime, "_run_sync", side_effect=side_effect_fn),
        ):
            result = runtime.execute({"tool": "python_exec", "arguments": {}})

        assert result["status"] == "error"
        assert "Pydantic validation failed" in result["error"]

        log_file = tmp_path / "tool_calls.jsonl"
        assert log_file.exists()


# ---------------------------------------------------------------------------
# _run_sync – no running loop (asyncio.run path)
# ---------------------------------------------------------------------------


class TestRunSyncNoLoop:
    def test_runs_coroutine_without_event_loop(self, runtime):
        async def coro():
            return 42

        assert runtime._run_sync(coro()) == 42


# ---------------------------------------------------------------------------
# _run_sync – with running loop (thread path)
# ---------------------------------------------------------------------------


class TestRunSyncWithLoop:
    def test_runs_coroutine_in_thread_when_loop_running(self, runtime):
        async def inner():
            async def coro():
                return "threaded"

            return runtime._run_sync(coro())

        result = asyncio.run(inner())
        assert result == "threaded"

    def test_thread_timeout_raises(self, runtime):
        leaked_coro = None

        async def inner():
            nonlocal leaked_coro

            async def slow_coro():
                await asyncio.sleep(60)

            leaked_coro = slow_coro()
            return runtime._run_sync(leaked_coro)

        with patch("trajectory_aware_gym.adapters.tool_runtime.Thread") as mock_thread_cls:
            mock_thread = mock_thread_cls.return_value
            mock_thread.is_alive.return_value = True

            with pytest.raises(TimeoutError, match="timed out"):
                asyncio.run(inner())

        if leaked_coro is not None:
            leaked_coro.close()

    def test_thread_runner_propagates_exception(self, runtime):
        async def inner():
            async def failing_coro():
                raise ValueError("boom")

            return runtime._run_sync(failing_coro())

        with pytest.raises(ValueError, match="boom"):
            asyncio.run(inner())


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

    def test_returns_tool_schemas(self, runtime):
        tool_a = self._make_tool("tool_a", "Does A", {"type": "object"})
        tool_b = self._make_tool("tool_b", "Does B", {})

        with (
            patch.object(runtime._server, "get_tools", new=MagicMock()),
            patch.object(runtime, "_run_sync", return_value={"tool_a": tool_a, "tool_b": tool_b}),
        ):
            schemas = runtime.list_schemas()

        assert schemas == [
            {"name": "tool_a", "description": "Does A", "parameters": {"type": "object"}},
            {"name": "tool_b", "description": "Does B", "parameters": {}},
        ]

    def test_empty_tools(self, runtime):
        with (
            patch.object(runtime._server, "get_tools", new=MagicMock()),
            patch.object(runtime, "_run_sync", return_value={}),
        ):
            assert runtime.list_schemas() == []

    def test_single_tool(self, runtime):
        tool = self._make_tool("only", "The only tool", {"x": "int"})

        with (
            patch.object(runtime._server, "get_tools", new=MagicMock()),
            patch.object(runtime, "_run_sync", return_value={"only": tool}),
        ):
            schemas = runtime.list_schemas()

        assert len(schemas) == 1
        assert schemas[0]["name"] == "only"
