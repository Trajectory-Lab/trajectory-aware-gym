"""Unit tests for ToolRuntime adapter."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

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
        mock_result = {"status": "success", "output": "42"}

        with patch.object(runtime, "_run_sync", return_value=mock_result):
            result = runtime.execute({"tool": "python_exec", "arguments": {"code": "print(42)"}})

        assert result == mock_result

        log_file = tmp_path / "tool_calls.jsonl"
        assert log_file.exists()
        entry = json.loads(log_file.read_text().strip())
        assert entry["tool"] == "python_exec"
        assert entry["args"] == {"code": "print(42)"}

    def test_execute_defaults_arguments_to_empty_dict(self, runtime):
        with patch.object(runtime, "_run_sync", return_value={"ok": True}):
            result = runtime.execute({"tool": "some_tool"})

        assert result == {"ok": True}

    def test_execute_normalizes_non_dict_result(self, runtime):
        with patch.object(runtime, "_run_sync", return_value="raw"):
            result = runtime.execute({"tool": "t", "arguments": {}})

        assert result == {"status": "success", "result": "raw"}


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
    def test_named_attribute_schemas(self, runtime):
        class Schema:
            name = "tool_a"
            description = "Does A"
            parameters = {"type": "object"}

        with patch.object(runtime, "_run_sync", return_value=[Schema()]):
            schemas = runtime.list_schemas()

        assert schemas == [
            {"name": "tool_a", "description": "Does A", "parameters": {"type": "object"}}
        ]

    def test_dict_schemas_passed_through(self, runtime):
        raw = {"name": "tool_b", "description": "Does B", "parameters": {}}

        with patch.object(runtime, "_run_sync", return_value=[raw]):
            schemas = runtime.list_schemas()

        assert schemas == [raw]

    def test_model_dump_schemas(self, runtime):
        class Schema:
            def model_dump(self) -> dict[str, Any]:
                return {"name": "tool_c", "description": "Does C", "parameters": {"x": "int"}}

        with patch.object(runtime, "_run_sync", return_value=[Schema()]):
            schemas = runtime.list_schemas()

        assert schemas == [{"name": "tool_c", "description": "Does C", "parameters": {"x": "int"}}]

    def test_model_dump_non_dict_falls_to_str(self, runtime):
        class Schema:
            def model_dump(self):
                return [1, 2]

        with patch.object(runtime, "_run_sync", return_value=[Schema()]):
            schemas = runtime.list_schemas()

        assert len(schemas) == 1
        assert "tool" in schemas[0]

    def test_unrecognized_schema_stringified(self, runtime):
        with patch.object(runtime, "_run_sync", return_value=["just_a_string"]):
            schemas = runtime.list_schemas()

        assert schemas == [{"tool": "just_a_string"}]

    def test_mixed_schema_types(self, runtime):
        class Named:
            name = "a"
            description = "desc"
            parameters = {}

        raw_dict = {"name": "b", "description": "desc", "parameters": {}}

        with patch.object(runtime, "_run_sync", return_value=[Named(), raw_dict, 999]):
            schemas = runtime.list_schemas()

        assert len(schemas) == 3
        assert schemas[0]["name"] == "a"
        assert schemas[1]["name"] == "b"
        assert schemas[2] == {"tool": "999"}

    def test_empty_list(self, runtime):
        with patch.object(runtime, "_run_sync", return_value=[]):
            assert runtime.list_schemas() == []
