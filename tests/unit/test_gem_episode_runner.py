"""Tests for the concrete GEM episode runner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from litellm.exceptions import ServiceUnavailableError  # type: ignore[import-untyped]

import trajectory_aware_gym.adapters.gem_episode_runner as gem_runner_module
from trajectory_aware_gym.adapters.gem_episode_runner import (
    GEMEpisodeRunner,
    _build_litellm_tools,
    _extract_json_payload,
    _reset_inference_semaphore,
    _supports_native_tools,
    generate_smoke_action,
)
from trajectory_aware_gym.config.core import Settings


@pytest.fixture(autouse=True)
def _reset_signal_patch():
    """Reset the gem.utils.math_grader monkey-patch flag between tests."""
    gem_runner_module._signal_patch_applied = False
    yield
    gem_runner_module._signal_patch_applied = False


class _FakeGemModule:
    def __init__(self, env):
        self._env = env

    def make(self, environment_id: str, **kwargs: object):
        assert environment_id
        return self._env


def _fake_import_module_factory(env):
    fake_gem = _FakeGemModule(env)

    def _fake_import_module(name: str):
        if name == "gem":
            return fake_gem
        if name == "gem.envs":
            return SimpleNamespace()
        if name == "gem.utils.math_grader":
            # The runner monkey-patches this module's run_with_timeout_signal
            # to bypass signal-based timeout in worker threads.
            return SimpleNamespace(run_with_timeout_signal=lambda *a, **kw: None)
        raise AssertionError(f"unexpected import: {name}")

    return _fake_import_module


def _make_response(text: str, *, prompt_tokens: int = 7, completion_tokens: int = 5):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=text, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_run_episode_records_trajectory_and_cost(monkeypatch):
    class FakeEnv:
        def reset(self, **kwargs):
            assert kwargs == {"seed": 42}
            return "Solve 2 + 2", {"source": "unit"}

        def step(self, action: str):
            assert action == "\\boxed{4}"
            return "Correct", 1.0, True, False, {"correct": True}

        def close(self):
            return None

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
        _fake_import_module_factory(FakeEnv()),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion",
        lambda **kwargs: _make_response("\\boxed{4}"),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        lambda *, completion_response: 0.0123,
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.settings.validate_aws",
        lambda: None,
    )

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="bedrock/us.meta.llama3-2-1b-instruct-v1:0",
        temperature=0.0,
        max_steps=3,
        seed=42,
        experiment_name="quick-test",
    )

    result = runner.run_episode("Solve carefully.", persist=False)
    trajectory = result.trajectory
    metrics = result.raw_metrics

    assert result.log_path is None
    assert trajectory.environment_id == "math:Orz57K"
    assert trajectory.system_prompt == "Solve carefully."
    assert trajectory.initial_observation == "Solve 2 + 2"
    assert trajectory.initial_info["source"] == "unit"
    assert trajectory.initial_info["experiment_name"] == "quick-test"
    assert trajectory.initial_info["task_model_id"] == "bedrock/us.meta.llama3-2-1b-instruct-v1:0"
    assert trajectory.initial_info["episode_index"] == 0
    assert trajectory.total_reward == 1.0
    assert trajectory.total_tokens == 12
    assert trajectory.total_cost_usd == 0.0123
    assert trajectory.episode_outcome == "success"
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action == "\\boxed{4}"
    assert len(trajectory.steps[0].llm_calls) == 1
    llm_call = trajectory.steps[0].llm_calls[0]
    assert llm_call.latency_ms is not None
    assert llm_call.latency_ms > 0
    assert llm_call.cost_type == "actual"
    assert llm_call.provider == "bedrock"

    assert metrics.run_id == trajectory.run_id
    assert metrics.step_count == 1
    assert metrics.total_reward == 1.0
    assert metrics.success is True
    assert metrics.total_tokens == 12
    assert metrics.llm_cost_usd == pytest.approx(0.0123)


def test_generate_smoke_action_keeps_ollama_cost_unavailable_when_litellm_returns_zero(
    monkeypatch,
):
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion",
        lambda **kwargs: _make_response("\\boxed{4}"),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        lambda *, completion_response: 0.0,
    )

    action, llm_call = generate_smoke_action(
        model_id="ollama/qwen3-1.7b-base",
        messages=[{"role": "user", "content": "Solve 2 + 2"}],
        temperature=0.0,
    )

    assert action == "\\boxed{4}"
    assert llm_call.cost_usd is None
    assert llm_call.cost_type == "unavailable"


def test_run_episode_executes_tool_call_before_final_action(monkeypatch):
    class FakeEnv:
        def reset(self, **kwargs):
            assert kwargs == {}
            return "What is 2 + 2?", {}

        def step(self, action: str):
            assert action == "\\boxed{4}"
            return "", 1.0, True, False, {"correct": True}

        def close(self):
            return None

    class FakeToolRuntime:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def execute(self, tool_call: dict[str, object]):
            self.calls.append(tool_call)
            return {"status": "success", "output": "4"}

        def list_schemas(self):
            return [
                {
                    "name": "python_exec",
                    "description": "Execute Python code.",
                    "parameters": {
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                        "type": "object",
                    },
                }
            ]

    responses = iter(
        [
            _make_response('{"tool":"python_exec","arguments":{"code":"print(2+2)"}}'),
            _make_response("\\boxed{4}"),
        ]
    )

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
        _fake_import_module_factory(FakeEnv()),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion",
        lambda **kwargs: next(responses),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        lambda *, completion_response: 0.001,
    )

    runtime = FakeToolRuntime()
    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama/qwen3-1.7b-base",
        temperature=0.0,
        max_steps=2,
        tools=["python_exec"],
        tool_runtime=runtime,
        max_tool_rounds=2,
    )

    result = runner.run_episode("Use tools when needed.", persist=False)
    trajectory = result.trajectory
    metrics = result.raw_metrics
    step = trajectory.steps[0]

    assert runtime.calls == [{"tool": "python_exec", "arguments": {"code": "print(2+2)"}}]
    assert step.action == "\\boxed{4}"
    assert len(step.tool_calls) == 1
    assert step.tool_calls[0].tool_name == "python_exec"
    assert step.tool_calls[0].success is True
    assert step.tool_calls[0].duration_ms is not None
    assert step.tool_calls[0].duration_ms >= 0
    assert len(step.llm_calls) == 2
    assert trajectory.total_tokens == 24
    assert metrics.total_tokens == 24
    assert metrics.step_count == 1


def test_runner_tracks_episode_history(monkeypatch):
    class FakeEnv:
        def reset(self, **kwargs):
            return "Solve 2 + 2", {}

        def step(self, action: str):
            return "Correct", 1.0, True, False, {"correct": True}

        def close(self):
            return None

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
        _fake_import_module_factory(FakeEnv()),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion",
        lambda **kwargs: _make_response("\\boxed{4}"),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        lambda *, completion_response: 0.01,
    )

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama/qwen3-1.7b-base",
        temperature=0.0,
        max_steps=1,
    )

    _ = runner.run("Solve carefully.")
    assert len(runner.episode_history) == 1

    runner.clear_episode_history()
    assert runner.episode_history == ()


def test_run_episode_sanitizes_non_finite_completion_cost(monkeypatch):
    class FakeEnv:
        def reset(self, **kwargs):
            return "Solve 2 + 2", {}

        def step(self, action: str):
            assert action == "\\boxed{4}"
            return "Correct", 1.0, True, False, {"correct": True}

        def close(self):
            return None

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
        _fake_import_module_factory(FakeEnv()),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion",
        lambda **kwargs: _make_response("\\boxed{4}"),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        lambda *, completion_response: float("nan"),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.settings.validate_aws",
        lambda: None,
    )

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="bedrock/us.meta.llama3-2-1b-instruct-v1:0",
        temperature=0.0,
        max_steps=1,
    )

    result = runner.run_episode("Solve carefully.", persist=False)

    assert result.trajectory is not None
    llm_call = result.trajectory.steps[0].llm_calls[0]
    assert llm_call.cost_usd is None
    assert llm_call.cost_type == "unavailable"
    assert result.logging_summary.numeric_anomaly_count == 1
    assert result.logging_summary.status == "partial"
    assert any(event.kind == "numeric_sanitized" for event in result.logging_summary.events)


def test_run_episode_save_failure_is_non_fatal(monkeypatch):
    class FakeEnv:
        def reset(self, **kwargs):
            return "Solve 2 + 2", {}

        def step(self, action: str):
            assert action == "\\boxed{4}"
            return "Correct", 1.0, True, False, {"correct": True}

        def close(self):
            return None

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
        _fake_import_module_factory(FakeEnv()),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion",
        lambda **kwargs: _make_response("\\boxed{4}"),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        lambda *, completion_response: 0.01,
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.storage.save_trajectory",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("db unavailable")),
    )

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama/qwen3-1.7b-base",
        temperature=0.0,
        max_steps=1,
    )

    result = runner.run_episode("Solve carefully.", persist=True)

    assert result.trajectory is not None
    assert result.raw_metrics is not None
    assert result.log_path is None
    assert result.logging_summary.trajectory_persisted is False
    assert result.logging_summary.status == "partial"
    assert any(event.kind == "persistence_failed" for event in result.logging_summary.events)


def test_run_episode_retries_transient_completion_error(monkeypatch):
    """generate_smoke_action() retries on transient LiteLLM errors."""

    # Fast retry waits for test speed
    Settings.reset()
    _reset_inference_semaphore()
    s = Settings()
    monkeypatch.setattr(s.retry, "initial_wait_seconds", 0.01)
    monkeypatch.setattr(s.retry, "max_wait_seconds", 0.05)
    monkeypatch.setattr(s.retry, "max_attempts", 3)

    class FakeEnv:
        def reset(self, **kwargs: Any) -> tuple[str, dict[str, str]]:
            return "Solve 2 + 2", {"source": "unit"}

        def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, bool]]:
            return "Correct", 1.0, True, False, {"correct": True}

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
        _fake_import_module_factory(FakeEnv()),
    )

    calls: list[int] = []

    def mock_completion(messages: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(1)
        if len(calls) == 1:
            raise ServiceUnavailableError(
                message="503 Service Unavailable",
                llm_provider="bedrock",
                model="test-model",
            )
        return _make_response("\\boxed{4}")

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion",
        mock_completion,
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        lambda *, completion_response: 0.01,
    )

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama/qwen3-1.7b-base",
        temperature=0.0,
        max_steps=3,
        seed=42,
    )

    result = runner.run_episode("Solve carefully.", persist=False)
    assert len(calls) == 2
    assert result.trajectory.total_reward == 1.0
    assert result.raw_metrics.success is True


# ---------------------------------------------------------------------------
# _extract_json_payload
# ---------------------------------------------------------------------------


class TestExtractJsonPayload:
    """Cover the rewritten JSON extraction with fast + slow paths."""

    def test_empty_string(self):
        assert _extract_json_payload("") is None

    def test_whitespace_only(self):
        assert _extract_json_payload("   \n  ") is None

    def test_plain_json_object(self):
        result = _extract_json_payload('{"tool": "python_exec", "arguments": {"code": "print(1)"}}')
        assert result == {"tool": "python_exec", "arguments": {"code": "print(1)"}}

    def test_code_fenced_json(self):
        text = '```json\n{"tool": "python_exec", "arguments": {"code": "x=1"}}\n```'
        result = _extract_json_payload(text)
        assert result is not None
        assert result["tool"] == "python_exec"

    def test_code_fenced_no_lang(self):
        text = '```\n{"tool": "search", "arguments": {"query": "test"}}\n```'
        result = _extract_json_payload(text)
        assert result is not None
        assert result["tool"] == "search"

    def test_embedded_json_in_prose(self):
        text = 'I will use the tool: {"tool": "python_exec", "arguments": {"code": "print(2+2)"}} to solve this.'
        result = _extract_json_payload(text)
        assert result is not None
        assert result["tool"] == "python_exec"

    def test_embedded_json_requires_tool_key(self):
        text = 'Some text {"key": "value"} more text'
        assert _extract_json_payload(text) is None

    def test_json_with_braces_in_code_string(self):
        text = '{"tool": "python_exec", "arguments": {"code": "d = {1: 2}; print(d)"}}'
        result = _extract_json_payload(text)
        assert result is not None
        assert result["tool"] == "python_exec"
        assert "{1: 2}" in result["arguments"]["code"]

    def test_multiple_json_objects_picks_one_with_tool(self):
        text = 'Reasoning: {"step": 1} Now: {"tool": "search", "arguments": {"query": "test"}}'
        result = _extract_json_payload(text)
        assert result is not None
        assert result["tool"] == "search"

    def test_no_json_at_all(self):
        assert _extract_json_payload("This is plain text with no JSON.") is None

    def test_malformed_json(self):
        assert _extract_json_payload('{"tool": "python_exec", broken}') is None

    def test_json_array_not_returned(self):
        assert _extract_json_payload("[1, 2, 3]") is None

    def test_non_dict_json_scalar(self):
        assert _extract_json_payload('"just a string"') is None


# ---------------------------------------------------------------------------
# _supports_native_tools
# ---------------------------------------------------------------------------


class TestSupportsNativeTools:
    @pytest.mark.parametrize(
        "model_id",
        [
            "bedrock/us.meta.llama3-1-8b-instruct-v1:0",
            "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ],
    )
    def test_bedrock_supported(self, model_id):
        assert _supports_native_tools(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "ollama/qwen3-1.7b-base",
            "sagemaker/qwen3-4b-base",
            "openrouter/meta-llama/llama-3.1-8b-instruct",
        ],
    )
    def test_non_bedrock_not_supported(self, model_id):
        assert _supports_native_tools(model_id) is False


# ---------------------------------------------------------------------------
# _build_litellm_tools
# ---------------------------------------------------------------------------


class TestBuildLitellmTools:
    def test_filters_by_tool_names(self):
        runtime = MagicMock()
        runtime.list_schemas.return_value = [
            {"name": "python_exec", "description": "Run Python", "parameters": {"type": "object"}},
            {"name": "search", "description": "Web search", "parameters": {"type": "object"}},
            {"name": "shell", "description": "Run shell", "parameters": {"type": "object"}},
        ]
        result = _build_litellm_tools(runtime, ["python_exec", "search"])
        assert len(result) == 2
        names = {t["function"]["name"] for t in result}
        assert names == {"python_exec", "search"}

    def test_returns_openai_format(self):
        runtime = MagicMock()
        runtime.list_schemas.return_value = [
            {"name": "python_exec", "description": "Run Python", "parameters": {"type": "object"}},
        ]
        result = _build_litellm_tools(runtime, ["python_exec"])
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "python_exec"
        assert tool["function"]["description"] == "Run Python"
        assert tool["function"]["parameters"] == {"type": "object"}

    def test_empty_when_no_match(self):
        runtime = MagicMock()
        runtime.list_schemas.return_value = [
            {"name": "python_exec", "description": "Run Python", "parameters": {}},
        ]
        result = _build_litellm_tools(runtime, ["search"])
        assert result == []


# ---------------------------------------------------------------------------
# _compose_system_prompt
# ---------------------------------------------------------------------------


class TestComposeSystemPrompt:
    def _make_runner(self, tools=None, model_id="ollama/qwen3-1.7b-base"):
        return GEMEpisodeRunner(
            environment_id="math:Orz57K",
            model_id=model_id,
            temperature=0.0,
            max_steps=1,
            tools=tools,
        )

    def test_no_tools_returns_prompt_unchanged(self):
        runner = self._make_runner(tools=None)
        assert runner._compose_system_prompt("Solve it.") == "Solve it."

    def test_empty_tools_returns_prompt_unchanged(self):
        runner = self._make_runner(tools=[])
        assert runner._compose_system_prompt("Solve it.") == "Solve it."

    def test_with_tools_appends_instructions(self):
        runner = self._make_runner(tools=["python_exec"])
        result = runner._compose_system_prompt("Solve it.")
        assert "python_exec" in result
        assert '{"tool"' in result
        assert result.startswith("Solve it.")

    def test_tool_descriptions_included(self):
        runner = self._make_runner(tools=["search", "python_exec"])
        result = runner._compose_system_prompt("Prompt")
        # Both tools should appear with their MCP docstring descriptions.
        assert "### python_exec" in result
        assert "### search" in result
        assert "## Available Tools" in result


# ---------------------------------------------------------------------------
# _resolve_tool_call
# ---------------------------------------------------------------------------


class TestResolveToolCall:
    def _make_runner(self, tools):
        return GEMEpisodeRunner(
            environment_id="math:Orz57K",
            model_id="ollama/qwen3-1.7b-base",
            temperature=0.0,
            max_steps=1,
            tools=tools,
        )

    def test_native_tool_call_preferred(self):
        runner = self._make_runner(tools=["python_exec"])
        native = [{"tool": "python_exec", "arguments": {"code": "print(1)"}}]
        result = runner._resolve_tool_call("some text", native)
        assert result == {"tool": "python_exec", "arguments": {"code": "print(1)"}}

    def test_native_tool_call_with_alias(self):
        runner = self._make_runner(tools=["search"])
        native = [{"tool": "web_search", "arguments": {"query": "test"}}]
        result = runner._resolve_tool_call("some text", native)
        assert result == {"tool": "search", "arguments": {"query": "test"}}

    def test_native_unknown_tool_falls_through_to_text(self):
        runner = self._make_runner(tools=["python_exec"])
        native = [{"tool": "unknown_tool", "arguments": {}}]
        text = '{"tool": "python_exec", "arguments": {"code": "x=1"}}'
        result = runner._resolve_tool_call(text, native)
        assert result is not None
        assert result["tool"] == "python_exec"

    def test_text_fallback_when_no_native(self):
        runner = self._make_runner(tools=["python_exec"])
        text = '{"tool": "python_exec", "arguments": {"code": "x=1"}}'
        result = runner._resolve_tool_call(text, [])
        assert result is not None
        assert result["tool"] == "python_exec"

    def test_returns_none_when_no_tool_found(self):
        runner = self._make_runner(tools=["python_exec"])
        result = runner._resolve_tool_call("Just a text answer", [])
        assert result is None

    @pytest.mark.parametrize(
        "bad_args",
        [
            [1, 2, 3],
            "code=print(1)",
            42,
            None,
        ],
    )
    def test_native_non_dict_args_falls_through_to_text(self, bad_args):
        """Malformed native args (non-dict) should fall through to text parsing.

        We deliberately do not iterate past the first native call or attempt
        repair — see ``_resolve_tool_call`` docstring.
        """
        runner = self._make_runner(tools=["python_exec"])
        native = [{"tool": "python_exec", "arguments": bad_args}]
        text = '{"tool": "python_exec", "arguments": {"code": "x=1"}}'
        result = runner._resolve_tool_call(text, native)
        assert result == {"tool": "python_exec", "arguments": {"code": "x=1"}}

    def test_native_non_dict_args_returns_none_without_text_fallback(self):
        runner = self._make_runner(tools=["python_exec"])
        native = [{"tool": "python_exec", "arguments": [1, 2, 3]}]
        result = runner._resolve_tool_call("free-form text answer", native)
        assert result is None


# ---------------------------------------------------------------------------
# _generate_action — native tool call extraction
# ---------------------------------------------------------------------------


class TestGenerateAction:
    def _make_native_response(
        self,
        text: str,
        tool_calls: list[dict] | None = None,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
    ):
        """Build a fake LiteLLM response with optional native tool calls."""
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        tc_objects = None
        if tool_calls:
            tc_objects = []
            for tc in tool_calls:
                fn = SimpleNamespace(name=tc["name"], arguments=tc.get("arguments", "{}"))
                tc_objects.append(SimpleNamespace(function=fn))

        message = SimpleNamespace(
            content=text,
            reasoning_content=None,
            tool_calls=tc_objects,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    def test_extracts_native_tool_calls(self, monkeypatch):
        response = self._make_native_response(
            text="",
            tool_calls=[{"name": "python_exec", "arguments": '{"code": "print(1)"}'}],
        )
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            lambda **kwargs: response,
        )
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
            lambda *, completion_response: 0.001,
        )

        runner = GEMEpisodeRunner(
            environment_id="math:Orz57K",
            model_id="ollama/qwen3-1.7b-base",
            temperature=0.0,
            max_steps=1,
        )
        _, metadata, native_calls, logging_events = runner._generate_action(
            [{"role": "user", "content": "test"}]
        )
        assert len(native_calls) == 1
        assert native_calls[0]["tool"] == "python_exec"
        assert native_calls[0]["arguments"] == {"code": "print(1)"}
        assert metadata.total_tokens == 15
        assert logging_events == []

    def test_no_native_tool_calls(self, monkeypatch):
        response = self._make_native_response(text="\\boxed{4}")
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            lambda **kwargs: response,
        )
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
            lambda *, completion_response: 0.001,
        )

        runner = GEMEpisodeRunner(
            environment_id="math:Orz57K",
            model_id="ollama/qwen3-1.7b-base",
            temperature=0.0,
            max_steps=1,
        )
        action, _, native_calls, logging_events = runner._generate_action(
            [{"role": "user", "content": "test"}]
        )
        assert action == "\\boxed{4}"
        assert native_calls == []
        assert logging_events == []

    def test_malformed_arguments_default_to_empty_dict(self, monkeypatch):
        response = self._make_native_response(
            text="",
            tool_calls=[{"name": "python_exec", "arguments": "not valid json"}],
        )
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            lambda **kwargs: response,
        )
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
            lambda *, completion_response: 0.001,
        )

        runner = GEMEpisodeRunner(
            environment_id="math:Orz57K",
            model_id="ollama/qwen3-1.7b-base",
            temperature=0.0,
            max_steps=1,
        )
        _, _, native_calls, logging_events = runner._generate_action(
            [{"role": "user", "content": "test"}]
        )
        assert len(native_calls) == 1
        assert native_calls[0]["arguments"] == {}
        assert logging_events == []


# ---------------------------------------------------------------------------
# Tool-round exhaustion: forces text-only final call
# ---------------------------------------------------------------------------


class TestToolRoundExhaustion:
    def test_forces_text_response_after_all_rounds_used(self, monkeypatch):
        """When all tool rounds produce tool calls but no text, a final
        text-only call is made without tool schemas."""

        class FakeEnv:
            def reset(self, **kwargs):
                return "What is 2+2?", {}

            def step(self, action: str):
                return "", 1.0, True, False, {"correct": True}

            def close(self):
                return None

        class FakeToolRuntime:
            def execute(self, tool_call):
                return {"status": "success", "output": "4"}

            def list_schemas(self):
                return []

        call_count = 0

        def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            # First 2 calls: tool calls only (no text)
            if call_count <= 2:
                usage = SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10)
                message = SimpleNamespace(
                    content='{"tool": "python_exec", "arguments": {"code": "print(2+2)"}}',
                    reasoning_content=None,
                    tool_calls=None,
                )
                return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
            # Third call (exhaustion): text-only response
            usage = SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10)
            message = SimpleNamespace(content="\\boxed{4}", reasoning_content=None, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
            _fake_import_module_factory(FakeEnv()),
        )
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            mock_completion,
        )
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
            lambda *, completion_response: 0.001,
        )

        runner = GEMEpisodeRunner(
            environment_id="math:Orz57K",
            model_id="ollama/qwen3-1.7b-base",
            temperature=0.0,
            max_steps=2,
            tools=["python_exec"],
            tool_runtime=FakeToolRuntime(),
            max_tool_rounds=2,
        )

        result = runner.run_episode("Use tools.", persist=False)
        step = result.trajectory.steps[0]

        # 2 tool rounds + 1 exhaustion call = 3 LLM calls
        assert len(step.llm_calls) == 3
        assert len(step.tool_calls) == 2
        assert step.action == "\\boxed{4}"
        assert call_count == 3


# ---------------------------------------------------------------------------
# cost_type semantics
# ---------------------------------------------------------------------------


def test_cost_type_unavailable_when_completion_cost_raises(monkeypatch):
    """When completion_cost raises, cost_type should be 'unavailable' and cost_usd None."""

    class FakeEnv:
        def reset(self, **kwargs):
            return "Solve 2 + 2", {}

        def step(self, action):
            return "Correct", 1.0, True, False, {"correct": True}

        def close(self):
            return None

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
        _fake_import_module_factory(FakeEnv()),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion",
        lambda **kwargs: _make_response("\\boxed{4}"),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        _raise_on_cost,
    )

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama/qwen3-1.7b-base",
        temperature=0.0,
        max_steps=3,
        seed=42,
    )

    result = runner.run_episode("Solve.", persist=False)
    llm_call = result.trajectory.steps[0].llm_calls[0]
    assert llm_call.cost_usd is None
    assert llm_call.cost_type == "unavailable"


def test_cost_type_unavailable_for_ollama_when_completion_cost_returns_zero(monkeypatch):
    class FakeEnv:
        def reset(self, **kwargs):
            return "Solve 2 + 2", {}

        def step(self, action):
            return "Correct", 1.0, True, False, {"correct": True}

        def close(self):
            return None

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
        _fake_import_module_factory(FakeEnv()),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion",
        lambda **kwargs: _make_response("\\boxed{4}"),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        lambda *, completion_response: 0.0,
    )

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama/qwen3-1.7b-base",
        temperature=0.0,
        max_steps=3,
        seed=42,
    )

    result = runner.run_episode("Solve.", persist=False)
    llm_call = result.trajectory.steps[0].llm_calls[0]
    assert llm_call.cost_usd is None
    assert llm_call.cost_type == "unavailable"
    assert result.raw_metrics.llm_cost_usd is None


def _raise_on_cost(*, completion_response):
    raise Exception("Model not mapped in LiteLLM pricing")
