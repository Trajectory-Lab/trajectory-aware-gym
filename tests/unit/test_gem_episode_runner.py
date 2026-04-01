"""Tests for the concrete GEM episode runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trajectory_aware_gym.adapters.gem_episode_runner import GEMEpisodeRunner


class _FakeGemModule:
    def __init__(self, env):
        self._env = env

    def make(self, environment_id: str):
        assert environment_id
        return self._env


def _fake_import_module_factory(env):
    fake_gem = _FakeGemModule(env)

    def _fake_import_module(name: str):
        if name == "gem":
            return fake_gem
        if name == "gem.envs":
            return SimpleNamespace()
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

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama/qwen3-1.7b-base",
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
    assert trajectory.initial_info["task_model_id"] == "ollama/qwen3-1.7b-base"
    assert trajectory.initial_info["episode_index"] == 0
    assert trajectory.total_reward == 1.0
    assert trajectory.total_tokens == 12
    assert trajectory.total_cost_usd == 0.0123
    assert trajectory.episode_outcome == "success"
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action == "\\boxed{4}"
    assert len(trajectory.steps[0].llm_calls) == 1

    assert metrics.run_id == trajectory.run_id
    assert metrics.step_count == 1
    assert metrics.total_reward == 1.0
    assert metrics.success is True
    assert metrics.total_tokens == 12
    assert metrics.llm_cost_usd == pytest.approx(0.0123)


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
