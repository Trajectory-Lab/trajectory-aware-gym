"""Tests for the concrete GEM episode runner."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from trajectory_aware_gym.adapters.gem_episode_runner import (
    GEMEpisodeRunner,
    _extract_json_payload,
    _extract_text_content,
    build_smoke_messages,
)


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


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": -1}, "max_steps"),
        ({"max_response_tokens": 0}, "max_response_tokens"),
        ({"max_tool_rounds": 0}, "max_tool_rounds"),
    ],
)
def test_constructor_rejects_invalid_params(kwargs, match):
    defaults = {
        "environment_id": "math:Orz57K",
        "model_id": "ollama_chat/qwen3:1.7b",
        "temperature": 0.0,
        "max_steps": 3,
    }
    with pytest.raises(ValueError, match=match):
        GEMEpisodeRunner(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# _extract_text_content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("hello", "hello"),
        ("  spaced  ", "spaced"),
        ("", ""),
        (
            [{"type": "text", "text": "chunk1"}, {"type": "text", "text": "chunk2"}],
            "chunk1\nchunk2",
        ),
        ([{"type": "image", "url": "x"}, {"type": "text", "text": "only text"}], "only text"),
        (["raw_string"], "raw_string"),
        ([], ""),
        (42, "42"),
        (None, "None"),
    ],
)
def test_extract_text_content(content, expected):
    assert _extract_text_content(content) == expected


# ---------------------------------------------------------------------------
# _extract_json_payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"tool": "python_exec", "arguments": {}}', {"tool": "python_exec", "arguments": {}}),
        ('```json\n{"key": "val"}\n```', {"key": "val"}),
        ("not json at all", None),
        ("", None),
        ("   ", None),
        ("{malformed", None),
        ("[1, 2, 3]", None),
    ],
)
def test_extract_json_payload(text, expected):
    assert _extract_json_payload(text) == expected


# ---------------------------------------------------------------------------
# build_smoke_messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observation", "system_prompt", "history", "expected_len"),
    [
        ("obs", "sys", None, 2),
        ("obs", "sys", [{"role": "user", "content": "prev"}], 3),
        ("obs", "sys", [], 2),
    ],
)
def test_build_smoke_messages(observation, system_prompt, history, expected_len):
    msgs = build_smoke_messages(
        observation=observation, system_prompt=system_prompt, history=history
    )
    assert len(msgs) == expected_len
    assert msgs[0] == {"role": "system", "content": system_prompt}
    assert msgs[-1] == {"role": "user", "content": observation}


# ---------------------------------------------------------------------------
# Episode runner integration
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_gem(monkeypatch):
    """Shared monkeypatch wiring for GEM imports and LLM calls.

    Yields a dict the test can populate with 'env', 'responses', and 'cost'.
    """
    config: dict = {"cost": 0.0}

    def _setup(env, responses, *, cost=0.0):
        config["cost"] = cost
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
            _fake_import_module_factory(env),
        )
        if callable(responses):
            monkeypatch.setattr(
                "trajectory_aware_gym.adapters.gem_episode_runner.acompletion",
                AsyncMock(side_effect=responses),
            )
        else:
            monkeypatch.setattr(
                "trajectory_aware_gym.adapters.gem_episode_runner.acompletion",
                AsyncMock(return_value=responses),
            )
        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
            lambda *, completion_response: config["cost"],
        )

    return _setup


async def test_run_episode_rejects_blank_prompt(patch_gem):
    class FakeEnv:
        def reset(self, **kwargs):
            return "obs", {}

        def step(self, action):
            return "", 0.0, True, False, {}

        def close(self):
            return None

    patch_gem(FakeEnv(), _make_response("x"))

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama_chat/qwen3:1.7b",
        temperature=0.0,
        max_steps=1,
    )

    with pytest.raises(ValueError, match="prompt must not be blank"):
        await runner.run_episode("   ", persist=False)


async def test_run_episode_records_trajectory_and_cost(patch_gem):
    class FakeEnv:
        def reset(self, **kwargs):
            assert kwargs == {"seed": 42}
            return "Solve 2 + 2", {"source": "unit"}

        def step(self, action: str):
            assert action == "\\boxed{4}"
            return "Correct", 1.0, True, False, {"correct": True}

        def close(self):
            return None

    patch_gem(FakeEnv(), _make_response("\\boxed{4}"), cost=0.0123)

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama_chat/qwen3:1.7b",
        temperature=0.0,
        max_steps=3,
        seed=42,
        experiment_name="quick-test",
    )

    result = await runner.run_episode("Solve carefully.", persist=False)
    trajectory = result.trajectory
    metrics = result.raw_metrics

    assert result.log_path is None
    assert trajectory.environment_id == "math:Orz57K"
    assert trajectory.system_prompt == "Solve carefully."
    assert trajectory.initial_observation == "Solve 2 + 2"
    assert trajectory.initial_info["source"] == "unit"
    assert trajectory.initial_info["experiment_name"] == "quick-test"
    assert trajectory.initial_info["task_model_id"] == "ollama_chat/qwen3:1.7b"
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


async def test_run_episode_executes_tool_call_before_final_action(patch_gem):
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

        async def execute(self, tool_call: dict[str, object]):
            self.calls.append(tool_call)
            return {"status": "success", "output": "4"}

    responses = iter(
        [
            _make_response('{"tool":"python_exec","arguments":{"code":"print(2+2)"}}'),
            _make_response("\\boxed{4}"),
        ]
    )

    patch_gem(FakeEnv(), lambda **kwargs: next(responses), cost=0.001)

    runtime = FakeToolRuntime()
    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama_chat/qwen3:1.7b",
        temperature=0.0,
        max_steps=2,
        tools=["python_exec"],
        tool_runtime=runtime,
        max_tool_rounds=2,
    )

    result = await runner.run_episode("Use tools when needed.", persist=False)
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
