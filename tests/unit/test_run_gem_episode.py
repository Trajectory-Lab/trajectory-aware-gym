"""Tests for the configurable smoke helpers in scripts/run_gem_episode.py."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryStep

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_gem_episode.py"
SPEC = importlib.util.spec_from_file_location("run_gem_episode", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_gem_episode = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_gem_episode
SPEC.loader.exec_module(run_gem_episode)


def make_args(**overrides):
    """Return CLI-like args for helper testing."""
    base = {
        "environment": None,
        "seed": None,
        "system_prompt": None,
        "smoke": True,
        "experiment_config": None,
        "task_model_id": None,
        "mode": "eval",
        "episodes": None,
        "max_steps": None,
        "temperature": None,
        "max_response_tokens": None,
        "show_log": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def make_trajectory(*, steps: list[TrajectoryStep] | None = None) -> TrajectoryLog:
    """Build a minimal faithful smoke trajectory for script tests."""
    started_at = datetime.now(UTC)
    resolved_steps = (
        steps
        if steps is not None
        else [
            TrajectoryStep(
                step_index=1,
                action="\\boxed{42}",
                observation="correct",
                reward=1.0,
                terminated=True,
                truncated=False,
            )
        ]
    )
    total_reward = sum(step.reward for step in resolved_steps)
    return TrajectoryLog(
        environment_id="math:Orz57K",
        seed=42,
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=1),
        initial_observation="Solve 6*7",
        steps=resolved_steps,
        total_reward=total_reward,
        episode_outcome="success" if resolved_steps else None,
    )


def test_build_smoke_run_spec_from_experiment_config():
    args = make_args(
        experiment_config=Path("experiments/quick-test/config.yaml"),
        task_model_id="bedrock/test-qwen-profile",
    )

    spec = run_gem_episode.build_smoke_run_spec(args)

    assert spec.environment_id == "math:Orz57K"
    assert spec.experiment_name == "quick-test"
    assert spec.model_id == "bedrock/test-qwen-profile"
    assert spec.seed == 42
    assert spec.episode_count == 1
    assert spec.episode_max_steps == 10
    assert spec.max_response_tokens == 4096
    assert spec.temperature == pytest.approx(0.0)


def test_build_smoke_run_spec_respects_overrides():
    args = make_args(
        experiment_config=Path("experiments/hotpotqa-tool/config.yaml"),
        environment="qa:HotpotQA",
        seed=999,
        task_model_id="bedrock/custom-qwen",
        mode="train",
        episodes=2,
        max_steps=3,
        temperature=0.3,
        max_response_tokens=1536,
        system_prompt="custom prompt",
    )

    spec = run_gem_episode.build_smoke_run_spec(args)

    assert spec.environment_id == "qa:HotpotQA"
    assert spec.model_id == "bedrock/custom-qwen"
    assert spec.seed == 999
    assert spec.episode_count == 2
    assert spec.episode_max_steps == 3
    assert spec.max_response_tokens == 1536
    assert spec.temperature == pytest.approx(0.3)
    assert spec.system_prompt == "custom prompt"


def test_build_smoke_run_spec_requires_model_without_config():
    args = make_args()

    with pytest.raises(ValueError, match="task-model-id"):
        run_gem_episode.build_smoke_run_spec(args)


def test_build_smoke_messages_include_observation():
    messages = run_gem_episode.build_smoke_messages(
        observation="Solve 2 + 2",
        system_prompt="smoke prompt",
    )

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "smoke prompt"
    assert messages[1]["role"] == "user"
    assert "Solve 2 + 2" in messages[1]["content"]


@pytest.mark.parametrize(
    ("model_id", "expected_api_base", "expect_stop", "expect_aws_region"),
    [
        ("ollama/qwen3-1.7b-base", "http://localhost:11434", True, False),
        ("bedrock/test-qwen-profile", None, False, True),
        ("sagemaker/qwen3-1-7b-base", None, False, True),
    ],
)
def test_build_completion_kwargs(model_id, expected_api_base, expect_stop, expect_aws_region):
    kwargs = run_gem_episode._build_completion_kwargs(model_id, temperature=0.0)

    assert kwargs["model"] == model_id
    assert kwargs["temperature"] == pytest.approx(0.0)
    assert kwargs["max_tokens"] == run_gem_episode.DEFAULT_SMOKE_MAX_TOKENS
    if expected_api_base is None:
        assert "api_base" not in kwargs
    else:
        assert kwargs["api_base"] == expected_api_base
    if expect_stop:
        assert "stop" in kwargs
        assert len(kwargs["stop"]) > 0
    else:
        assert "stop" not in kwargs
    if expect_aws_region:
        assert kwargs["aws_region_name"] == "us-east-1"
    else:
        assert "aws_region_name" not in kwargs


@pytest.mark.parametrize(
    ("trajectory", "expected_message"),
    [
        (None, "faithful trajectory log"),
        (
            make_trajectory(steps=[]),
            "contains no steps",
        ),
    ],
)
def test_run_smoke_episode_requires_faithful_trajectory(monkeypatch, trajectory, expected_message):
    class FakeRunner:
        def __init__(self, **_kwargs):
            pass

        def run_episode(self, *_args, **_kwargs):
            return SimpleNamespace(trajectory=trajectory, log_path=Path("logs/trajectories.db"))

    monkeypatch.setattr(run_gem_episode, "GEMEpisodeRunner", FakeRunner)

    spec = run_gem_episode.SmokeRunSpec(
        environment_id="math:Orz57K",
        experiment_name="quick-test",
        model_id="ollama/qwen3-1.7b-base",
        seed=42,
        episode_count=1,
        episode_max_steps=1,
        max_response_tokens=64,
        temperature=0.0,
        system_prompt="solve",
    )

    with pytest.raises(RuntimeError, match=expected_message):
        run_gem_episode.run_smoke_episode(spec)
