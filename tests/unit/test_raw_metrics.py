"""Tests for raw cost, latency, and efficiency metric extraction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryStep
from trajectory_aware_gym.metrics.raw_metrics import extract_episode_raw_metrics


def _build_trajectory(
    *,
    steps: list[TrajectoryStep],
    total_reward: float,
    elapsed_seconds: float,
) -> TrajectoryLog:
    start = datetime(2026, 2, 14, 11, 5, 59, tzinfo=UTC)
    finish = start + timedelta(seconds=elapsed_seconds)
    return TrajectoryLog(
        run_id="run-1",
        environment_id="game:GuessTheNumber-v0-easy",
        seed=123,
        started_at=start,
        finished_at=finish,
        initial_observation="start",
        steps=steps,
        total_reward=total_reward,
    )


def test_extract_episode_raw_metrics_with_step_level_llm_data() -> None:
    """Extract metrics from a trajectory with cost/token/latency data in step info."""
    steps = [
        TrajectoryStep(
            step_index=1,
            action="\\\\boxed{5}",
            observation="lower",
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                "cost_usd": 0.0012,
                "latency_seconds": 0.25,
            },
        ),
        TrajectoryStep(
            step_index=2,
            action="\\\\boxed{5}",
            observation="lower",
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "usage": {"prompt_tokens": 80, "completion_tokens": 10, "total_tokens": 90},
                "cost_usd": 0.0008,
                "latency_seconds": 0.4,
            },
        ),
        TrajectoryStep(
            step_index=3,
            action="\\\\boxed{1}",
            observation="win",
            reward=1.0,
            terminated=True,
            truncated=False,
            info={},
        ),
    ]
    trajectory = _build_trajectory(steps=steps, total_reward=1.0, elapsed_seconds=2.0)

    metrics = extract_episode_raw_metrics(trajectory)

    assert metrics.success is True
    assert metrics.step_count == 3
    assert metrics.episode_latency_seconds == 2.0
    assert metrics.llm_cost_usd == pytest.approx(0.002)
    assert metrics.prompt_tokens == 180
    assert metrics.completion_tokens == 30
    assert metrics.total_tokens == 210
    assert metrics.cost_per_step_usd == pytest.approx(0.002 / 3)
    assert metrics.cost_per_success_usd == pytest.approx(0.002)
    assert metrics.tokens_per_step == pytest.approx(70.0)
    assert metrics.mean_llm_latency_seconds == pytest.approx(0.325)
    assert metrics.p95_llm_latency_seconds == pytest.approx(0.4)
    assert metrics.repeat_action_rate == pytest.approx(0.5)
    assert metrics.cost_data_coverage == pytest.approx(2 / 3)
    assert metrics.token_data_coverage == pytest.approx(2 / 3)
    assert metrics.llm_latency_data_coverage == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("terminated", "truncated", "total_reward", "expected_success"),
    [
        (True, False, 1.0, True),
        (True, False, 0.0, False),
        (False, True, 1.0, False),
        (False, False, 0.0, False),
    ],
)
def test_success_logic(terminated: bool, truncated: bool, total_reward: float, expected_success: bool) -> None:
    """Success requires terminal completion plus positive episode reward."""
    steps = [
        TrajectoryStep(
            step_index=1,
            action="action",
            observation="obs",
            reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            info={},
        )
    ]
    trajectory = _build_trajectory(steps=steps, total_reward=total_reward, elapsed_seconds=1.0)

    metrics = extract_episode_raw_metrics(trajectory)

    assert metrics.success is expected_success


def test_extract_episode_raw_metrics_without_llm_data() -> None:
    """Missing LLM instrumentation should preserve latency/efficiency metrics with null cost fields."""
    steps = [
        TrajectoryStep(
            step_index=1,
            action="a",
            observation="b",
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"suffix": "next"},
        ),
        TrajectoryStep(
            step_index=2,
            action="c",
            observation="d",
            reward=0.0,
            terminated=False,
            truncated=True,
            info={"suffix": "next"},
        ),
    ]
    trajectory = _build_trajectory(steps=steps, total_reward=0.0, elapsed_seconds=0.5)

    metrics = extract_episode_raw_metrics(trajectory)

    assert metrics.llm_cost_usd is None
    assert metrics.prompt_tokens is None
    assert metrics.completion_tokens is None
    assert metrics.total_tokens is None
    assert metrics.mean_llm_latency_seconds is None
    assert metrics.p95_llm_latency_seconds is None
    assert metrics.cost_per_step_usd is None
    assert metrics.tokens_per_step is None
    assert metrics.cost_data_coverage == 0.0
    assert metrics.token_data_coverage == 0.0
    assert metrics.llm_latency_data_coverage == 0.0
    assert metrics.steps_per_second == pytest.approx(4.0)
