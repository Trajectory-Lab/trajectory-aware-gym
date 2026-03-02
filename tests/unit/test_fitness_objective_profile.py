"""Unit tests for profile-based trajectory fitness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryStep
from trajectory_aware_gym.fitness.objective_profile import (
    FitnessWeights,
    compute_efficiency_component,
    compute_outcome_component,
    compute_progress_component,
    compute_stability_component,
    score_trajectory_profile,
)


def build_trajectory(
    *,
    rewards: list[float],
    actions: list[str] | None = None,
    terminated: bool = True,
    truncated: bool = False,
) -> TrajectoryLog:
    started = datetime.now(UTC)
    if actions is None:
        actions = [f"a{index}" for index in range(len(rewards))]

    steps = [
        TrajectoryStep(
            step_index=index + 1,
            action=actions[index],
            observation=f"obs-{index}",
            reward=reward,
            terminated=terminated and index == len(rewards) - 1,
            truncated=truncated and index == len(rewards) - 1,
        )
        for index, reward in enumerate(rewards)
    ]

    return TrajectoryLog(
        environment_id="toy-env",
        seed=1,
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        initial_observation="start",
        steps=steps,
        total_reward=sum(rewards),
    )


def test_outcome_component_rewards_successful_trajectory():
    success = build_trajectory(rewards=[0.1, 1.0], terminated=True)
    failure = build_trajectory(rewards=[0.0, 0.0], terminated=False)

    assert compute_outcome_component(success) > compute_outcome_component(failure)


def test_progress_component_prefers_non_decreasing_rewards():
    improving = build_trajectory(rewards=[0.0, 0.2, 0.4])
    regressing = build_trajectory(rewards=[0.4, 0.2, 0.0])

    assert compute_progress_component(improving) > compute_progress_component(regressing)


@pytest.mark.parametrize(
    ("steps", "max_steps", "expected"),
    [
        (2, 10, 0.8),
        (10, 10, 0.0),
        (15, 10, 0.0),
    ],
)
def test_efficiency_component_clipped(steps: int, max_steps: int, expected: float):
    trajectory = build_trajectory(rewards=[0.0] * steps)
    assert compute_efficiency_component(trajectory, max_steps=max_steps) == pytest.approx(expected)


def test_stability_component_penalizes_repetition_and_oscillation():
    stable = build_trajectory(rewards=[0.1, 0.2, 0.3], actions=["a", "b", "c"])
    unstable = build_trajectory(rewards=[0.1, 0.2, 0.3], actions=["a", "b", "a"])

    assert compute_stability_component(stable) > compute_stability_component(unstable)


def test_score_profile_returns_diagnostics_for_truncated_low_progress_patterns():
    trajectory = build_trajectory(
        rewards=[0.3, 0.1, -0.1],
        actions=["same", "same", "same"],
        terminated=False,
        truncated=True,
    )
    result = score_trajectory_profile(trajectory)

    assert "trajectory_truncated" in result.diagnostics
    assert "low_progress" in result.diagnostics
    assert "unstable_action_pattern" in result.diagnostics


def test_custom_weights_shift_score_priority_to_efficiency():
    short = build_trajectory(rewards=[0.2, 0.2])
    long = build_trajectory(rewards=[0.2] * 10)

    default_short = score_trajectory_profile(short, max_steps=10).score
    default_long = score_trajectory_profile(long, max_steps=10).score

    weights = FitnessWeights(outcome=0.10, progress=0.10, efficiency=0.70, stability=0.10)
    weighted_short = score_trajectory_profile(short, weights=weights, max_steps=10).score
    weighted_long = score_trajectory_profile(long, weights=weights, max_steps=10).score

    assert default_short > default_long
    assert weighted_short > weighted_long
