"""Unit tests for profile-based trajectory fitness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import (
    LLMCallMetadata,
    TrajectoryLog,
    TrajectoryStep,
)
from trajectory_aware_gym.fitness.objective_profile import (
    FitnessWeights,
    compute_efficiency_component,
    compute_outcome_component,
    compute_progress_component,
    compute_stability_component,
    score_trajectory_profile,
)


def _stub_llm_call() -> LLMCallMetadata:
    return LLMCallMetadata(model_id="stub", prompt_tokens=0, completion_tokens=0, total_tokens=0)


def build_trajectory(
    *,
    rewards: list[float],
    actions: list[str] | None = None,
    terminated: bool = True,
    truncated: bool = False,
    calls_per_step: int = 1,
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
            llm_calls=[_stub_llm_call() for _ in range(calls_per_step)],
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
    """With one call per step and call_budget_per_step=1, step and call
    efficiency coincide, so the averaged component matches the env-step value.
    """
    trajectory = build_trajectory(rewards=[0.0] * steps, calls_per_step=1)
    assert compute_efficiency_component(
        trajectory, max_steps=max_steps, call_budget_per_step=1
    ) == pytest.approx(expected)


def test_efficiency_falls_back_to_env_steps_when_no_calls_logged():
    """With no LLM/tool calls recorded, efficiency uses env-step count only."""
    trajectory = build_trajectory(rewards=[0.0, 0.0, 1.0], calls_per_step=0)
    # 3 steps / max_steps=10 -> 1 - 0.3 = 0.7
    assert compute_efficiency_component(
        trajectory, max_steps=10, call_budget_per_step=8
    ) == pytest.approx(0.7)


def test_efficiency_averages_step_and_call_signals_when_both_available():
    """When calls are logged, efficiency averages env-step and call efficiency."""
    # 2 env-steps, max_steps=10 -> step efficiency = 0.8
    # 2 env-steps * 4 calls = 8 calls / (10*4=40) -> call efficiency = 0.8
    # average = 0.8
    trajectory = build_trajectory(rewards=[0.0, 1.0], calls_per_step=4)
    assert compute_efficiency_component(
        trajectory, max_steps=10, call_budget_per_step=4
    ) == pytest.approx(0.8)

    # Same env-step count but pricier: 2*8=16 calls / 40 -> call efficiency = 0.6
    # step=0.8, call=0.6 -> average = 0.7
    pricey = build_trajectory(rewards=[0.0, 1.0], calls_per_step=8)
    assert compute_efficiency_component(
        pricey, max_steps=10, call_budget_per_step=4
    ) == pytest.approx(0.7)


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

    # call_budget_per_step=1 makes total_calls == num_steps, matching the
    # "shorter trajectories are more efficient" intent of this test.
    default_short = score_trajectory_profile(short, max_steps=10, call_budget_per_step=1).score
    default_long = score_trajectory_profile(long, max_steps=10, call_budget_per_step=1).score

    weights = FitnessWeights(outcome=0.10, progress=0.10, efficiency=0.70, stability=0.10)
    weighted_short = score_trajectory_profile(
        short, weights=weights, max_steps=10, call_budget_per_step=1
    ).score
    weighted_long = score_trajectory_profile(
        long, weights=weights, max_steps=10, call_budget_per_step=1
    ).score

    assert default_short > default_long
    assert weighted_short > weighted_long
