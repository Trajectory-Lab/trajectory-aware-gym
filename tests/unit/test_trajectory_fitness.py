"""Unit tests for trajectory-aware fitness scoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryStep
from trajectory_aware_gym.fitness import TrajectoryFitnessConfig, score_trajectory


def _build_trajectory(
    *, actions: list[str], rewards: list[float], terminated: bool
) -> TrajectoryLog:
    started = datetime.now(UTC)
    steps = [
        TrajectoryStep(
            step_index=index + 1,
            action=action,
            observation=f"obs-{index}",
            reward=rewards[index],
            terminated=terminated and index == len(actions) - 1,
            truncated=False,
        )
        for index, action in enumerate(actions)
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


def test_score_trajectory_success_has_positive_signal():
    trajectory = _build_trajectory(actions=["a", "b"], rewards=[0.2, 1.0], terminated=True)
    result = score_trajectory(trajectory)

    assert result.success_indicator == 1.0
    assert result.discounted_success_component > 0
    assert result.final_fitness > 0


def test_score_trajectory_loop_penalty_applied():
    trajectory = _build_trajectory(
        actions=["same", "same", "same"], rewards=[0.0, 0.0, 0.1], terminated=True
    )
    result = score_trajectory(trajectory)

    assert result.repeated_action_count == 2
    assert result.loop_penalty_component > 0


@pytest.mark.parametrize(
    ("max_steps", "expected_sign"),
    [
        (3, 0),
        (10, 1),
    ],
)
def test_step_efficiency_bonus_depends_on_budget(max_steps: int, expected_sign: int):
    trajectory = _build_trajectory(
        actions=["a", "b", "c"], rewards=[0.0, 0.0, 1.0], terminated=True
    )
    config = TrajectoryFitnessConfig(max_steps=max_steps)
    result = score_trajectory(trajectory, config)

    if expected_sign == 0:
        assert result.step_efficiency_component == 0.0
    else:
        assert result.step_efficiency_component > 0
