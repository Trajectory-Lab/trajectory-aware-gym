"""Controlled integration checks for trajectory fitness behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryStep
from trajectory_aware_gym.fitness.objective_profile import score_trajectory_profile


def controlled_trajectory(
    *,
    rewards: list[float],
    actions: list[str],
    terminated: bool,
    truncated: bool,
) -> TrajectoryLog:
    started = datetime.now(UTC)
    steps = [
        TrajectoryStep(
            step_index=index + 1,
            action=action,
            observation=f"obs-{index}",
            reward=rewards[index],
            terminated=terminated and index == len(actions) - 1,
            truncated=truncated and index == len(actions) - 1,
        )
        for index, action in enumerate(actions)
    ]
    return TrajectoryLog(
        environment_id="toy-env",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        initial_observation="start",
        steps=steps,
        total_reward=sum(rewards),
    )


def test_successful_efficient_stable_trajectory_scores_highest():
    best_case = controlled_trajectory(
        rewards=[0.2, 0.6, 1.0],
        actions=["a", "b", "c"],
        terminated=True,
        truncated=False,
    )
    poor_case = controlled_trajectory(
        rewards=[0.1, 0.0, -0.1, -0.2],
        actions=["x", "x", "x", "x"],
        terminated=False,
        truncated=True,
    )

    assert score_trajectory_profile(best_case).score > score_trajectory_profile(poor_case).score


def test_repetitive_long_trajectory_penalized_against_shorter_equally_rewarded_one():
    short = controlled_trajectory(
        rewards=[0.3, 0.3, 0.4],
        actions=["a", "b", "c"],
        terminated=True,
        truncated=False,
    )
    long_repetitive = controlled_trajectory(
        rewards=[0.1, 0.1, 0.1, 0.1, 0.2, 0.3],
        actions=["r", "r", "r", "r", "r", "r"],
        terminated=True,
        truncated=False,
    )

    assert score_trajectory_profile(short).score > score_trajectory_profile(long_repetitive).score
