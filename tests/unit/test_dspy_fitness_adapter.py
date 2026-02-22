"""Unit tests for DSPy trajectory fitness adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trajectory_aware_gym.adapters.dspy_fitness_adapter import DSPyTrajectoryFitnessAdapter
from trajectory_aware_gym.adapters.trajectory_logger import (
    TrajectoryLog,
    TrajectoryStep,
)


def make_trajectory(*, reward: float, terminated: bool = True, truncated: bool = False) -> TrajectoryLog:
    started = datetime.now(UTC)
    return TrajectoryLog(
        environment_id="toy-env",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        initial_observation="start",
        steps=[
            TrajectoryStep(
                step_index=1,
                action="act",
                observation="obs",
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
        ],
        total_reward=reward,
    )


def test_adapter_scalar_metric_prefers_higher_reward_prompt():
    class Runner:
        def run(self, prompt: str) -> TrajectoryLog:
            return make_trajectory(reward=1.0 if "good" in prompt else 0.0)

    adapter = DSPyTrajectoryFitnessAdapter(Runner())

    assert adapter.scalar_metric("good prompt") > adapter.scalar_metric("bad prompt")


def test_adapter_batch_evaluation_preserves_prompt_order():
    class Runner:
        def run(self, prompt: str) -> TrajectoryLog:
            return make_trajectory(reward=float(len(prompt) % 2))

    adapter = DSPyTrajectoryFitnessAdapter(Runner())
    prompts = ["a", "bb", "ccc"]
    evaluations = adapter.evaluate_batch(prompts)

    assert [item.prompt for item in evaluations] == prompts
