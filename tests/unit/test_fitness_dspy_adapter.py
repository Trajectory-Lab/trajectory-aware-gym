"""Tests for the DSPy fitness metric adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import dspy
import pytest

from trajectory_aware_gym.adapters.dspy_adapter import TrajectoryFitnessMetric
from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryStep
from trajectory_aware_gym.config import FitnessModel, settings


def _cfg(**overrides: object) -> FitnessModel:
    """Build a FitnessModel from YAML defaults with specific overrides."""
    return settings.fitness.model_copy(update=overrides)


@pytest.fixture
def sample_trajectory() -> TrajectoryLog:
    """A simple successful 2-step trajectory."""
    now = datetime.now(UTC)
    return TrajectoryLog(
        environment_id="test-env",
        seed=42,
        started_at=now,
        finished_at=now + timedelta(seconds=2),
        initial_observation="start",
        steps=[
            TrajectoryStep(
                step_index=1,
                action="think",
                observation="hint",
                reward=0.0,
                terminated=False,
                truncated=False,
            ),
            TrajectoryStep(
                step_index=2,
                action="answer",
                observation="correct",
                reward=1.0,
                terminated=True,
                truncated=False,
            ),
        ],
        total_reward=1.0,
        num_steps=2,
    )


@pytest.fixture
def config() -> FitnessModel:
    return _cfg(gamma=0.5, lambda_=0.1)


class TestTrajectoryFitnessMetric:
    """Tests for the DSPy metric bridge."""

    def test_default_config_uses_settings(self, sample_trajectory):
        metric = TrajectoryFitnessMetric(return_feedback=False)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="correct", trajectory=sample_trajectory)
        assert metric(example, prediction) > 0

    def test_with_trajectory_returns_prediction(self, sample_trajectory, config):
        metric = TrajectoryFitnessMetric(config=config, return_feedback=True)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="correct", trajectory=sample_trajectory)

        result = metric(example, prediction)

        assert isinstance(result, dspy.Prediction)
        assert result.score > 0
        assert "Fitness score:" in result.feedback

    def test_with_trajectory_returns_float(self, sample_trajectory, config):
        metric = TrajectoryFitnessMetric(config=config, return_feedback=False)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="correct", trajectory=sample_trajectory)

        result = metric(example, prediction)

        assert isinstance(result, float)
        assert result > 0

    def test_without_trajectory_returns_zero_prediction(self, config):
        metric = TrajectoryFitnessMetric(config=config, return_feedback=True)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="incorrect")

        result = metric(example, prediction)

        assert isinstance(result, dspy.Prediction)
        assert result.score == 0.0
        assert "No trajectory data" in result.feedback

    def test_without_trajectory_returns_zero_float(self, config):
        metric = TrajectoryFitnessMetric(config=config, return_feedback=False)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="incorrect")

        result = metric(example, prediction)

        assert result == 0.0

    def test_feedback_contains_term_breakdown(self, sample_trajectory, config):
        metric = TrajectoryFitnessMetric(config=config, return_feedback=True)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="correct", trajectory=sample_trajectory)

        result = metric(example, prediction)

        assert "discounted_return" in result.feedback
        assert "loop_detection_penalty" in result.feedback
        assert "step_efficiency_bonus" in result.feedback
        assert "Trajectory length: 2 steps" in result.feedback

    def test_score_consistency(self, sample_trajectory, config):
        metric_with = TrajectoryFitnessMetric(config=config, return_feedback=True)
        metric_without = TrajectoryFitnessMetric(config=config, return_feedback=False)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="correct", trajectory=sample_trajectory)

        result_with = metric_with(example, prediction)
        result_without = metric_without(example, prediction)

        assert result_with.score == pytest.approx(result_without, abs=1e-9)
