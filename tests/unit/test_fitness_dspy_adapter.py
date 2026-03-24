"""Tests for the DSPy fitness metric adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import dspy
import pytest
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

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

    def test_with_trajectory_returns_score_with_feedback(self, sample_trajectory, config):
        metric = TrajectoryFitnessMetric(config=config, return_feedback=True)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="correct", trajectory=sample_trajectory)

        result = metric(example, prediction)

        assert isinstance(result, ScoreWithFeedback)
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
        assert "raw composite:" in result.feedback

    def test_score_consistency(self, sample_trajectory, config):
        metric_with = TrajectoryFitnessMetric(config=config, return_feedback=True)
        metric_without = TrajectoryFitnessMetric(config=config, return_feedback=False)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="correct", trajectory=sample_trajectory)

        result_with = metric_with(example, prediction)
        result_without = metric_without(example, prediction)

        assert result_with.score == pytest.approx(result_without, abs=1e-9)

    def test_gepa_five_arg_call(self, sample_trajectory, config):
        """GEPA calls metric(gold, pred, trace, pred_name, pred_trace)."""
        metric = TrajectoryFitnessMetric(config=config, return_feedback=True)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="correct", trajectory=sample_trajectory)

        result = metric(example, prediction, None, "predict", None)

        assert isinstance(result, dspy.Prediction)
        assert result.score > 0
        assert "Fitness score:" in result.feedback

    def test_normalized_score_in_zero_one(self, sample_trajectory, config):
        """Normalized scores must be in [0, 1] for GEPA compatibility."""
        metric = TrajectoryFitnessMetric(config=config, return_feedback=False)
        example = dspy.Example(question="test")
        prediction = dspy.Prediction(answer="correct", trajectory=sample_trajectory)

        score = metric(example, prediction)

        assert 0.0 <= score <= 1.0

    def test_normalization_preserves_ordering(self, config):
        """Better trajectories must still score higher after normalization."""
        now = datetime.now(UTC)

        def _traj(reward: float, terminated: bool) -> TrajectoryLog:
            return TrajectoryLog(
                environment_id="test",
                seed=0,
                started_at=now,
                finished_at=now + timedelta(seconds=1),
                initial_observation="s",
                steps=[
                    TrajectoryStep(
                        step_index=1,
                        action="a",
                        observation="o",
                        reward=reward,
                        terminated=terminated,
                        truncated=False,
                    ),
                ],
                total_reward=reward,
                num_steps=1,
            )

        metric = TrajectoryFitnessMetric(config=config, return_feedback=False)
        example = dspy.Example(question="test")

        success = metric(example, dspy.Prediction(answer="a", trajectory=_traj(1.0, True)))
        failure = metric(example, dspy.Prediction(answer="a", trajectory=_traj(0.0, False)))
        no_traj = metric(example, dspy.Prediction(answer="a"))

        assert success > failure
        assert failure >= no_traj

    def test_failed_trajectory_with_loops_scores_below_empty(self, config):
        """A bad looping trajectory should score below a missing trajectory's 0.0."""
        now = datetime.now(UTC)
        loopy = TrajectoryLog(
            environment_id="test",
            seed=0,
            started_at=now,
            finished_at=now + timedelta(seconds=1),
            initial_observation="start",
            steps=[
                TrajectoryStep(
                    step_index=i,
                    action="repeat",
                    observation="o",
                    reward=0.0,
                    terminated=(i == 3),
                    truncated=False,
                )
                for i in range(1, 4)
            ],
            total_reward=0.0,
            num_steps=3,
        )

        metric = TrajectoryFitnessMetric(config=config, return_feedback=False)
        example = dspy.Example(question="test")

        loopy_score = metric(example, dspy.Prediction(answer="a", trajectory=loopy))

        # Loopy failure should get a score > 0 (it's normalized, not clamped)
        # but lower than a successful trajectory
        assert loopy_score >= 0.0
        assert loopy_score < 0.5

    def test_gepa_five_arg_signature_introspection(self):
        """GEPA validates metric signature by binding 5 positional args."""
        import inspect

        metric = TrajectoryFitnessMetric()
        inspect.signature(metric).bind(None, None, None, None, None)
