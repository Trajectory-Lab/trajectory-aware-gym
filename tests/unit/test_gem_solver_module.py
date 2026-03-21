"""Tests for GEMSolverModule — the DSPy module bridging GEPA and GEM."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import dspy
import pytest

from trajectory_aware_gym.adapters.gem_solver_module import GEMSolverModule, _extract_final_answer
from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryStep


def _make_trajectory(*, action: str = "\\boxed{42}", reward: float = 1.0) -> TrajectoryLog:
    now = datetime.now(UTC)
    return TrajectoryLog(
        environment_id="math:Orz57K",
        seed=42,
        started_at=now,
        finished_at=now + timedelta(seconds=1),
        initial_observation="Solve: 6*7",
        steps=[
            TrajectoryStep(
                step_index=1,
                action=action,
                observation="correct",
                reward=reward,
                terminated=True,
                truncated=False,
            ),
        ],
        total_reward=reward,
        num_steps=1,
    )


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.run.return_value = _make_trajectory()
    return runner


class TestGEMSolverModule:
    def test_forward_returns_prediction_with_trajectory(self, mock_runner):
        module = GEMSolverModule(mock_runner, default_instructions="Solve math problems.")
        result = module(problem="Solve: 6*7", seed=42)

        assert isinstance(result, dspy.Prediction)
        assert isinstance(result.trajectory, TrajectoryLog)
        assert result.answer == "\\boxed{42}"

    def test_forward_passes_instructions_as_system_prompt(self, mock_runner):
        module = GEMSolverModule(mock_runner, default_instructions="Be precise.")
        module(problem="test", seed=42)

        mock_runner.run.assert_called_once_with(
            "Be precise.",
            seed_override=42,
            expected_observation="test",
        )

    def test_instructions_property(self, mock_runner):
        module = GEMSolverModule(mock_runner, default_instructions="Custom instructions.")
        assert module.instructions == "Custom instructions."

    def test_default_instructions_uses_signature_docstring(self, mock_runner):
        module = GEMSolverModule(mock_runner)
        module(problem="test", seed=42)

        # No explicit instructions -> uses the Signature docstring
        called_prompt = mock_runner.run.call_args[0][0]
        assert isinstance(called_prompt, str)
        assert len(called_prompt) > 0

    def test_forward_without_seed_uses_expected_observation_only(self, mock_runner):
        module = GEMSolverModule(mock_runner, default_instructions="Be precise.")
        module(problem="test")

        mock_runner.run.assert_called_once_with(
            "Be precise.",
            seed_override=None,
            expected_observation="test",
        )

    def test_forward_invokes_predictor_for_gepa_trace(self, mock_runner):
        """The predict call populates DSPy's trace for GEPA reflection."""
        module = GEMSolverModule(mock_runner, default_instructions="Be precise.")

        with dspy.context(trace=[]):
            module(problem="test", seed=42)
            trace = dspy.settings.trace

        assert len(trace) == 1
        predictor, inputs, prediction = trace[0]
        assert inputs["problem"] == "test"
        assert inputs["seed"] == 42

    def test_trace_predict_uses_dummy_lm_not_real(self, mock_runner):
        """The predict call must use DummyLM, never a real LLM."""
        module = GEMSolverModule(mock_runner, default_instructions="test")

        with patch("trajectory_aware_gym.adapters.gem_solver_module.DummyLM") as mock_dummy:
            mock_predict = MagicMock(return_value=dspy.Prediction(answer="x"))
            mock_predict.signature = module.predict.signature
            module.predict = mock_predict
            mock_dummy.return_value = "sentinel_lm"

            module(problem="test", seed=1)

            mock_dummy.assert_called_once()

    def test_trace_contains_real_answer_not_placeholder(self, mock_runner):
        """Trace prediction should carry the runner's actual answer."""
        module = GEMSolverModule(mock_runner, default_instructions="test")

        with dspy.context(trace=[]):
            module(problem="Solve: 6*7", seed=42)
            trace = dspy.settings.trace

        assert len(trace) == 1
        _predictor, _inputs, prediction = trace[0]
        assert prediction.answer == "\\boxed{42}"

    def test_has_predict_for_gepa(self, mock_runner):
        module = GEMSolverModule(mock_runner, default_instructions="test")
        predictors = module.predictors()
        assert len(predictors) == 1

    def test_named_predictors_discoverable(self, mock_runner):
        module = GEMSolverModule(mock_runner, default_instructions="test")
        named = dict(module.named_predictors())
        assert "predict" in named


class TestExtractFinalAnswer:
    def test_extracts_last_step_action(self):
        trajectory = _make_trajectory(action="\\boxed{99}")
        assert _extract_final_answer(trajectory) == "\\boxed{99}"

    def test_empty_steps_returns_sentinel(self):
        now = datetime.now(UTC)
        trajectory = TrajectoryLog(
            environment_id="test",
            seed=0,
            started_at=now,
            finished_at=now,
            initial_observation="start",
            steps=[],
            total_reward=0.0,
            num_steps=0,
        )
        assert _extract_final_answer(trajectory) == "[no-steps]"
