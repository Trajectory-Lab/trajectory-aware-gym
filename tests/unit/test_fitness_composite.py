"""Tests for composite fitness function."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import (
    LLMCallMetadata,
    TrajectoryLog,
    TrajectoryStep,
)
from trajectory_aware_gym.config import FitnessModel, settings
from trajectory_aware_gym.fitness.composite import CompositeFitness
from trajectory_aware_gym.fitness.types import FitnessFunction, FitnessTerm


def _cfg(**overrides: object) -> FitnessModel:
    """Build a FitnessModel from YAML defaults with specific overrides."""
    return settings.fitness.model_copy(update=overrides)


def _stub_llm_call() -> LLMCallMetadata:
    return LLMCallMetadata(model_id="stub", prompt_tokens=0, completion_tokens=0, total_tokens=0)


@pytest.fixture
def make_trajectory():
    """Factory for building TrajectoryLog objects with controlled properties.

    Each step is given one stub LLM call by default so call-based efficiency
    terms have something to measure.
    """

    def _make(
        rewards: list[float],
        actions: list[str] | None = None,
        environment_id: str = "test-env",
        calls_per_step: int = 1,
    ) -> TrajectoryLog:
        now = datetime.now(UTC)
        n = len(rewards)
        if actions is None:
            actions = [f"action_{i}" for i in range(n)]
        steps = [
            TrajectoryStep(
                step_index=i + 1,
                action=actions[i],
                observation=f"obs_{i}",
                reward=rewards[i],
                terminated=(i == n - 1),
                truncated=False,
                llm_calls=[_stub_llm_call() for _ in range(calls_per_step)],
            )
            for i in range(n)
        ]
        return TrajectoryLog(
            environment_id=environment_id,
            seed=42,
            started_at=now,
            finished_at=now + timedelta(seconds=max(n, 1)),
            initial_observation="start",
            steps=steps,
            total_reward=sum(rewards),
            num_steps=n,
        )

    return _make


class TestCompositeFitness:
    """Tests for the composite fitness orchestrator."""

    def test_protocol_conformance(self):
        fitness = CompositeFitness(settings.fitness)
        assert isinstance(fitness, FitnessFunction)

    def test_default_config_uses_settings(self, make_trajectory):
        fitness = CompositeFitness()
        trajectory = make_trajectory(rewards=[0.0, 1.0])
        result = fitness.evaluate(trajectory)
        assert result.score > 0

    def test_breakdown_has_four_terms(self, make_trajectory):
        fitness = CompositeFitness(settings.fitness)
        trajectory = make_trajectory(rewards=[0.0, 1.0])
        result = fitness.evaluate(trajectory)

        assert len(result.breakdown) == 4
        term_names = [b.term_name for b in result.breakdown]
        assert term_names == [
            "discounted_return",
            "loop_detection_penalty",
            "step_efficiency_bonus",
            "call_efficiency_bonus",
        ]

    def test_score_equals_evaluate_score(self, make_trajectory):
        fitness = CompositeFitness(settings.fitness)
        trajectory = make_trajectory(rewards=[0.0, 0.0, 1.0])
        assert fitness.score(trajectory) == fitness.evaluate(trajectory).score

    def test_score_equals_sum_of_weighted_terms(self, make_trajectory):
        config = _cfg(
            gamma=0.5,
            lambda_=0.1,
            loop_penalty_weight=1.0,
            step_efficiency_weight=1.0,
            call_efficiency_weight=1.0,
            max_steps=50,
        )
        fitness = CompositeFitness(config)
        trajectory = make_trajectory(rewards=[0.0, 1.0])
        result = fitness.evaluate(trajectory)

        expected_total = sum(b.weighted_value for b in result.breakdown)
        assert result.score == pytest.approx(expected_total, abs=1e-9)

    def test_metadata_captures_config(self, make_trajectory):
        config = _cfg(gamma=0.8, lambda_=0.2)
        fitness = CompositeFitness(config)
        trajectory = make_trajectory(rewards=[1.0], environment_id="math12k")
        result = fitness.evaluate(trajectory)

        assert result.metadata["environment_id"] == "math12k"
        assert result.metadata["gamma"] == 0.8
        assert result.metadata["lambda"] == 0.2
        assert "run_id" in result.metadata

    def test_trajectory_length_in_result(self, make_trajectory):
        fitness = CompositeFitness(settings.fitness)
        trajectory = make_trajectory(rewards=[0.0, 0.0, 0.0, 1.0])
        result = fitness.evaluate(trajectory)
        assert result.trajectory_length == 4

    def test_empty_trajectory(self, make_trajectory):
        fitness = CompositeFitness(settings.fitness)
        trajectory = make_trajectory(rewards=[])
        result = fitness.evaluate(trajectory)

        assert result.score == 0.0
        assert result.trajectory_length == 0
        for b in result.breakdown:
            assert b.raw_value == 0.0


class TestCompositeFitnessAblation:
    """Tests for ablation support via weight zeroing."""

    def test_disable_loop_penalty(self, make_trajectory):
        config = _cfg(
            loop_penalty_weight=0.0,
            gamma=0.5,
            lambda_=0.0,
        )
        fitness = CompositeFitness(config)
        # All identical actions should produce loops, but penalty weight is 0
        trajectory = make_trajectory(rewards=[0.0, 0.0, 1.0], actions=["a", "a", "a"])
        result = fitness.evaluate(trajectory)

        loop_term = next(b for b in result.breakdown if b.term_name == "loop_detection_penalty")
        assert loop_term.raw_value < 0  # Loops detected
        assert loop_term.weighted_value == 0.0  # But zeroed out by weight

    def test_disable_step_efficiency(self, make_trajectory):
        config = _cfg(
            step_efficiency_weight=0.0,
            gamma=0.5,
            lambda_=0.0,
        )
        fitness = CompositeFitness(config)
        trajectory = make_trajectory(rewards=[0.0, 1.0])
        result = fitness.evaluate(trajectory)

        eff_term = next(b for b in result.breakdown if b.term_name == "step_efficiency_bonus")
        assert eff_term.raw_value > 0  # Bonus exists
        assert eff_term.weighted_value == 0.0  # But zeroed out

    def test_disable_call_efficiency(self, make_trajectory):
        config = _cfg(
            call_efficiency_weight=0.0,
            gamma=0.5,
            lambda_=0.0,
        )
        fitness = CompositeFitness(config)
        trajectory = make_trajectory(rewards=[0.0, 1.0])
        result = fitness.evaluate(trajectory)

        call_term = next(b for b in result.breakdown if b.term_name == "call_efficiency_bonus")
        assert call_term.raw_value > 0  # Bonus exists (calls are logged)
        assert call_term.weighted_value == 0.0  # But zeroed out

    def test_only_discounted_return(self, make_trajectory):
        config = _cfg(
            loop_penalty_weight=0.0,
            step_efficiency_weight=0.0,
            call_efficiency_weight=0.0,
            gamma=0.5,
            lambda_=0.0,
        )
        fitness = CompositeFitness(config)
        trajectory = make_trajectory(rewards=[0.0, 1.0], actions=["a", "a"])
        result = fitness.evaluate(trajectory)

        dr_term = next(b for b in result.breakdown if b.term_name == "discounted_return")
        assert result.score == pytest.approx(dr_term.weighted_value, abs=1e-9)


class TestFitnessTermProtocol:
    """Tests for Protocol conformance of individual terms."""

    @pytest.mark.parametrize(
        "term_class",
        [
            "trajectory_aware_gym.fitness.terms.DiscountedReturnTerm",
            "trajectory_aware_gym.fitness.terms.LoopDetectionPenaltyTerm",
            "trajectory_aware_gym.fitness.terms.StepEfficiencyBonusTerm",
            "trajectory_aware_gym.fitness.terms.CallEfficiencyBonusTerm",
        ],
    )
    def test_term_satisfies_protocol(self, term_class):
        module_path, class_name = term_class.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        instance = cls(settings.fitness)
        assert isinstance(instance, FitnessTerm)
