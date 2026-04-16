"""Validation tests for fitness behavior with controlled trajectories.

These sanity-check tests verify that the composite fitness function
produces the expected behavioral ordering across different trajectory
types — confirming alignment with the capstone proposal.
"""

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


def _cfg(**overrides: object) -> FitnessModel:
    """Build a FitnessModel from YAML defaults with specific overrides."""
    return settings.fitness.model_copy(update=overrides)


def _stub_llm_call() -> LLMCallMetadata:
    return LLMCallMetadata(model_id="stub", prompt_tokens=0, completion_tokens=0, total_tokens=0)


@pytest.fixture
def make_trajectory():
    """Factory for building TrajectoryLog objects with controlled properties.

    Each step records one stub LLM call by default so that call-based
    efficiency terms produce a non-zero signal.
    """

    def _make(
        rewards: list[float],
        actions: list[str] | None = None,
        environment_id: str = "validation-env",
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


class TestBehavioralOrdering:
    """Verify expected fitness ordering across trajectory types."""

    def test_shorter_success_scores_higher_than_longer_success(self, make_trajectory):
        """Step efficiency: fewer steps for same outcome = higher fitness.

        Uses γ=0 so the discounted return is length-independent (only the
        final step receives credit), letting the efficiency bonus dominate.
        """
        config = _cfg(
            gamma=0.0,
            lambda_=0.0,
            step_efficiency_weight=1.0,
            max_steps=50,
        )
        fitness = CompositeFitness(config)

        short = make_trajectory(rewards=[0.0, 0.0, 1.0])
        long = make_trajectory(rewards=[0.0] * 9 + [1.0])

        assert fitness.score(short) > fitness.score(long)

    def test_no_loops_scores_higher_than_loopy_trajectory(self, make_trajectory):
        """Loop penalty: repeated actions reduce fitness."""
        config = _cfg(
            gamma=0.99,
            lambda_=0.0,
            loop_penalty_weight=1.0,
        )
        fitness = CompositeFitness(config)

        clean = make_trajectory(
            rewards=[0.0, 0.0, 1.0],
            actions=["think", "reason", "answer"],
        )
        loopy = make_trajectory(
            rewards=[0.0, 0.0, 1.0],
            actions=["think", "think", "answer"],
        )

        assert fitness.score(clean) > fitness.score(loopy)

    def test_success_always_beats_failure(self, make_trajectory):
        """Successful trajectory always scores higher than failed one."""
        config = _cfg(
            gamma=0.99,
            lambda_=0.1,
        )
        fitness = CompositeFitness(config)

        success = make_trajectory(rewards=[0.0, 1.0])
        failure = make_trajectory(rewards=[0.0, 0.0])

        assert fitness.score(success) > fitness.score(failure)

    def test_success_beats_failure_regardless_of_length(self, make_trajectory):
        """Even a long successful trajectory scores higher than a short failure."""
        config = _cfg(
            gamma=0.99,
            lambda_=0.1,
            max_steps=50,
        )
        fitness = CompositeFitness(config)

        long_success = make_trajectory(rewards=[0.0] * 20 + [1.0])
        short_failure = make_trajectory(rewards=[0.0, 0.0])

        assert fitness.score(long_success) > fitness.score(short_failure)


class TestAblationConsistency:
    """Verify ablation configs produce expected equivalences."""

    def test_disabling_both_terms_equals_discounted_return_only(self, make_trajectory):
        """With all auxiliary weights at 0, composite equals discounted return alone."""
        config = _cfg(
            gamma=0.5,
            lambda_=0.1,
            loop_penalty_weight=0.0,
            step_efficiency_weight=0.0,
            call_efficiency_weight=0.0,
        )
        fitness = CompositeFitness(config)

        trajectory = make_trajectory(
            rewards=[0.0, 0.0, 1.0],
            actions=["a", "a", "a"],  # loops present but zeroed
        )
        result = fitness.evaluate(trajectory)

        dr_term = next(b for b in result.breakdown if b.term_name == "discounted_return")
        assert result.score == pytest.approx(dr_term.weighted_value, abs=1e-9)

    def test_full_composite_differs_from_discounted_return_only(self, make_trajectory):
        """With auxiliary terms active, composite score diverges from discounted return."""
        full_config = _cfg(
            gamma=0.5,
            lambda_=0.1,
            loop_penalty_weight=1.0,
            step_efficiency_weight=1.0,
            call_efficiency_weight=1.0,
            max_steps=50,
        )
        dr_only_config = _cfg(
            gamma=0.5,
            lambda_=0.1,
            loop_penalty_weight=0.0,
            step_efficiency_weight=0.0,
            call_efficiency_weight=0.0,
        )

        trajectory = make_trajectory(
            rewards=[0.0, 0.0, 1.0],
            actions=["a", "a", "answer"],
        )

        full_score = CompositeFitness(full_config).score(trajectory)
        dr_only_score = CompositeFitness(dr_only_config).score(trajectory)

        assert full_score != pytest.approx(dr_only_score, abs=1e-9)
