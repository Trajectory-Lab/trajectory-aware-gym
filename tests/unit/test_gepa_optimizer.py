"""Unit tests for GEPA optimizer integration and loop behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryStep
from trajectory_aware_gym.optimizers import (
    GEPAOptimizer,
    build_trajectory_evaluator,
    validate_optimization_trend,
)


def test_optimizer_rejects_blank_seed_prompt():
    optimizer = GEPAOptimizer(
        evaluator=lambda prompt: float(len(prompt)),
        mutator=lambda prompt, iteration, candidate_index: (
            f"{prompt}:{iteration}:{candidate_index}"
        ),
        population_size=4,
        iterations=2,
    )

    with pytest.raises(ValueError, match="seed_prompt"):
        optimizer.optimize("   ")


def test_optimizer_runs_mutation_selection_loop():
    def evaluator(prompt: str) -> float:
        return float(prompt.count("+") + len(prompt) / 100)

    def mutator(prompt: str, iteration: int, candidate_index: int) -> str:
        return f"{prompt}+{iteration}-{candidate_index}"

    optimizer = GEPAOptimizer(
        evaluator=evaluator,
        mutator=mutator,
        population_size=6,
        iterations=4,
        elite_count=2,
        random_seed=7,
    )
    result = optimizer.optimize("seed")

    assert len(result.history) == 4
    assert len(result.final_population) == 6
    assert result.best_fitness == result.history[-1].best_fitness
    assert result.best_prompt


def test_build_trajectory_evaluator_connects_to_fitness():
    class Runner:
        def run(self, prompt: str) -> TrajectoryLog:
            started = datetime.now(UTC)
            success = "win" in prompt
            steps = [
                TrajectoryStep(
                    step_index=1,
                    action=prompt,
                    observation="obs",
                    reward=1.0 if success else 0.0,
                    terminated=True,
                    truncated=False,
                )
            ]
            return TrajectoryLog(
                environment_id="toy-env",
                seed=1,
                started_at=started,
                finished_at=started + timedelta(seconds=1),
                initial_observation="start",
                steps=steps,
                total_reward=1.0 if success else 0.0,
            )

    evaluator = build_trajectory_evaluator(Runner())

    assert evaluator("please win") > evaluator("please lose")


def test_validate_optimization_trend_flags_improvement():
    optimizer = GEPAOptimizer(
        evaluator=lambda prompt: float(prompt.count("+")),
        mutator=lambda prompt, iteration, candidate_index: f"{prompt}+",
        population_size=5,
        iterations=5,
        elite_count=2,
        random_seed=11,
    )
    result = optimizer.optimize("seed")
    trend = validate_optimization_trend(result.history, min_expected_improvement=1.0)

    assert trend.is_improving
    assert trend.improvement >= 1.0
