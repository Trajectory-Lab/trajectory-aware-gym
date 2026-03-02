"""Composite fitness function combining multiple terms."""

from __future__ import annotations

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog
from trajectory_aware_gym.config.core import FitnessModel
from trajectory_aware_gym.fitness.terms import (
    DiscountedReturnTerm,
    LoopDetectionPenaltyTerm,
    StepEfficiencyBonusTerm,
)
from trajectory_aware_gym.fitness.types import (
    FitnessBreakdown,
    FitnessResult,
    FitnessTerm,
)


class CompositeFitness:
    """Weighted combination of fitness terms with detailed breakdown.

    Default configuration includes all three proposal terms.
    Set weight to 0.0 in FitnessModel to disable any term for ablation.
    """

    def __init__(self, config: FitnessModel | None = None) -> None:
        if config is None:
            from trajectory_aware_gym.config import settings

            config = settings.fitness
        self._config = config
        self._terms: list[tuple[float, FitnessTerm]] = self._build_default_terms()

    def _build_default_terms(self) -> list[tuple[float, FitnessTerm]]:
        return [
            (1.0, DiscountedReturnTerm(self._config)),
            (self._config.loop_penalty_weight, LoopDetectionPenaltyTerm(self._config)),
            (self._config.step_efficiency_weight, StepEfficiencyBonusTerm(self._config)),
        ]

    def evaluate(self, trajectory: TrajectoryLog) -> FitnessResult:
        breakdown: list[FitnessBreakdown] = []
        total_score = 0.0

        for weight, term in self._terms:
            raw = term.compute(trajectory)
            weighted = raw * weight
            total_score += weighted
            breakdown.append(
                FitnessBreakdown(
                    term_name=term.name,
                    raw_value=raw,
                    weight=weight,
                    weighted_value=weighted,
                )
            )

        return FitnessResult(
            score=total_score,
            breakdown=breakdown,
            trajectory_length=len(trajectory.steps),
            metadata={
                "environment_id": trajectory.environment_id,
                "run_id": trajectory.run_id,
                "gamma": self._config.gamma,
                "lambda": self._config.lambda_,
            },
        )

    def score(self, trajectory: TrajectoryLog) -> float:
        return self.evaluate(trajectory).score
