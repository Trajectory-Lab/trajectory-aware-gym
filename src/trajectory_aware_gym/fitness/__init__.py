"""Trajectory-aware fitness functions for GEPA prompt optimization."""

from trajectory_aware_gym.fitness.composite import CompositeFitness
from trajectory_aware_gym.fitness.config import FitnessConfig
from trajectory_aware_gym.fitness.terms import (
    DiscountedReturnTerm,
    LoopDetectionPenaltyTerm,
    StepEfficiencyBonusTerm,
)
from trajectory_aware_gym.fitness.types import (
    FitnessBreakdown,
    FitnessFunction,
    FitnessResult,
    FitnessTerm,
)

__all__ = [
    "CompositeFitness",
    "DiscountedReturnTerm",
    "FitnessBreakdown",
    "FitnessConfig",
    "FitnessFunction",
    "FitnessResult",
    "FitnessTerm",
    "LoopDetectionPenaltyTerm",
    "StepEfficiencyBonusTerm",
]
