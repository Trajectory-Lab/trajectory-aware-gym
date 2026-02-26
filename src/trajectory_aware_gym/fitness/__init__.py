"""Trajectory-aware fitness functions for GEPA prompt optimization."""

from trajectory_aware_gym.config import FitnessModel
from trajectory_aware_gym.fitness.composite import CompositeFitness
from trajectory_aware_gym.fitness.terms import (
    ActionStabilityTerm,
    DiscountedReturnTerm,
    LoopDetectionPenaltyTerm,
    NormalizedProgressTerm,
    StepEfficiencyBonusTerm,
)
from trajectory_aware_gym.fitness.types import (
    FitnessBreakdown,
    FitnessFunction,
    FitnessResult,
    FitnessTerm,
)

__all__ = [
    "ActionStabilityTerm",
    "CompositeFitness",
    "DiscountedReturnTerm",
    "FitnessBreakdown",
    "FitnessFunction",
    "FitnessModel",
    "FitnessResult",
    "FitnessTerm",
    "LoopDetectionPenaltyTerm",
    "NormalizedProgressTerm",
    "StepEfficiencyBonusTerm",
]
