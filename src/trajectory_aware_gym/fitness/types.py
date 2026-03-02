"""Fitness function protocols and result types."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog


class FitnessBreakdown(BaseModel):
    """Per-term contribution to the composite fitness score."""

    term_name: str = Field(min_length=1)
    raw_value: float
    weight: float
    weighted_value: float


class FitnessResult(BaseModel):
    """Detailed fitness evaluation of a single trajectory."""

    score: float
    breakdown: list[FitnessBreakdown] = Field(default_factory=list)
    trajectory_length: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class FitnessTerm(Protocol):
    """Protocol for a single fitness term that scores a trajectory."""

    @property
    def name(self) -> str: ...

    def compute(self, trajectory: TrajectoryLog) -> float: ...


@runtime_checkable
class FitnessFunction(Protocol):
    """Protocol for a full fitness function returning detailed results."""

    def evaluate(self, trajectory: TrajectoryLog) -> FitnessResult: ...

    def score(self, trajectory: TrajectoryLog) -> float: ...
