"""DSPy-facing adapter for trajectory profile fitness evaluation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Protocol, cast

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog
from trajectory_aware_gym.fitness.objective_profile import (
    FitnessWeights,
    TrajectoryFitnessResult,
    score_trajectory_profile,
)


class PromptEpisodeRunner(Protocol):
    """Runner protocol for prompt-conditioned episode execution."""

    def run(self, prompt: str) -> TrajectoryLog | Awaitable[TrajectoryLog]: ...


@dataclass(frozen=True)
class DSPyFitnessEvaluation:
    """Evaluation record for one DSPy prompt candidate."""

    prompt: str
    fitness: TrajectoryFitnessResult


class DSPyTrajectoryFitnessAdapter:
    """Adapter that evaluates prompt candidates from trajectory logs."""

    def __init__(
        self,
        runner: PromptEpisodeRunner,
        *,
        weights: FitnessWeights | None = None,
        max_steps: int = 50,
    ):
        self._runner = runner
        self._weights = weights
        self._max_steps = max_steps

    def evaluate_prompt(self, prompt: str) -> DSPyFitnessEvaluation:
        """Evaluate one prompt candidate and return structured fitness output."""
        trajectory_or_awaitable = self._runner.run(prompt)
        if isawaitable(trajectory_or_awaitable):
            trajectory = asyncio.run(
                _await_trajectory(cast(Awaitable[TrajectoryLog], trajectory_or_awaitable))
            )
        else:
            trajectory = trajectory_or_awaitable
        result = score_trajectory_profile(
            trajectory,
            weights=self._weights,
            max_steps=self._max_steps,
        )
        return DSPyFitnessEvaluation(prompt=prompt, fitness=result)

    def evaluate_batch(self, prompts: list[str]) -> list[DSPyFitnessEvaluation]:
        """Evaluate a batch of prompt candidates."""
        return [self.evaluate_prompt(prompt) for prompt in prompts]

    def scalar_metric(self, prompt: str) -> float:
        """Return scalar score for optimizer integrations expecting a float metric."""
        return self.evaluate_prompt(prompt).fitness.score


async def _await_trajectory(awaitable: Awaitable[TrajectoryLog]) -> TrajectoryLog:
    return await awaitable
