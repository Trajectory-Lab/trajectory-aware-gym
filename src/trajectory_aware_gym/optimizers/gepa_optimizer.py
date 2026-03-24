"""GEPA optimizer interfaces and mutation-selection loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from inspect import isawaitable
from random import Random
from statistics import mean
from typing import Protocol

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog
from trajectory_aware_gym.fitness import score_trajectory


class PromptMutator(Protocol):
    """Callable interface for mutating a prompt string."""

    def __call__(self, prompt: str, iteration: int, candidate_index: int) -> str: ...


class PromptEvaluator(Protocol):
    """Callable interface for evaluating a prompt to scalar fitness."""

    def __call__(self, prompt: str) -> float: ...


class TrajectoryProgramRunner(Protocol):
    """Minimal interface to run a prompt-conditioned episode/program."""

    async def run(self, prompt: str) -> TrajectoryLog: ...


@dataclass
class PromptCandidate:
    """One candidate prompt in the GEPA population."""

    prompt: str
    fitness: float | None = None
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationIteration:
    """Per-iteration aggregate metrics."""

    iteration: int
    best_fitness: float
    average_fitness: float


@dataclass(frozen=True)
class OptimizationResult:
    """Final optimization output and diagnostics."""

    best_prompt: str
    best_fitness: float
    history: list[OptimizationIteration]
    final_population: list[PromptCandidate]


@dataclass(frozen=True)
class TrendValidationResult:
    """Sanity-check output for optimization dynamics."""

    is_improving: bool
    improvement: float
    first_best_fitness: float
    last_best_fitness: float


def build_trajectory_evaluator(
    runner: TrajectoryProgramRunner,
) -> PromptEvaluator:
    """Adapt a trajectory-producing runner into a scalar prompt evaluator."""

    def evaluate(prompt: str) -> float:
        trajectory_or_awaitable = runner.run(prompt)
        if isawaitable(trajectory_or_awaitable):
            trajectory = asyncio.run(trajectory_or_awaitable)
        else:
            trajectory = trajectory_or_awaitable
        return score_trajectory(trajectory).final_fitness

    return evaluate


def validate_optimization_trend(
    history: list[OptimizationIteration],
    *,
    min_expected_improvement: float = 0.0,
) -> TrendValidationResult:
    """Validate that best fitness improves over iterations by a minimum amount."""
    if not history:
        return TrendValidationResult(
            is_improving=False,
            improvement=0.0,
            first_best_fitness=0.0,
            last_best_fitness=0.0,
        )

    first_best = history[0].best_fitness
    last_best = history[-1].best_fitness
    improvement = last_best - first_best

    return TrendValidationResult(
        is_improving=improvement >= min_expected_improvement,
        improvement=improvement,
        first_best_fitness=first_best,
        last_best_fitness=last_best,
    )


class GEPAOptimizer:
    """Minimal GEPA-style evolutionary optimizer for prompt strings."""

    def __init__(
        self,
        *,
        evaluator: PromptEvaluator,
        mutator: PromptMutator,
        population_size: int,
        iterations: int,
        elite_count: int = 2,
        random_seed: int | None = None,
    ):
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if elite_count < 1 or elite_count >= population_size:
            raise ValueError("elite_count must be in [1, population_size - 1]")

        self._evaluator = evaluator
        self._mutator = mutator
        self._population_size = population_size
        self._iterations = iterations
        self._elite_count = elite_count
        self._rng = Random(random_seed)  # nosec B311

    def _evaluate_population(self, population: list[PromptCandidate]) -> list[PromptCandidate]:
        for candidate in population:
            if candidate.fitness is None:
                candidate.fitness = self._evaluator(candidate.prompt)
        return population

    def _select_elites(self, population: list[PromptCandidate]) -> list[PromptCandidate]:
        ranked = sorted(
            population,
            key=lambda candidate: candidate.fitness or float("-inf"),
            reverse=True,
        )
        return [
            PromptCandidate(
                prompt=elite.prompt,
                fitness=elite.fitness,
                metadata=dict(elite.metadata),
            )
            for elite in ranked[: self._elite_count]
        ]

    def _spawn_offspring(
        self, elites: list[PromptCandidate], iteration: int
    ) -> list[PromptCandidate]:
        offspring: list[PromptCandidate] = []
        for candidate_index in range(self._population_size - len(elites)):
            parent = self._rng.choice(elites)
            mutated_prompt = self._mutator(parent.prompt, iteration, candidate_index)
            offspring.append(
                PromptCandidate(
                    prompt=mutated_prompt,
                    metadata={"parent_prompt": parent.prompt, "iteration": iteration},
                )
            )
        return offspring

    def optimize(self, seed_prompt: str) -> OptimizationResult:
        """Run GEPA mutation-selection optimization from a seed prompt."""
        if not seed_prompt.strip():
            raise ValueError("seed_prompt must not be blank")

        population = [PromptCandidate(prompt=seed_prompt)]
        for candidate_index in range(1, self._population_size):
            population.append(
                PromptCandidate(prompt=self._mutator(seed_prompt, 0, candidate_index))
            )

        history: list[OptimizationIteration] = []

        for iteration in range(1, self._iterations + 1):
            self._evaluate_population(population)
            best = max(
                candidate.fitness for candidate in population if candidate.fitness is not None
            )
            average = mean(
                candidate.fitness for candidate in population if candidate.fitness is not None
            )
            history.append(
                OptimizationIteration(
                    iteration=iteration,
                    best_fitness=best,
                    average_fitness=average,
                )
            )

            if iteration == self._iterations:
                break

            elites = self._select_elites(population)
            population = elites + self._spawn_offspring(elites, iteration)

        final_ranked = sorted(
            population,
            key=lambda candidate: candidate.fitness or float("-inf"),
            reverse=True,
        )
        best_candidate = final_ranked[0]

        return OptimizationResult(
            best_prompt=best_candidate.prompt,
            best_fitness=best_candidate.fitness or float("-inf"),
            history=history,
            final_population=final_ranked,
        )
