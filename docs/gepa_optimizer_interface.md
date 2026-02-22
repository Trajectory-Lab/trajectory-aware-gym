# GEPA–DSPy Integration Interface

This document defines the minimal integration contract used by `trajectory_aware_gym.optimizers`.

## Core Interfaces

- `PromptMutator`
  - Signature: `(prompt: str, iteration: int, candidate_index: int) -> str`
  - Responsibility: Produce one mutated prompt from a parent prompt.

- `PromptEvaluator`
  - Signature: `(prompt: str) -> float`
  - Responsibility: Score one prompt; higher is better.

- `TrajectoryProgramRunner`
  - Method: `run(prompt: str) -> TrajectoryLog`
  - Responsibility: Execute a prompt-conditioned episode and return a trajectory.

## Adapter Function

- `build_trajectory_evaluator(runner: TrajectoryProgramRunner) -> PromptEvaluator`
  - Converts trajectory-returning execution into scalar optimization fitness.
  - Uses `score_trajectory(...)` from `trajectory_aware_gym.fitness`.

## Optimizer

- `GEPAOptimizer`
  - Inputs: evaluator, mutator, population size, iteration count, elite count, random seed.
  - Behavior:
    1. Initialize population from a seed prompt + mutations.
    2. Evaluate all candidates each iteration.
    3. Select elites by descending fitness.
    4. Spawn offspring from elites via mutation.
    5. Return best prompt and optimization history.

## Outputs

- `OptimizationResult`
  - `best_prompt: str`
  - `best_fitness: float`
  - `history: list[OptimizationIteration]`
  - `final_population: list[PromptCandidate]`

- `OptimizationIteration`
  - `iteration: int`
  - `best_fitness: float`
  - `average_fitness: float`

## Sanity Trend Validation

- `validate_optimization_trend(history, min_expected_improvement=0.0)`
  - Checks whether the best-fitness trajectory improves by at least a configured margin.
