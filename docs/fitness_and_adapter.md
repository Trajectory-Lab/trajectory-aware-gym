# Trajectory-Aware Fitness & DSPy Adapter

This document describes the fitness evaluation system and the DSPy adapter that
together enable GEPA (Genetic Evolution of Prompt Assemblies) to optimize prompts
using fine-grained trajectory signals from GEM environments.

## Overview

Traditional prompt optimization treats each task as a black-box pass/fail. The
trajectory-aware fitness system instead leverages the full episode trace captured
by GEM's OpenAI Gym interface — per-turn rewards, action sequences, and step
counts — to produce a richer optimization signal. This gives GEPA's reflection
model detailed feedback about *how* a prompt performed, not just whether it
succeeded.

## Architecture

```
GEM Environment (reset / step)
         │
         ▼
   TrajectoryLogger          Collects episode trace
         │
         ▼
   TrajectoryLog             Validated Pydantic model
         │
         ▼
   CompositeFitness          Evaluates with weighted terms
         │
         ▼
   FitnessResult             Score + per-term breakdown
         │
         ▼
   TrajectoryFitnessMetric   DSPy-compatible metric → GEPA reflection
```

## Fitness Terms

### DiscountedReturnTerm (primary)

Implements the trajectory fitness function from Equation 3.1 in the proposal.

```
F(τ) = Σ_{t=0}^{T-1} γ^(T-1-t) · w_t · R_T  +  λ · Σ_{t=0}^{T-1} r_t
```

| Symbol | Meaning | Default |
|--------|---------|---------|
| `T` | Number of steps in trajectory | — |
| `γ` | Discount factor (reverse-time) | 0.99 |
| `w_t` | Per-turn weight (currently 1.0) | 1.0 |
| `R_T` | Binary success indicator: 1 if final reward > 0, else 0 | — |
| `r_t` | Actual per-step reward from environment | — |
| `λ` | Auxiliary reward scaling factor | 0.1 |

**Key design choices:**

- **Reverse-time discounting** — `γ^(T-1-t)` gives full weight (`γ^0 = 1.0`) to
  the final step and progressively discounts earlier steps. This is the opposite
  of standard RL discounting. Because we evaluate complete trajectories, the
  steps closest to the outcome carry the most signal.
- **Dual-signal** — The main term rewards successful completions while the
  auxiliary term (`λ · Σ r_t`) provides gradient signal even from failed
  trajectories, enabling partial credit.

**Range:** unbounded (typically 0–3 for successful trajectories).

### LoopDetectionPenaltyTerm

Penalizes repetitive action patterns within a sliding window.

```
For each step i ∈ [1, T):
    if action[i] ∈ {action[j] : j ∈ [max(0, i-window), i)}:
        loop_count += 1

penalty = -(loop_count / (T - 1))
```

**Range:** [-1.0, 0.0]. Returns 0.0 for trajectories with fewer than 2 steps.

**Purpose:** Discourages agents from retrying identical actions — a common
failure mode in multi-turn tool-using environments like CodeContest.

### StepEfficiencyBonusTerm

Rewards successful trajectories that complete in fewer steps.

```
efficiency = max(0, 1.0 - actual_steps / max_steps)

Returns 0.0 if the final reward ≤ 0 (failed trajectory).
```

**Range:** [0.0, 1.0]. Only successful trajectories receive a bonus; failed
trajectories get 0.0 regardless of length.

### NormalizedProgressTerm (experimental)

Measures reward-trend quality as the fraction of non-decreasing consecutive
reward pairs.

```
progress = count(r_{t+1} >= r_t) / (T - 1)
```

Single-step trajectories return 0.5 if the reward is positive, 0.0 otherwise.

**Range:** [0.0, 1.0]. See [fitness_objective_profile.md](fitness_objective_profile.md) for background.

### ActionStabilityTerm (experimental)

Penalizes both consecutive repetition and two-step oscillation patterns.

```
repeat_ratio    = count(a_t == a_{t-1}) / (T - 1)
oscillation_ratio = count(a_t == a_{t-2} ≠ a_{t-1}) / max(1, T - 2)

stability = clamp(1.0 - 0.7·repeat_ratio - 0.3·oscillation_ratio, 0, 1)
```

**Range:** [0.0, 1.0]. Returns 1.0 (perfectly stable) for single-step trajectories.

## Composite Fitness

`CompositeFitness` aggregates terms via weighted sum:

```
score = Σ weight_i · term_i.compute(trajectory)
```

**Default terms:**

| Term | Weight | Purpose |
|------|--------|---------|
| `DiscountedReturnTerm` | 1.0 (fixed) | Primary success signal |
| `LoopDetectionPenaltyTerm` | `fitness_loop_penalty_weight` | Penalize repetition |
| `StepEfficiencyBonusTerm` | `fitness_step_efficiency_weight` | Reward brevity |

Set any weight to 0.0 to disable a term for ablation studies.

`evaluate()` returns a `FitnessResult` containing:
- `score` — weighted total
- `breakdown` — list of `FitnessBreakdown` (per-term raw value, weight, contribution)
- `trajectory_length` — number of steps
- `metadata` — environment ID, run ID, hyperparameters

## Configuration

`FitnessConfig` lives in the centralized config module
(`trajectory_aware_gym.config.settings`) alongside all other project
configuration classes. It uses Pydantic Settings and can be set through
environment variables or a `.env` file.

| Env Var | Default | Constraints | Description |
|---------|---------|-------------|-------------|
| `FITNESS_GAMMA` | 0.99 | [0.0, 1.0] | Reverse-time discount factor |
| `FITNESS_LAMBDA` | 0.1 | ≥ 0.0 | Auxiliary per-turn reward scaling |
| `FITNESS_LOOP_PENALTY_WEIGHT` | 1.0 | ≥ 0.0 | Loop penalty term weight |
| `FITNESS_STEP_EFFICIENCY_WEIGHT` | 1.0 | ≥ 0.0 | Step efficiency term weight |
| `FITNESS_MAX_STEPS` | 50 | ≥ 1 | Max steps for efficiency normalization |
| `FITNESS_LOOP_WINDOW` | 3 | ≥ 1 | Sliding window for loop detection |

```python
from trajectory_aware_gym.config import FitnessConfig
from trajectory_aware_gym.fitness import CompositeFitness

config = FitnessConfig(fitness_gamma=0.95, fitness_lambda=0.2)
fitness = CompositeFitness(config)
```

## DSPy Adapter

`TrajectoryFitnessMetric` bridges the fitness system with DSPy's metric
protocol. It expects a `prediction.trajectory` attribute (a `TrajectoryLog`)
set by the GEM-DSPy adapter during episode execution.

### Modes

| Mode | `return_feedback` | Returns | Use case |
|------|-------------------|---------|----------|
| Reflection | `True` (default) | `dspy.Prediction(score, feedback)` | GEPA — reflection model receives detailed breakdown |
| Evaluation | `False` | `float` | `dspy.Evaluate` — plain numerical scoring |

### Feedback format

When `return_feedback=True`, the metric produces a structured string:

```
Fitness score: 1.8500
Trajectory length: 3 steps
  discounted_return: 1.8500 (weight=1.00, contribution=1.8500)
  loop_detection_penalty: 0.0000 (weight=1.00, contribution=0.0000)
  step_efficiency_bonus: 0.9400 (weight=1.00, contribution=0.9400)
```

This gives GEPA's reflection model (Claude Sonnet 4.5) granular signals about
what went well and what to improve when mutating prompts.

### Usage

```python
from trajectory_aware_gym.adapters import TrajectoryFitnessMetric

# For GEPA optimization (with feedback)
metric = TrajectoryFitnessMetric(return_feedback=True)

# For evaluation benchmarking (plain score)
metric = TrajectoryFitnessMetric(return_feedback=False)

# DSPy metric protocol
score = metric(example=example, prediction=prediction)
```

## Trajectory Logger

`TrajectoryLogger` captures GEM episode traces into validated `TrajectoryLog`
objects.

### Data models

**`TrajectoryStep`** — a single environment transition:
- `step_index` (1-indexed), `action`, `observation`, `reward`, `terminated`, `truncated`, `info`

**`TrajectoryLog`** — a complete episode:
- `run_id` (UUID), `environment_id`, `seed`, `started_at`, `finished_at`
- `initial_observation`, `initial_info`, `steps`, `total_reward`
- Validates that `total_reward == sum(step.reward)` and `finished_at >= started_at`

### Usage

```python
from trajectory_aware_gym.adapters import TrajectoryLogger

logger = TrajectoryLogger(environment_id="math12k", seed=42)

# After env.reset()
obs, info = env.reset(seed=42)
logger.set_initial_state(observation=obs, info=info)

# After each env.step()
obs, reward, terminated, truncated, info = env.step(action)
logger.add_step(
    action=action,
    observation=obs,
    reward=reward,
    terminated=terminated,
    truncated=truncated,
    info=info,
)

# Build validated log and optionally persist
trajectory_log = logger.build_log()
saved_path = logger.save()  # writes JSON to logs/
```

## Related

- [fitness_objective_profile.md](fitness_objective_profile.md) — experimental normalized objective-profile design
