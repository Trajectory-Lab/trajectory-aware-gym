# Normalized Objective-Profile Fitness (Supplementary Design)

> Adapted from the objective-profile approach proposed by @mattjm100 in PR #118.
> These ideas complement the primary fitness implementation (Equation 3.1) and
> are available as experimental fitness terms (`NormalizedProgressTerm`,
> `ActionStabilityTerm`) for ablation studies.

## Motivation

The primary fitness function uses raw, unbounded term values combined via
weighted sum. The objective-profile approach normalizes all components to
`[0, 1]`, which can improve stability when combining signals across
environments with different reward scales.

## Objective Components

| Component     | Range   | Description |
|---------------|---------|-------------|
| **outcome**   | [0, 1]  | Terminal success bonus + normalized final reward signal |
| **progress**  | [0, 1]  | Stepwise reward trend quality (non-decreasing ratio) |
| **efficiency**| [0, 1]  | Fewer steps relative to `max_steps` |
| **stability** | [0, 1]  | Penalizes action repetition (70%) and oscillation (30%) |

## Scalar Fitness

Given component vector **c** and weight vector **w**:

```
F(τ) = Σᵢ wᵢ · cᵢ
```

Default weights: outcome=0.40, progress=0.25, efficiency=0.20, stability=0.15.

## Integration with Primary Fitness

Rather than replacing the discounted return formula (Eq. 3.1), these
normalized components are exposed as individual `FitnessTerm` implementations:

- `NormalizedProgressTerm` — reward trend quality
- `ActionStabilityTerm` — repetition + oscillation penalty

These can be added to `CompositeFitness` alongside the existing terms for
ablation experiments testing whether normalized signals improve convergence.

## Diagnostic Signals

The profile approach also suggests diagnostic annotations:

| Condition                   | Diagnostic tag              |
|-----------------------------|-----------------------------|
| Trajectory truncated        | `trajectory_truncated`      |
| Progress component < 0.4   | `low_progress`              |
| Stability component < 0.5  | `unstable_action_pattern`   |
| Outcome component < 0.5    | `weak_outcome_signal`       |

These can be incorporated into `FitnessResult.metadata` when the
experimental terms are enabled.

## Validation Expectations

- Successful, short, non-repetitive trajectories should rank highest.
- Failed or truncated trajectories should rank lower.
- Excessive repeated/oscillating actions should reduce stability and overall score.
