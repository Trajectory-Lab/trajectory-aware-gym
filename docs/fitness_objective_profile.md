# Trajectory-Aware Fitness Objective Profile (Issue #16)

This document defines a trajectory fitness approach based on a **normalized objective profile** with both scalar score and structured diagnostics.

## Goals

- Produce a single scalar score for optimization loops.
- Preserve interpretable per-objective component values for analysis.
- Keep scoring stable across environments with different reward scales.

## Distinct Design Choice

Instead of a raw weighted sum of unnormalized terms, this design uses:

1. **Normalized components** in `[0, 1]`
2. **Configurable objective weights** that sum to 1.0 by default
3. **Diagnostic messages** for failure modes (looping, no progress, truncated runs)

## Objective Components

- `outcome`: Terminal success + normalized final reward signal.
- `progress`: Stepwise reward trend quality (positive-improvement ratio).
- `efficiency`: Fewer steps relative to `max_steps`.
- `stability`: Penalizes action repetition and action oscillation.

All components are clipped to `[0, 1]`.

## Scalar Fitness

Let component vector be:

$$
\mathbf{c} = (c_{outcome}, c_{progress}, c_{efficiency}, c_{stability})
$$

and weights:

$$
\mathbf{w} = (w_{outcome}, w_{progress}, w_{efficiency}, w_{stability})
$$

The scalar score is:

$$
F(\tau) = \sum_i w_i c_i
$$

with default:

- `outcome=0.40`
- `progress=0.25`
- `efficiency=0.20`
- `stability=0.15`

## Structured Feedback

Scoring returns:

- Scalar fitness value
- Per-component normalized values
- Objective weights used
- Diagnostic notes for interpretability/debugging

## Controlled Validation Expectations

- Successful, short, non-repetitive trajectories should rank highest.
- Failed or truncated trajectories should rank lower.
- Excessive repeated/oscillating actions should reduce stability and overall score.
