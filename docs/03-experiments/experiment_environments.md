# K2: Target Environments and Held-Out Test Sets

This document specifies the two target environments, their dataset sources,
train/validation/test partitioning, and subsampling strategy. It serves as
the authoritative reference for data splits across all experiments.

## Environment Selection Rationale

We select two GEM environments that satisfy three criteria:

1. **Published RL baselines** — GEM reports REINFORCE+ReBN results with Qwen3-4B
2. **Tool-augmented multi-turn interaction** — exercises the trajectory-aware fitness signal
3. **Distinct task domains** — mathematical reasoning vs. multi-hop QA

| Property | Orz57K | HotpotQA |
|---|---|---|
| GEM env ID | `math:Orz57K` | `qa:HotpotQA` |
| Domain | Mathematical reasoning | Multi-hop question answering |
| Tool | Python execution | Web search |
| DSPy module | ReAct | ReAct |
| Max steps/episode | 10 | 10 |
| Discount factor (γ) | 1.0 | 0.9 |
| Reward structure | Binary (1 = correct, 0 = wrong) | Binary (1 = correct, 0 = wrong) |

## Dataset Sources

### Orz57K (Math)

| Split | HuggingFace ID | Size | Purpose |
|---|---|---|---|
| Train | `axon-rl/ORZ-57k` | 57,000 problems | Source pool for GEPA optimization |
| Eval | `axon-rl/math-eval` (MATH500 split) | 500 problems | Held-out final evaluation |

- **Training subsample**: 500 problems drawn uniformly from the 57K pool
- **Validation**: 10% of training subsample (50 problems), held out for early stopping
- **Evaluation**: MATH500 — the same 500-problem held-out set used by GEM for all reported baselines

### HotpotQA (QA)

| Split | HuggingFace ID | Size | Purpose |
|---|---|---|---|
| Train | `axon-rl/HotpotQA` | 90,400 questions | Source pool for GEPA optimization |
| Eval | `axon-rl/search-eval` (hotpotqa split) | 512 questions | Held-out final evaluation |

- **Training subsample**: 500 questions drawn uniformly from the 90.4K pool
- **Validation**: 10% of training subsample (50 questions), held out for early stopping
- **Evaluation**: 512 questions from `axon-rl/search-eval`, matching GEM's evaluation partition

## Subsampling Strategy

All experiments use **uniform random subsampling** controlled by a fixed
`data_seed = 42`. This ensures:

- **Identical training sets** across all replications and optimizer comparisons
  (GEPA, MIPROv2, zero-shot all see the same 500 tasks)
- **Reproducibility** — any run with `data_seed=42` produces the same subsample
- **Fair comparison** — RL baselines trained on the full dataset; we subsample
  for cost reasons but evaluate on the same held-out sets

### Why 500 training tasks?

- Matches GEM's training step count (500 gradient steps per experiment)
- Keeps per-experiment token budget under 94M tokens (see cost budget in configs)
- Sufficient for GEPA's evolutionary search at `auto="heavy"` (18 candidates,
  rollout budget derived by DSPy from train+val sizes)

## Data Isolation Guarantees

The following invariants are enforced throughout all experiments:

1. **Eval data is never seen during optimization.** GEPA fitness is computed
   only on training subsample tasks. The eval split is loaded only for
   final reporting.

2. **Validation data is separate from training.** The 10% validation holdout
   is used for early stopping and hyperparameter selection, never for
   fitness computation.

3. **Identical eval sets across methods.** Zero-shot, MIPROv2, and GEPA are
   all evaluated on the same held-out eval split (MATH500 / hotpotqa).

4. **Seed-controlled partitioning.** The `data_seed` is fixed at 42 across
   all experiments. Replication variability comes only from `replication_seeds`
   (42, 123, 456), which control GEPA's stochastic search and LLM sampling.

## Quick-Test Configuration

For development and CI, a lightweight config uses Orz57K with:
- Training subsample: 50 (not 500)
- Validation: 5 problems
- Single replication (seed 42)
- Light GEPA budget (`auto="light"`, 6 candidates)
- Eval: still MATH500/500 (full eval set for comparable metrics)

## Schema Reference

All fields above are enforced by the `DatasetSplit` and `EnvironmentConfig`
Pydantic models in `src/trajectory_aware_gym/models/experiment.py`.
Experiment YAML configs in `experiments/*/config.yaml` are validated against
these schemas at load time.
