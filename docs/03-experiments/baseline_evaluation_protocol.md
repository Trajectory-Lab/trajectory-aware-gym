# K3: Baseline Methods and Comparison Protocols

This document specifies the baseline methods, evaluation protocol, and
statistical comparison framework for all experiments. It serves as the
authoritative reference for how GEPA results are compared against RL and
prompt-optimization baselines.

## RL Baselines (Published, Not Re-trained)

We compare against GEM's published REINFORCE+ReBN results. These are
**reference values only** — we do not train RL agents ourselves.

### Orz57K (Math)

| Condition | Model | MATH500 Score | Source |
|---|---|---|---|
| Base (no tool) | Qwen3-4B-Base | 61.0% | GEM Table 1 |
| Base (with tool) | Qwen3-4B-Base | 62.4% | GEM Table 1 |
| Base + RL (no tool) | Qwen3-4B | 67.4% | GEM Table 1 |
| **Base + RL (with tool)** | **Qwen3-4B** | **71.0%** | **GEM Table 1** |

**Primary comparison target: 71.0%** (REINFORCE+ReBN, with Python tool, single env)

### HotpotQA (QA)

| Condition | Model | HotpotQA Score | Source |
|---|---|---|---|
| Base (no tool) | Qwen3-4B-Base | 11.1% | GEM Table 2 |
| +RL (no tool, single env) | Qwen3-4B | 21.1% | GEM Table 2 |
| +RL (no tool, mixed env) | Qwen3-4B | 22.1% | GEM Table 2 |
| **+RL (with tool, single env)** | **Qwen3-4B** | **43.2%** | **GEM Table 2** |
| +RL (with tool, mixed env) | Qwen3-4B | 45.5% | GEM Table 2 |

**Primary comparison target: 43.2%** (REINFORCE+ReBN, with search tool, single env training)

We use the **single-environment** baseline (not mixed) because GEPA also
optimizes per-environment prompts independently.

### RL Training Hyperparameters (Reference)

These are the GEM paper's published settings. We record them for
reproducibility and to contextualize the comparison.

| Parameter | Value |
|---|---|
| Algorithm | REINFORCE+ReBN |
| Optimizer | AdamW |
| Adam betas | (0.9, 0.95) |
| Learning rate | 1e-6 (constant schedule) |
| Gradient norm clipping | 1.0 |
| Policy clipping (PPO-style) | 0.2 |
| KL coefficient | 0.0 |
| Inner proximal epochs | 2 |
| GAE λ | 0.95 |
| Discount γ | 0.9 (QA), 1.0 (Math) |
| Training steps | 500 |
| Max response tokens | 4096 |
| Sampling (top-p, top-k) | (1.0, -1) |
| Infrastructure | 8× A100 GPUs, ~1 day |

## Prompt-Optimization Baselines (Run by Us)

In addition to the RL reference, we run two prompt-optimization baselines
to isolate GEPA's contribution.

### 1. Zero-Shot Baseline

- **Description**: Unoptimized DSPy ReAct agent with hand-written task instructions
- **Optimizer**: None — uses the default system prompt
- **Purpose**: Lower bound showing raw model capability without any optimization
- **Requires optimization**: No

### 2. MIPROv2 Baseline

- **Description**: DSPy's Bayesian prompt optimizer (Opsahl-Ong et al., 2024)
- **Optimizer**: MIPROv2
- **Purpose**: State-of-the-art single-objective prompt optimization baseline;
  demonstrates whether GEPA's trajectory-aware fitness and Pareto selection
  provide gains over standard prompt tuning
- **Requires optimization**: Yes (runs on the same training subsample as GEPA)

### Comparison Matrix

| Method | Type | Trains on | Optimizes | Cost Profile |
|---|---|---|---|---|
| REINFORCE+ReBN | Weight-space RL | Full dataset | Model weights | 8×A100, ~1 day |
| Zero-shot | No optimization | — | — | Inference only |
| MIPROv2 | Token-space (Bayesian) | 500 subsample | System prompt | API calls only |
| **GEPA** | **Token-space (evolutionary)** | **500 subsample** | **System prompt** | **API calls only** |

## Evaluation Protocol

All methods are evaluated under identical conditions matching the GEM paper.

### Decoding Settings

| Parameter | Training | Evaluation |
|---|---|---|
| Temperature | 1.0 (exploration) | 0.0 (deterministic) |
| Top-p | 1.0 | 1.0 |
| Top-k | -1 (disabled) | -1 (disabled) |
| Max response tokens | 4096 | 4096 |

### Evaluation Procedure

1. **Load the held-out eval set** (MATH500 or hotpotqa split — never seen during optimization)
2. **For each task**, run **5 independent rollouts** with different random seeds
3. **Score each rollout**: binary success (1) or failure (0)
4. **Compute pass@1**: fraction of tasks where at least 1 of 5 rollouts succeeds
5. **Repeat across 3 replications** (seeds: 42, 123, 456) to measure optimizer variance
6. **Report**: mean success rate across replications with 95% CI

### Metric Definition

**Primary metric: success_rate (pass@1)**

For each task *i* with *k* = 5 rollouts:
- `success_i = 1` if any rollout produces the correct answer, else `0`
- `success_rate = (1/N) Σ success_i` where N = number of eval tasks

This matches GEM's published evaluation metric exactly.

## Statistical Comparison Protocol

### Equivalence Testing (H1)

We use **TOST (Two One-Sided Tests)** to assess whether GEPA achieves
performance *equivalent to* (not superior to) the RL baseline.

| Parameter | Value | Rationale |
|---|---|---|
| Test | TOST | Tests equivalence, not superiority |
| Equivalence margin | ±5 percentage points (0.05) | Clinically meaningful performance difference |
| Significance level (α) | 0.05 | Standard threshold |
| Null hypothesis | \|μ_GEPA - μ_RL\| ≥ 0.05 | Difference exceeds margin |
| Alternative hypothesis | \|μ_GEPA - μ_RL\| < 0.05 | Difference within margin |

**Interpretation**: If both one-sided tests reject at α = 0.05, we conclude
GEPA is equivalent to the RL baseline within ±5pp.

### Confidence Intervals

- **Method**: Bootstrap resampling (1,000 iterations)
- **Level**: 95% confidence intervals on success_rate
- **Computed over**: Task-level success indicators across all replications

### Effect Sizes

- **Cohen's d**: Standardized mean difference between GEPA and RL baseline
  success rates across replications
- **Interpretation**: |d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium

### Reporting Template

For each environment, we report:

```
Environment: [Orz57K | HotpotQA]
RL Baseline:       71.0% (REINFORCE+ReBN, GEM Table 1)
Zero-shot:         XX.X% [95% CI: XX.X–XX.X]
MIPROv2:           XX.X% [95% CI: XX.X–XX.X]
GEPA:              XX.X% [95% CI: XX.X–XX.X]
Δ (GEPA - RL):     X.X pp
TOST p-value:      0.XXX
Cohen's d:         X.XX
Equivalence:       [Established | Not established] at ±5pp, α=0.05
```

### Cost Comparison (H2)

For each method, we additionally report:
- Total tokens consumed (training + evaluation)
- Total cost in USD
- Wall-clock time
- Infrastructure requirements (GPUs vs. API calls)

The hypothesis is that GEPA requires ≥1 order of magnitude fewer compute
resources than RL training.

## Seed Management

| Seed Type | Value(s) | Controls |
|---|---|---|
| `data_seed` | 42 | Which tasks appear in the training subsample (fixed across all experiments) |
| `replication_seeds` | 42, 123, 456 | GEPA's stochastic search and LLM sampling temperature (varied per replication) |

All three replications use the same training data (same `data_seed`) but
different optimizer randomness (different `replication_seeds`). This isolates
optimizer variance from data variance.

## Schema Reference

All fields above are enforced by the following Pydantic models in
`src/trajectory_aware_gym/models/experiment.py`:

- `RLBaselineResult` — published RL baseline provenance
- `EvalProtocol` — decoding and evaluation settings
- `ComparisonProtocol` — statistical test configuration
- `PromptBaselineConfig` — zero-shot and MIPROv2 specs
- `SeedConfig` — data and replication seeds

Experiment YAML configs in `experiments/*/config.yaml` are validated against
these schemas at load time.
