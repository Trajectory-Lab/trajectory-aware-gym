# Phase 3 Raw Metrics (N1)

This document defines the pragmatic raw metrics required for:

- [N1] Collect cost, latency, and efficiency metrics (#74)
- [N2] Normalize and aggregate metrics across runs (#75)
- [N3] Cross-check metrics against logs and configs (#76)

The goal in N1 is not final analysis. It is to collect **stable, per-episode raw signals** that can be normalized and validated later.

## Why these metrics

The project hypotheses require both quality and resource accounting:

- H1 focuses on task performance parity.
- H2 requires evidence that GEPA uses substantially fewer resources.
- H3 focuses on optimization dynamics and convergence behavior.

N1 therefore captures three groups of metrics for every trajectory log:

1. Cost metrics
2. Latency metrics
3. Efficiency metrics

## Collected fields

`collect_raw_metrics.py` outputs one row per trajectory with the following fields:

- `run_id`, `environment_id`, `seed`
- `started_at`, `finished_at`, `episode_latency_seconds`
- `step_count`, `terminated`, `truncated`, `success`
- `total_reward`, `reward_per_step`, `steps_per_second`, `reward_per_second`
- `repeat_action_rate`
- `llm_cost_usd`, `prompt_tokens`, `completion_tokens`, `total_tokens`
- `mean_llm_latency_seconds`, `p95_llm_latency_seconds`
- `cost_per_step_usd`, `cost_per_success_usd`, `tokens_per_step`
- `cost_data_coverage`, `token_data_coverage`, `llm_latency_data_coverage`

## Field semantics

- `success`: `True` when final step is terminated and `total_reward > 0`.
- `episode_latency_seconds`: wall-clock duration from log start to finish timestamps.
- `repeat_action_rate`: fraction of adjacent action pairs that are identical.
- `cost_data_coverage`: fraction of steps with step-level cost data present.
- `token_data_coverage`: fraction of steps with token data present.
- `llm_latency_data_coverage`: fraction of steps with step-level latency present.

Coverage metrics are critical in N1 because some environments and wrappers currently log only trajectory state/reward. They prevent silent misuse of incomplete metrics in later phases.

## Extraction strategy

The extractor reads known keys from each step `info` payload using tolerant aliases. For example:

- Cost aliases: `cost_usd`, `llm_cost_usd`, `usage.cost_usd`
- Token aliases: `prompt_tokens`, `completion_tokens`, `total_tokens`, and nested `usage.*`
- Latency aliases: `latency_seconds`, `llm_latency_seconds`, `duration_seconds`

If `total_tokens` is missing but prompt and completion tokens exist, it is derived as their sum.

## Run collection

```bash
uv run python scripts/collect_raw_metrics.py \
  --input-dir logs \
  --output-dir results/raw_metrics \
  --output-prefix phase3_raw_metrics
```

Outputs:

- `results/raw_metrics/phase3_raw_metrics.csv`
- `results/raw_metrics/phase3_raw_metrics.jsonl`
- `results/raw_metrics/phase3_raw_metrics_summary.json`

## N1 acceptance mapping

- Task completed as described: per-episode raw cost/latency/efficiency metrics defined and extractable.
- Deliverable produced: CSV + JSONL + summary artifacts under `results/raw_metrics/`.
- Tests passing: unit tests cover extraction logic and missing-data behavior.
