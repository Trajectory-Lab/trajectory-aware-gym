# Production Experiment Runner

The production runner (`scripts/run_experiment.py`) executes GEPA optimization across configured task models, replication seeds, and evaluation protocols. It handles cost tracking, resume on failure, and structured output.

## Usage

```bash
poe run-experiment --config experiments/orz57k/config.yaml
```

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | Path | (required) | Path to `ExperimentConfig` YAML |
| `--max-metric-calls` | int | derived from config | Override GEPA metric call budget |
| `--seed-prompt` | str | math solver prompt | Initial system prompt for GEPA |
| `--models` | list | all in config | Subset of task model names to run |
| `--seeds` | list | all in config | Subset of replication seeds to run |
| `--fresh` | flag | false | Skip resume; start a new run |
| `--resume` | str | none | Resume a specific run by timestamp (e.g. `20260408T150000Z`) |
| `--danger-purge` | flag | false | Delete ALL prior results for this config (requires typing "yes") |
| `--results-root` | Path | `results/` | Root directory for output |
| `--halt-on-budget-exceeded` | flag | false | Stop if cost exceeds budget |

### Examples

```bash
# Run all models and seeds for Orz57K
poe run-experiment --config experiments/orz57k/config.yaml

# Run only Qwen3-4B with seed 42, budget override, fresh
poe run-experiment \
  --config experiments/orz57k/config.yaml \
  --models Qwen3-4B-Base \
  --seeds 42 \
  --max-metric-calls 256 \
  --fresh

# Resume a crashed run (auto-detect most recent incomplete)
poe run-experiment --config experiments/orz57k/config.yaml

# Resume a specific run by timestamp
poe run-experiment \
  --config experiments/orz57k/config.yaml \
  --resume 20260408T150000Z

# Ignore incomplete runs, start fresh
poe run-experiment --config experiments/orz57k/config.yaml --fresh

# Delete all Orz57K results and start over
poe run-experiment --config experiments/orz57k/config.yaml --danger-purge
```

## Results Directory Structure

Each invocation writes to a timestamped directory under `results/`:

```
results/
└── {config_name}/
    └── {timestamp}/                     # e.g. 20260408T150000Z
        ├── run_summary.json             # Experiment-level metadata and status
        └── {task_model_name}/
            └── replication_{seed}/
                ├── run_metadata.json    # Replication status, timing, eval summary
                ├── config_snapshot.yaml # Full config for reproducibility
                ├── optimized_prompt.txt # GEPA-evolved system prompt
                ├── fitness_history.json # Fitness scores per GEPA iteration
                ├── pareto_frontier.json # Pareto frontier of candidate prompts
                ├── cost_summary.json    # Token/cost breakdown (task vs reflection, train vs eval)
                ├── training_metrics.csv  # Per-episode metrics (training only)
                ├── raw_metrics.csv      # Per-episode metrics (train + eval)
                ├── raw_metrics.jsonl    # Same data, JSONL format
                ├── raw_metrics_summary.json  # Aggregated stats
                └── gepa_logs/           # GEPA internal optimization logs
```

## Resume Behavior

The runner writes `run_summary.json` at the **start** of each invocation (with `finished_at: null`) and updates it on completion (with `finished_at` set). This enables resume detection:

1. **Default**: scans for the most recent timestamp directory whose `run_summary.json` has no `finished_at`. If found, resumes into that directory. Completed replications (where `run_metadata.json` has `status: "completed"`) are skipped; incomplete ones re-run.
2. **`--fresh`**: ignores any incomplete runs, creates a new timestamp directory.
3. **`--resume <timestamp>`**: resumes a specific run by its timestamp.
4. **`--danger-purge`**: deletes `results/{config_name}/` entirely before starting.

## Key Output Files

### run_summary.json

Written at the experiment level (one per invocation). Contains `run_id`, `config_hash`, `git_commit`, per-model results, and timing.

### run_metadata.json

Written per replication. Tracks status (`running` / `completed` / `failed`), timing, the GEPA result summary, and eval scores.

### cost_summary.json

Per-replication cost breakdown:

| Field | Description |
|-------|-------------|
| `task_model_tokens` / `task_model_cost` | Total task model usage (train + eval) |
| `reflection_tokens` / `reflection_cost` | Reflection model usage during GEPA |
| `total_tokens` / `total_cost` | Combined totals |
| `training_task_model_tokens` / `eval_task_model_tokens` | Train vs eval split |
| `effective_budget_usd` | Budget from config |

### raw_metrics.csv / .jsonl

One row per episode (training and evaluation combined). Fields match the `EpisodeRawMetrics` schema documented in [phase3_raw_metrics.md](phase3_raw_metrics.md).

## Data Splits

The experiment config controls three distinct data partitions via the `environment` section:

| Field | Purpose | Default |
|-------|---------|---------|
| `train_size` | Number of training examples for GEPA optimization | (required) |
| `val_size` | GEPA validation set size (subset of trainset used for full-valset scoring) | 10% of `train_size` |
| `eval_size` | Held-out evaluation set size (never seen during optimization) | falls back to `val_size` |

Seed ranges ensure no overlap between splits:

- **Train**: seeds `data_seed` to `data_seed + train_size - 1`
- **Val**: first `val_size` examples from the trainset (same seeds, subset)
- **Eval**: seeds `data_seed + train_size` to `data_seed + train_size + eval_size - 1`

The `dataset` sub-section is metadata about the source datasets (HuggingFace IDs, total sizes) and does not control how many episodes the runner executes. The `total_eval_size` field records the full evaluation split size for reference.

Example from `experiments/orz57k/config.yaml`:

```yaml
environment:
  train_size: 500    # 500 Orz57K problems for GEPA optimization
  eval_size: 50      # 50 held-out problems for final evaluation
  # val_size omitted → defaults to 50 (10% of 500)
  dataset:
    eval_split: MATH500
    total_eval_size: 500  # metadata: MATH500 has 500 problems total
```

## Experiment Configs

Primary configs live under `experiments/`:

- `experiments/orz57k/config.yaml` — Orz57K mathematical reasoning benchmark
- `experiments/hotpotqa/config.yaml` — HotpotQA multi-hop QA benchmark
- `experiments/quick-test/config.yaml` — Lightweight config for development/CI

Each config specifies environment settings, task models, reflection model, GEPA budget, replication seeds, evaluation protocol, and cost budget. See `src/trajectory_aware_gym/models/experiment.py` for the full `ExperimentConfig` schema.
