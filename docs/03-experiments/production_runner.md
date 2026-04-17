# Production Experiment Runner

The production runner (`scripts/run_experiment.py`) executes GEPA optimization across configured task models, replication seeds, and evaluation protocols. It handles cost tracking, resume on failure, and structured local output. Experiment execution is local-first: each replication writes its artifacts to the results directory and its episode trajectories to the local SQLite DB. Optional S3 sync happens later via `scripts/upload_experiment_artifacts.py`.

## Usage

```bash
poe run-experiment --config experiments/orz57k-tool/config.yaml
```

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | Path | (required) | Path to `ExperimentConfig` YAML |
| `--budget-mode` | `light\|medium\|heavy` | from config's `gepa_budget.mode` | Override GEPA `auto` budget mode (mutually exclusive with `--max-metric-calls`) |
| `--max-metric-calls` | int | from config's `auto` mode | Manually set rollout budget (mutually exclusive with `--budget-mode`) |
| `--seed-prompt` | str | math solver prompt | Initial system prompt for GEPA |
| `--models` | list | all in config | Subset of task model names to run |
| `--seeds` | list | all in config | Subset of replication seeds to run |
| `--fresh` | flag | false | Skip resume; start a new run |
| `--resume` | str | none | Resume a specific run by timestamp (e.g. `20260408T150000Z`) |
| `--danger-purge` | flag | false | Delete ALL prior results for this config (requires typing "yes") |
| `--results-root` | Path | `results/` | Root directory for output |
| `--halt-on-budget-exceeded` | flag | false | Stop if cost exceeds budget |

### GEPA Budget

DSPy's `dspy.GEPA(auto=...)` parameter controls the optimization budget by
candidate count: `light=6`, `medium=12`, `heavy=18`. DSPy then derives the
total rollout budget internally from train/val sizes (matches the GEPA paper's
methodology — see `docs/01-references/GEPA_paper.md`).

The mode lives in the experiment config under `gepa_budget.mode` and defaults
to `heavy` for the four primary configs (orz57k-tool, orz57k-notool,
hotpotqa-tool, hotpotqa-notool). Override per-invocation with `--budget-mode`,
or bypass `auto` entirely with `--max-metric-calls`.

| Source | Budget |
|---|---|
| Config (`gepa_budget.mode: heavy`) + no flags | `auto="heavy"` (18 candidates) |
| `--budget-mode light` | `auto="light"` (6 candidates) |
| `--max-metric-calls 1500` | exactly 1500 metric calls, no auto |

### Examples

```bash
# Use the budget mode set in the config (heavy by default for primary configs)
poe run-experiment --config experiments/orz57k-tool/config.yaml

# Override to light/medium budget at runtime
poe run-experiment \
  --config experiments/orz57k-tool/config.yaml \
  --models Llama-3.1-8B-Instruct \
  --seeds 42 \
  --budget-mode light \
  --fresh

# Manually set rollout count (skips auto entirely)
poe run-experiment \
  --config experiments/orz57k-tool/config.yaml \
  --models Llama-3.1-8B-Instruct \
  --seeds 42 \
  --max-metric-calls 1500 \
  --fresh

# Resume a crashed run (auto-detect most recent incomplete)
poe run-experiment --config experiments/orz57k-tool/config.yaml

# Resume a specific run by timestamp
poe run-experiment \
  --config experiments/orz57k-tool/config.yaml \
  --resume 20260408T150000Z

# Ignore incomplete runs, start fresh
poe run-experiment --config experiments/orz57k-tool/config.yaml --fresh

# Delete all Orz57K results and start over
poe run-experiment --config experiments/orz57k-tool/config.yaml --danger-purge
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
                ├── run_metadata.json    # Replication status, experiment_run_id, timing, validation + eval summaries
                ├── config_snapshot.yaml # Effective runtime config (YAML + CLI overrides)
                ├── optimized_prompt.txt # GEPA-evolved system prompt
                ├── fitness_history.json # Fitness scores per GEPA iteration
                ├── pareto_frontier.json # Pareto frontier of candidate prompts
                ├── cost_summary.json    # Token/cost breakdown (task vs reflection, train vs eval)
                ├── run_report.json      # Unified summary (identical keys across providers)
                ├── training_metrics.csv  # Per-episode metrics (training only)
                ├── training_metrics_summary.json  # Aggregated training-phase stats
                ├── raw_metrics.csv      # Per-episode metrics (held-out eval only)
                ├── raw_metrics.jsonl    # Same data, JSONL format
                ├── raw_metrics_summary.json  # Held-out eval stats split by baseline vs optimized
                ├── upload_manifest.json # Written only by the separate S3 sync script
                └── gepa_logs/           # GEPA internal optimization logs
```

## Experiment Run Tracking

Each replication is registered in the SQLite database (`logs/trajectories.db`) as an `experiment_run` record with a deterministic ID. That ID is also written into `run_metadata.json` immediately so resume can keep later evaluation, reporting, and any later artifact sync attached to the same logical replication.

### experiment_run_id Format

`{config_name}-{provider}-{model_short}-{operator}-seed{seed}-{YYYYMMDD}T{HHMM}Z`

Example:

`orz57k-tool-ollama-qwen3-1-7b-base-edward-seed42-20260417T0421Z`

### Lifecycle

```
save_experiment_run(status="running")
  → GEPA optimization
update_experiment_run(status="gepa_done", optimized_prompt=...)
  → Evaluation
update_experiment_run(
  status="completed",
  finished_at=...,
  result_summary=...,
  cost_summary=...,
  logging_summary=...,
)
```

On exception:

```
update_experiment_run(
  status="failed",
  finished_at=...,
  error_summary=...,
  logging_summary=...,
)
```

If a replication resumes from `status="gepa_done"`, the runner reloads the persisted `experiment_run_id` from `run_metadata.json` and continues updating the same DB row and local replication folder.

Experiment-run registry updates and run-report publication are **best-effort** — failures log a warning and the experiment continues. Per-episode logging/persistence degradation is recorded in `logging_summary` instead of forcing fabricated values into the results.

### Local-First Artifact Sync

The runner itself never uploads to S3. To sync completed local results later, use:

```bash
poe upload-artifacts -- --replication-dir results/.../replication_42
poe upload-artifacts -- --run-dir results/orz57k-tool/20260408T150000Z
poe upload-artifacts -- --config-dir results/orz57k-tool
```

The upload script scans local replication folders, requires `run_metadata.json` to show `status: "completed"`, uploads the local artifact files (not the SQLite DB), and writes `upload_manifest.json` beside the artifacts with uploaded, skipped-existing, and failed keys.

### Synced Artifacts

| Artifact | Description |
|----------|-------------|
| `config_snapshot.yaml` | Effective runtime config, including CLI overrides such as `--seed-prompt`, `--budget-mode`, or `--max-metric-calls` |
| `run_metadata.json` | Replication status, `experiment_run_id`, timing, and eval summary |
| `cost_summary.json` | Token/cost breakdown |
| `optimized_prompt.txt` | GEPA-evolved system prompt |
| `fitness_history.json` | Fitness scores per GEPA iteration |
| `run_report.json` | Unified cross-provider summary with provider, accuracy, cost, timing, and logging fields |

S3 key format: `{s3_prefix}{experiment_run_id}/{filename}`

Upload failures are isolated to the sync script and are recorded in `upload_manifest.json`; they never change experiment completion status.

### run_report.json

A unified summary with identical JSON keys regardless of provider (Bedrock or Ollama). Includes experiment metadata, performance summaries, actual/partial/unavailable cost semantics, normalized Ollama proxy cost when task-model token coverage is complete, timing, logging summary, and git info. `mean_llm_latency_ms` is scoped to trajectories linked to the current `experiment_run_id`, not every run in the same environment. Accuracy uses stored `episode_outcome` when available rather than positive reward heuristics.

Key fields:

- `experiment_run_id`, `config_name`, `provider`, `task_model_id`, `environment_id`, `seed`
- `baseline_eval`, `eval_summary`
- `total_tokens`, `total_tokens_known`, `task_model_cost_usd`, `task_model_cost_known_usd`
- `reflection_cost_usd`, `total_cost_usd`, `total_cost_known_usd`, `cost_type`
- `normalized_cost_usd`, `normalization_reference`
- `wall_clock_seconds`, `mean_llm_latency_ms`
- `git_commit`, `started_at`, `finished_at`, `logging_summary`

## Resume Behavior

The runner writes `run_summary.json` at the **start** of each invocation (with `finished_at: null`) and updates it on completion (with `finished_at` set). This enables resume detection:

1. **Default**: scans for the most recent timestamp directory whose `run_summary.json` has no `finished_at`. If found, resumes into that directory. Completed replications (where `run_metadata.json` has `status: "completed"`) are skipped; incomplete ones re-run.
2. **`--fresh`**: ignores any incomplete runs, creates a new timestamp directory.
3. **`--resume <timestamp>`**: resumes a specific run by its timestamp.
4. **`--danger-purge`**: deletes `results/{config_name}/` entirely before starting.

Resume safety checks compare the hash of the effective runtime config, not just the raw YAML file. Changing CLI overrides between runs (for example `--seed-prompt` or `--max-metric-calls`) causes resume to fail fast instead of silently mixing incompatible outputs.

## Key Output Files

### run_summary.json

Written at the experiment level (one per invocation). Contains `run_id`, `config_hash`, `git_commit`, per-model results, and timing. `config_hash` is computed from the effective runtime snapshot, so CLI overrides change the run identity.

### run_metadata.json

Written per replication. Tracks status (`running` / `gepa_done` / `completed` / `failed`), `experiment_run_id`, timing, the GEPA result summary, eval scores, `logging_summary`, and `error` on failure. The runner preserves the original `started_at` and `experiment_run_id` when resuming a partially completed replication.

### cost_summary.json

Per-replication cost breakdown:

| Field | Description |
|-------|-------------|
| `task_model_tokens` / `task_model_cost` | Complete task-model totals when coverage is 100% |
| `task_model_tokens_known` / `task_model_cost_known` | Known task-model totals even when some episodes have missing data |
| `task_model_token_data_coverage` / `task_model_cost_data_coverage` | Fraction of episodes with complete task-model token/cost data |
| `reflection_tokens` / `reflection_cost` | Reflection model usage during GEPA |
| `total_tokens` / `total_cost` | Complete combined totals when task-model coverage is 100% |
| `total_tokens_known` / `total_cost_known` | Combined known totals when task-model data is partial |
| `cost_type` | `actual`, `partial`, or `unavailable` for the run-level cost summary |
| `training_task_model_tokens` / `eval_task_model_tokens` | Train vs eval split |
| `effective_budget_usd` | Budget from config |

### raw_metrics.csv / .jsonl

One row per held-out evaluation episode (baseline eval + optimized eval). Fields match the `EpisodeRawMetrics` schema documented in [phase3_raw_metrics.md](phase3_raw_metrics.md).

### training_metrics.csv / .jsonl

One row per GEPA optimization-phase episode. This is the right artifact for inspecting train/validation-side behavior during prompt search.

## Data Splits

The experiment config controls three distinct data partitions via the `environment` section:

| Field | Location | Purpose | Default |
|-------|----------|---------|---------|
| `train_size` | `environment` | Number of training examples for GEPA optimization | (required) |
| `val_size` | `environment` | Validation set size (subset of trainset) | 10% of `train_size` |
| `eval_size` | `environment.dataset` | Held-out evaluation set size (never seen during optimization) | — |

Seed ranges ensure no overlap between the three disjoint splits
(matches the GEPA paper's train/val/test convention):

- **Train**: seeds `[data_seed, data_seed + train_size)`
- **Val**:   seeds `[data_seed + train_size, data_seed + train_size + val_size)`
- **Eval**:  seeds `[data_seed + train_size + val_size, data_seed + train_size + val_size + eval_size)`

Example from `experiments/orz57k-tool/config.yaml` (with-tool variant):

```yaml
environment:
  train_size: 500    # 500 Orz57K problems for GEPA optimization
  # val_size omitted → defaults to 50 (10% of 500)
  dataset:
    eval_split: MATH500
    eval_size: 500   # 500 MATH500 problems for held-out eval
```

## Experiment Configs

Primary configs live under `experiments/`:

- `experiments/orz57k-tool/config.yaml` — Orz57K with Python tool
- `experiments/orz57k-notool/config.yaml` — Orz57K without tools
- `experiments/hotpotqa-tool/config.yaml` — HotpotQA with search tool
- `experiments/hotpotqa-notool/config.yaml` — HotpotQA without tools
- `experiments/quick-test/config.yaml` — Lightweight config for development/CI

Each config specifies environment settings, task models, reflection model, GEPA budget, replication seeds, evaluation protocol, and cost budget. See `src/trajectory_aware_gym/models/experiment.py` for the full `ExperimentConfig` schema.
