# Publishing Experiment Runs

`poe publish` (wrapping `scripts/publish_run.py`) promotes a completed local
run from `results/` into `docs/07-results/` so finished replications can be
reviewed alongside the rest of the capstone documentation. It also maintains
a top-level `docs/07-results/summary.md` aggregate table that covers every
published run. Publishing is a pure copy step — it never touches the SQLite
trajectories DB, the source `results/` tree, or S3.

## Usage

```bash
# Publish the most recent completed run across the four production configs
poe publish

# Publish the most recent run for a specific config
poe publish --config orz57k-tool

# Publish a specific run (any config)
poe publish --run-dir results/orz57k-tool/20260421T121625Z

# Rebuild summary.md from already-published runs, no copy
poe publish --update-summary-only

# Preview without touching the filesystem
poe publish --dry-run

# Overwrite an existing publish non-interactively (CI)
poe publish --force
```

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-dir` | Path | — | Explicit run directory (`results/{config}/{timestamp}`). Mutually exclusive with `--config` / `--update-summary-only`. |
| `--config` | `hotpotqa-tool\|hotpotqa-notool\|orz57k-tool\|orz57k-notool` | — | Publish the most recent run for this config. |
| `--update-summary-only` | flag | false | Skip the copy; rebuild `docs/07-results/summary.md` from already-published runs. |
| `--dest` | Path | `docs/07-results` | Destination root. |
| `--force` | flag | false | Skip the interactive overwrite confirmation (for CI / non-TTY). |
| `--skip-branch-check` | flag | false | Skip the check that `origin/development` is fully merged into HEAD (offline / CI). |
| `--dry-run` | flag | false | Log the planned copies/removals without touching disk. |

With no scope flag, the script walks `results/{config}/` for the four
production configs and picks the lexicographically latest timestamp (the
filename format `YYYYMMDDTHHMMSSZ` is sortable).

## What Gets Copied

For every `{model}` subdirectory inside the source run, the script copies:

```
results/{config}/{timestamp}/{model}/replication_{seed}/...
results/{config}/{timestamp}/run_summary.json
```

into:

```
docs/07-results/{config}/{model}/{timestamp}/replication_{seed}/...
docs/07-results/{config}/{model}/{timestamp}/run_summary.json
```

The full replication artifact tree is preserved verbatim (see
[production_runner.md](production_runner.md#results-directory-structure) for
the replication-level layout). `run_summary.json` is duplicated under each
model folder so a model directory is self-contained when viewed in isolation.

Any `.gitkeep` sentinel in the target model folder is removed on publish,
since the folder is no longer empty.

## Safety Checks

### Branch-freshness refusal (hard stop)

Before any filesystem work, the script runs `git fetch origin development`
and aborts with exit code `1` if any `origin/development` commit is missing
from `HEAD`. This prevents publishing from a stale feature branch where the
run-health, protocol, or summary-generation logic lags behind `development`.
Output includes the missing commit count and a `git merge origin/development`
hint.

When git is unavailable, the remote can't be reached, or the ref doesn't
exist, the script logs a warning and proceeds rather than block a legitimate
offline run. Pass `--skip-branch-check` to skip this check explicitly in
offline / CI contexts that lack fetch access.

### Protocol-drift refusal (hard stop)

Before the health scan, every replication's `config_snapshot.yaml` is compared
against the canonical per-config protocol. Any mismatch aborts the publish
with exit code `1` and an explanation naming the drifted field, the expected
value, and why that value is canonical. There is **no `--force` override** —
a config tweak has to be reverted (or the run re-executed with the correct
config) before it can be published.

| Config | `val_size` | `eval_size` | `tasks_per_minibatch` |
|---|---|---|---|
| `orz57k-tool` | 300 | 500 (full MATH500 held-out set) | 3 |
| `orz57k-notool` | 300 | 500 (full MATH500 held-out set) | 3 |
| `hotpotqa-tool` | 300 | 512 (full axon-rl/search-eval hotpotqa split) | 3 |
| `hotpotqa-notool` | 300 | 512 (full axon-rl/search-eval hotpotqa split) | 3 |

Example output when someone shrinks the hotpot eval set:

```
ERROR Refusing to publish: replication config drifted from the hotpotqa-tool protocol (no --force override).
ERROR   results/hotpotqa-tool/20260423T011444Z/Gemma-3-4B-IT/replication_42
ERROR     - eval_size=400 but hotpotqa-tool requires 512 (full axon-rl/search-eval hotpotqa split)
```

The same predicate filters replications out of the aggregate `summary.md`, so
a drifted run that somehow slipped into `docs/07-results/` will be silently
excluded from the table on the next `--update-summary-only` refresh.

### Run-health refusal (hard stop)

Before anything else, every replication in the source run is scanned. If any
replication shows one of these, the publish is aborted with exit code `1`:

- `run_metadata.json` missing or unreadable
- `status != "completed"`
- `baseline_eval` / `eval`: `failed`, `timed_out`, or `metrics_unavailable` > 0
- `logging_summary.status != "complete"`
- `logging_summary.trajectory_failed_episodes`, `metrics_unavailable_episodes`, or `numeric_anomaly_count` > 0
- `eval_failure_manifest.jsonl` contains any rows

There is no override. A faulty run has to be re-run (or manually scrubbed at
the source) before it can be published — we don't want error'd data
contaminating `summary.md`.

### Allowed-seed refusal (hard stop)

Replication seeds must be in `{42, 123, 456, 789, 101112}`. Any replication
whose `replication_{seed}` folder name falls outside this set aborts the
publish with exit code `1`. This keeps ad-hoc `seed=0` / `seed=999` smoke
runs out of the summary aggregation. There is no override — re-run with an
allowed seed.

### Seed-conflict overwrite confirmation

For every replication in the source run, the script searches across **all
timestamps** under `docs/07-results/{config}/{model}/` for any existing
`replication_{seed}` folder. If matches are found the script logs:

```
WARNING Seed 42 already published for Llama-3.1-8B-Instruct: docs/07-results/hotpotqa-tool/Llama-3.1-8B-Instruct/20260422T054613Z/replication_42
```

and prompts:

```
Overwrite the existing seed directories above? Type 'yes' to continue:
```

Only the literal `yes` (case-insensitive) proceeds; anything else aborts.
When stdin is not a TTY the script refuses and tells you to re-run with
`--force`. `--force` skips the prompt — use it in CI. On overwrite the old
`replication_{seed}` folder is removed, and if the enclosing timestamp
folder has no remaining replications it is removed entirely (so a stranded
`run_summary.json` sibling does not linger).

In `--dry-run` the prompt is not shown; the script logs that it would prompt
and continues printing the rest of the plan.

### Budget-mode warning

The script reads `gepa_budget_mode` from `run_summary.json` (falling back to
each replication's `run_metadata.json`). When the value is anything other
than `medium` it logs a warning and still publishes:

```
WARNING gepa_budget.mode is 'heavy' (expected 'medium'). Publishing anyway.
```

`medium` is treated as the canonical budget for the four primary configs.
Heavy/light ablations are still publishable — the warning exists so an
accidental non-standard run does not slip in silently. Note that only
medium-budget replications are included in the aggregate `summary.md` table.

## Destination Layout

```
docs/07-results/
├── summary.md                                 # aggregate table (auto-generated)
└── {config}/                                  # hotpotqa-tool | hotpotqa-notool | orz57k-tool | orz57k-notool
    └── {model}/                               # Qwen3-4B-Base, Llama-3.1-8B-Instruct, …
        └── {timestamp}/                       # e.g. 20260422T054613Z
            ├── run_summary.json
            └── replication_{seed}/
                └── …                          # full replication artifact tree
```

The `{config}/{model}/` folders are pre-created (with `.gitkeep`) for all
models listed in each config's `task_models` so the tree is committed even
before any runs have been published.

## Aggregate Summary (`summary.md`)

Every successful `poe publish` — and any explicit `poe publish
--update-summary-only` — regenerates `docs/07-results/summary.md` from
scratch by walking every `run_metadata.json` under `docs/07-results/`.

A replication is **eligible** for the summary when **all** of:

- `gepa_budget_mode == "medium"`
- `_replication_health_issues(metadata)` is empty (same predicate used by
  the publish-time hard stop)
- `_protocol_issues(config, config_snapshot)` is empty — `val_size`,
  `eval_size`, and `tasks_per_minibatch` match the canonical protocol for
  the config (same predicate as the protocol-drift hard stop)
- `eval_failure_manifest.jsonl` has zero rows
- `baseline_eval.accuracy` and `eval.accuracy` are numeric
- seed is in `{42, 123, 456, 789, 101112}`

The table groups eligible replications per `{config}/{model}` and reports:

| Column | Content |
|---|---|
| Runs | Count of eligible replications. `—` if none. |
| Seeds | Comma-separated seed list (ascending). `—` if none. |
| Baseline accuracy | Mean of `baseline_eval.accuracy`; mean ± sample stddev once `n ≥ 2`. |
| Optimized accuracy | Mean of `eval.accuracy`; mean ± sample stddev once `n ≥ 2`. |

Models without eligible replications are still listed (with `—`) so the
table shape matches the config's `task_models` list.

Because `summary.md` is always rebuilt from scratch, it is safe to edit or
delete `{config}/{model}/{timestamp}/` folders directly and then run
`poe publish --update-summary-only` to resync.

## Relationship to Other Post-Run Tooling

- **`poe upload-artifacts`** (`scripts/upload_experiment_artifacts.py`): syncs
  replication artifacts to S3. Independent of `poe publish`. Either, both, or
  neither can be run against a completed replication.
- **SQLite trajectories DB** (`logs/trajectories.db`): canonical trajectory
  storage, never touched by `poe publish`.
- **Experiment run status lifecycle**: `poe publish` is read-only against
  both the source `results/` tree and the DB. Publishing does not change
  `status`, `finished_at`, or any other `experiment_run` field.

## Excluded From Hooks / Lint

`docs/07-results/` is excluded from pre-commit hooks, ruff, and pyright so
large generated JSON/JSONL/CSV/binary artifacts do not trip
`trailing-whitespace`, `check-added-large-files`, or formatter runs. See
`.pre-commit-config.yaml` and `[tool.ruff] / [tool.pyright]` in
`pyproject.toml`.
