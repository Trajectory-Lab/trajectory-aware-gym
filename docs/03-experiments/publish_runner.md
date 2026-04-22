# Publishing Experiment Runs

`poe publish` (wrapping `scripts/publish_run.py`) promotes a completed local
run from `results/` into `docs/07-results/` so finished replications can be
reviewed alongside the rest of the capstone documentation. It is a pure copy
step — it never touches the SQLite trajectories DB, the source `results/`
tree, or S3.

## Usage

```bash
# Publish the most recent completed run across the four production configs
poe publish

# Publish the most recent run for a specific config
poe publish --config orz57k-tool

# Publish a specific run (any config)
poe publish --run-dir results/orz57k-tool/20260421T121625Z

# Preview without touching the filesystem
poe publish --dry-run

# Replace an existing publish for the same {config}/{model}/{timestamp}
poe publish --force
```

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-dir` | Path | — | Explicit run directory (`results/{config}/{timestamp}`). Mutually exclusive with `--config`. |
| `--config` | `hotpotqa-tool\|hotpotqa-notool\|orz57k-tool\|orz57k-notool` | — | Publish the most recent run for this config. |
| `--dest` | Path | `docs/07-results` | Destination root. |
| `--force` | flag | false | Overwrite existing `{config}/{model}/{timestamp}` folders. |
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

### Existing publish refusal

Before copying anything the script walks every planned
`{config}/{model}/{timestamp}` path. If any already exist it logs a
`WARNING` per path, prints

```
Refusing to overwrite. Re-run with --force to replace the existing publish.
```

and exits with status `1` without touching the filesystem. This applies to
both real publishes and `--dry-run`.

### Budget-mode warning

The script reads `gepa_budget_mode` from `run_summary.json` (falling back to
each replication's `run_metadata.json`). When the value is anything other
than `medium` it logs a warning and still publishes:

```
WARNING gepa_budget.mode is 'heavy' (expected 'medium'). Publishing anyway.
```

`medium` is treated as the canonical budget for the four primary configs.
Heavy/light ablations are still publishable — the warning exists so an
accidental non-standard run does not slip in silently.

## Destination Layout

```
docs/07-results/
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
