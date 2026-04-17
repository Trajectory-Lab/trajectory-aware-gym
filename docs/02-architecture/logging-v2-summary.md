# Logging V2: Change Summary

**Branch:** `feat/logging-v2`
**Schema version:** 1.3.0 (bumped from 1.1.0)
**Date:** 2026-04-16

---

## 1. Schema Changes

### New table: `experiment_runs` (24 columns)

Provides experiment-level registry enabling cross-experiment queries by config, operator, provider, and status.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `experiment_run_id` | TEXT | PRIMARY KEY | Deterministic ID from naming protocol (see §2) |
| `config_name` | TEXT | NOT NULL | Experiment config name (e.g., `orz57k-tool`) |
| `config_hash` | TEXT | NOT NULL | SHA-256 of the effective runtime config snapshot (YAML plus any CLI overrides) |
| `config_yaml` | TEXT | NOT NULL | Full effective YAML snapshot written at run start |
| `operator` | TEXT | NOT NULL | `git config user.name` or `$USER` fallback |
| `git_commit` | TEXT | nullable | HEAD commit hash at run start |
| `git_branch` | TEXT | nullable | Current branch at run start |
| `provider` | TEXT | NOT NULL | `"ollama"`, `"bedrock"`, or `"sagemaker"` |
| `task_model_id` | TEXT | NOT NULL | Full LiteLLM model string |
| `reflection_model_id` | TEXT | nullable | Bedrock model used for GEPA reflection |
| `environment_id` | TEXT | NOT NULL | GEM environment (e.g., `orz57k`, `hotpotqa`) |
| `gepa_budget_mode` | TEXT | nullable | `"light"` / `"medium"` / `"heavy"` |
| `replication_seed` | INTEGER | nullable | Random seed for this replication |
| `seed_prompt` | TEXT | nullable | Initial system prompt before optimization |
| `optimized_prompt` | TEXT | nullable | Best prompt after GEPA optimization |
| `started_at` | TEXT | NOT NULL | ISO 8601 timestamp |
| `finished_at` | TEXT | nullable | ISO 8601 timestamp (set on completion/failure) |
| `status` | TEXT | NOT NULL, DEFAULT 'running' | Lifecycle: `running` → `gepa_done` → `completed` \| `failed` |
| `hostname` | TEXT | nullable | Machine hostname |
| `result_summary` | TEXT | nullable | JSON blob with eval metrics |
| `cost_summary` | TEXT | nullable | JSON blob with cost breakdown |
| `error_summary` | TEXT | nullable | Failure reason mirrored from `run_metadata.json["error"]` |
| `logging_summary` | TEXT | nullable | JSON blob summarizing logging/persistence degradation |
| `schema_version` | TEXT | NOT NULL | Schema version at write time (`1.3.0`) |

**Indexes:**
- `idx_experiment_runs_config` on `config_name`
- `idx_experiment_runs_operator` on `operator`

**Migration behavior:**
- Existing SQLite files are migrated in place on first open before v1.2 indexes are created. Legacy `episodes`, `llm_calls`, and `tool_calls` tables are patched with the missing columns required by logging-v2 writes.

### Modified table: `episodes`

| Change | Description |
|--------|-------------|
| New column: `experiment_run_id` | `TEXT REFERENCES experiment_runs(experiment_run_id)`, nullable for backward compat |
| New index: `idx_episodes_experiment_run` | On `experiment_run_id` for FK lookups |

### Modified table: `llm_calls`

| Change | Description |
|--------|-------------|
| New column: `provider` | `TEXT`, nullable. Derived from `model_id` prefix (`ollama/` → `"ollama"`, etc.) |
| New column: `cost_type` | `TEXT`, nullable. Current writer values: `"actual"` (LiteLLM has pricing) or `"unavailable"` (no per-call pricing available) |
| Now populated: `latency_ms` | Was always `None`; now set via `time.monotonic()` around LLM calls |
| Now populated: `cost_usd` | Was 0.0 for Ollama; now `None` when `cost_type = "unavailable"` |

### Modified table: `tool_calls`

| Change | Description |
|--------|-------------|
| Now populated: `duration_ms` | Was always `None`; now set via `time.monotonic()` around tool execution |

---

## 2. Run Naming Protocol

**Module:** `src/trajectory_aware_gym/storage/naming.py`

### Format

```
{config_name}-{provider}-{model_short}-{operator}-seed{seed}-{YYYYMMDD}T{HHMM}Z
```

### Examples

```
orz57k-tool-bedrock-llama8b-jinyu-seed42-20260415T1430Z
hotpotqa-tool-ollama-qwen3-1-7b-base-alice-seed1-20260416T0900Z
orz57k-notool-sagemaker-qwen3-4b-base-bob-seed99-20260417T2200Z
```

### Functions

| Function | Description |
|----------|-------------|
| `generate_experiment_run_id(config_name, provider, model_id, operator, seed, timestamp)` | Produces the deterministic run ID |
| `get_operator()` | `git config user.name` → `$USER` → `"unknown"` fallback chain |
| `get_git_info()` | Reads `.git/HEAD` directly (no subprocess); returns `(commit, branch)` tuple; handles symbolic refs, detached HEAD, and packed-refs fallback |

### Sanitization Rules

- All segments lowercased
- Non-alphanumeric characters replaced with hyphens
- Model ID has provider prefix stripped before sanitization (e.g., `bedrock/us.meta.llama3-1-8b-instruct-v1:0` → `us-meta-llama3-1-8b-instruct-v1-0`)

---

## 3. Local-First Artifact Sync

**Module:** `src/trajectory_aware_gym/storage/s3_upload.py`

Experiment execution is local-first: every replication writes canonical artifacts into its local `results/.../replication_{seed}/` folder, and trajectories are persisted only to the local SQLite DB. Optional S3 sync happens later via `scripts/upload_experiment_artifacts.py`.

### Key Format

```
{s3_prefix}{experiment_run_id}/{filename}
```

Default prefix from YAML: `experiments/`

### Example

```
s3://trajectory-aware-gym-results/experiments/orz57k-tool-bedrock-llama8b-jinyu-seed42-20260415T1430Z/config_snapshot.yaml
s3://trajectory-aware-gym-results/experiments/orz57k-tool-bedrock-llama8b-jinyu-seed42-20260415T1430Z/run_metadata.json
s3://trajectory-aware-gym-results/experiments/orz57k-tool-bedrock-llama8b-jinyu-seed42-20260415T1430Z/cost_summary.json
```

### Functions

| Function | Description |
|----------|-------------|
| `upload_artifact_bundle(experiment_run_id, artifacts, *, bucket, prefix, client_config)` | Uploads `dict[str, Path]` of artifacts. Immutable: check-before-write via S3 HEAD request. Skips existing keys with a logged warning. |
| `upload_artifact_bundle_detailed(experiment_run_id, artifacts, *, bucket, prefix, client_config)` | Same upload path, but returns uploaded/skipped/failed key detail for manifest generation |
| `list_remote_runs(*, bucket, prefix, client_config)` | Paginated S3 prefix listing; returns `list[str]` of run IDs |
| `download_artifact(experiment_run_id, filename, dest_dir, *, bucket, prefix, client_config)` | Downloads a single file; raises `FileNotFoundError` for missing S3 keys |

### Design Decisions

- **Lazy boto3 import:** `boto3` is imported inside `_get_s3_client()` to avoid a hard dependency for local-only workflows.
- **Immutability:** Uses HEAD request to check key existence before PUT. This is a lightweight courtesy check, not a locking mechanism. True immutability would require S3 Object Lock (out of scope for v2).
- **Decoupled failure domain:** upload failures never affect experiment completion because the runner does not upload directly. The sync script records uploaded/skipped/failed keys in `upload_manifest.json`.

---

## 4. Cost Normalization

**Module:** `src/trajectory_aware_gym/metrics/cost_normalization.py`
**Config section:** `cost_normalization.reference_prices` in YAML

### Purpose

Ollama models run locally and have no API pricing. To support H2 (compute efficiency comparison), we map Ollama token counts to Bedrock-equivalent USD using reference prices from comparable models.

### Formula

```
normalized_cost = (prompt_tokens × input_per_1m / 1,000,000)
                + (completion_tokens × output_per_1m / 1,000,000)
```

Returns `None` if the model_id is not in the reference price table.

### Current Reference Prices

| Ollama Model | Reference Bedrock Model | Input $/1M | Output $/1M | Source |
|---|---|---|---|---|
| `ollama/qwen3-1.7b-base` | `us.meta.llama3-2-1b-instruct-v1:0` | $0.10 | $0.10 | AWS Bedrock pricing, 2026-04, us-east-1 |
| `ollama/qwen3-4b-base` | `us.meta.llama3-2-3b-instruct-v1:0` | $0.22 | $0.22 | AWS Bedrock pricing, 2026-04, us-east-1 |

### How to Update

When AWS pricing changes:
1. Check [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) for the target region
2. Update `cost_normalization.reference_prices` in `src/trajectory_aware_gym/config/trajectory-aware-gym.yaml`
3. Update the source comment (date, region) next to each entry
4. Run `poe test` to verify config loading

### Disclaimer

Normalized costs are **proxy estimates**, not actual cloud spend. They are used strictly for cross-provider efficiency comparisons in the paper. The mapping assumes parameter-count proximity implies cost proximity, which is an approximation.

---

## 5. Cost Type Semantics

The `cost_type` field on `LLMCallMetadata` and in the `llm_calls` DB table captures how `cost_usd` was determined for each stored call:

| Value | Meaning | When Set |
|-------|---------|----------|
| `"actual"` | Real API pricing from `litellm.completion_cost()` | Bedrock, SageMaker — LiteLLM has pricing tables |
| `"unavailable"` | No pricing data available; `cost_usd = None` | `completion_cost()` raises an exception |

For Ollama runs, normalized proxy cost is computed later in `run_report.json` as `normalized_cost_usd`; it is not written back into each `llm_calls` row.

### Provider Detection

Provider is auto-derived from `model_id` prefix via a `model_validator` on `LLMCallMetadata`:

| Prefix | Provider |
|--------|----------|
| `ollama/` | `"ollama"` |
| `bedrock/` | `"bedrock"` |
| `sagemaker/` | `"sagemaker"` |
| (other) | `None` |

---

## 6. Unified Run Report

**Module:** `src/trajectory_aware_gym/metrics/run_report.py`

### Purpose

Produces identically structured JSON summaries regardless of whether the experiment ran on Bedrock (actual pricing) or Ollama (normalized proxy pricing), so paper figures and tables can consume them uniformly.

### RunReport Fields

| Field | Type | Description |
|-------|------|-------------|
| `experiment_run_id` | str | Deterministic ID from naming protocol (§2) |
| `config_name` | str | Experiment config name |
| `operator` | str | Who ran the experiment |
| `provider` | str | `"bedrock"`, `"ollama"`, or `"sagemaker"` |
| `task_model_id` | str | Full LiteLLM model string |
| `environment_id` | str | GEM environment ID |
| `seed` | int \| None | Replication seed |
| `baseline_eval` | dict \| None | Pre-optimization evaluation metrics |
| `eval_summary` | dict \| None | Post-optimization evaluation metrics |
| `total_tokens` | int \| None | Combined task + reflection tokens when task-model coverage is complete |
| `total_tokens_known` | int \| None | Combined known task + reflection tokens when task-model coverage is partial |
| `task_model_cost_usd` | float \| None | Task model cost (actual or None) |
| `task_model_cost_known_usd` | float \| None | Known task-model spend when actual cost is partial |
| `task_model_token_data_coverage` | float \| None | Fraction of episodes with complete task-model token data |
| `task_model_cost_data_coverage` | float \| None | Fraction of episodes with complete task-model cost data |
| `reflection_cost_usd` | float \| None | Reflection model cost |
| `total_cost_usd` | float \| None | Combined cost when task-model coverage is complete |
| `total_cost_known_usd` | float \| None | Combined known cost when task-model coverage is partial |
| `cost_type` | str \| None | `"actual"`, `"partial"`, or `"unavailable"` |
| `normalized_cost_usd` | float \| None | Bedrock-equivalent proxy cost (Ollama only, when task-model token coverage is complete) |
| `normalization_reference` | str \| None | Human-readable pricing source (e.g., `"ollama/qwen3-1.7b-base @ $0.1/1M"`) |
| `wall_clock_seconds` | float \| None | Total replication wall time |
| `mean_llm_latency_ms` | float \| None | Mean per-call LLM latency from trajectories linked to this `experiment_run_id` |
| `git_commit` | str \| None | HEAD commit at run start |
| `started_at` | str \| None | ISO 8601 start timestamp |
| `finished_at` | str \| None | ISO 8601 end timestamp |
| `logging_summary` | dict \| None | Run-level summary of logging anomalies, missing metrics, and persistence failures |

### Builder Function

```python
build_run_report(
    experiment_run_id: str,
    db_path: Path,
    cost_summary: dict[str, Any],
    baseline_eval_summary: dict[str, Any] | None = None,
    eval_summary: dict[str, Any] | None = None,
    wall_clock_seconds: float | None = None,
    reference_prices: dict[str, dict[str, float]] | None = None,
) -> RunReport
```

Loads the `ExperimentRunRecord` from the DB, then assembles cost, timing, eval data, and logging summary from caller-provided summaries. For Ollama models, computes `normalized_cost_usd` using a 70/30 prompt/completion token split approximation only when complete task-model token totals are available. `mean_llm_latency_ms` is derived only from trajectories whose `episodes.experiment_run_id` matches the report being built.

### Resume semantics

- `run_metadata.json` persists `experiment_run_id` as soon as a replication starts.
- If a replication is resumed after `status = "gepa_done"`, the runner reuses that persisted `experiment_run_id` so evaluation trajectories, the final `experiment_runs` row update, and any later S3 artifact sync all stay attached to one logical replication.

### Output Location

Written to `replication_dir/run_report.json`. If artifacts are synced later, the upload script includes it in the local-first artifact bundle (§3).

---

## 7. Assumptions & Justifications

| Assumption | Justification |
|---|---|
| **Reference pricing: Qwen3-1.7B ≈ Llama-1B** | Closest parameter-count match available on Bedrock. Both are small instruction models. |
| **Reference pricing: Qwen3-4B ≈ Llama-3B** | Same rationale. Next-smallest Bedrock model. |
| **Provider detection via model_id prefix** | LiteLLM convention: all model strings use `provider/model-name` format. Stable across versions. |
| **Operator detection via git config** | `git config user.name` is set by all contributors as part of onboarding. `$USER` fallback covers CI. |
| **S3 immutability via HEAD check** | Lightweight and sufficient for our use case (sequential experiment runs, not concurrent uploads of the same ID). |
| **Nullable FK on episodes** | Backward compatibility with pre-v1.2 databases that have episodes without experiment runs. |
| **cost_type defaults to None** | Backward compat: old LLMCallMetadata instances (pre-v1.2) will have `cost_type=None`, distinguishable from `"unavailable"`. |

---

## 8. New & Modified Files

### New Files

| File | Description |
|------|-------------|
| `src/trajectory_aware_gym/storage/models.py` | `ExperimentRunRecord`, `EpisodeLoggingSummary`, and `LoggingSummary` Pydantic models |
| `src/trajectory_aware_gym/storage/naming.py` | Run naming protocol, operator/git info helpers |
| `src/trajectory_aware_gym/storage/s3_upload.py` | S3 artifact upload/download/listing |
| `src/trajectory_aware_gym/metrics/cost_normalization.py` | `compute_normalized_cost()` utility |
| `src/trajectory_aware_gym/metrics/run_report.py` | `RunReport` Pydantic model + `build_run_report()` builder |
| `scripts/upload_experiment_artifacts.py` | Separate post-run S3 sync entrypoint writing `upload_manifest.json` |
| `tests/unit/test_run_report.py` | RunReport tests including partial-cost and logging-summary cases |
| `tests/unit/test_naming.py` | 24 tests for naming protocol |
| `tests/unit/test_cost_normalization.py` | 10 tests for cost normalization + Settings |
| `tests/unit/test_s3_upload.py` | 8 tests with boto3 mock |
| `tests/unit/test_upload_experiment_artifacts.py` | Local-first upload script and manifest tests |
| `docs/02-architecture/logging-v2-plan.md` | Master plan document |
| `docs/02-architecture/logging-v2-summary.md` | This document |

### Modified Files

| File | Changes |
|------|---------|
| `src/trajectory_aware_gym/adapters/gem_episode_runner.py` | `time.monotonic()` wraps for latency_ms and duration_ms; logging degradation is recorded but non-fatal for eval/persistence; cost_type tracking in both code paths |
| `src/trajectory_aware_gym/adapters/trajectory_logger.py` | Schema version bump to 1.3.0; `provider` field + `model_validator`; `cost_type` field on `LLMCallMetadata`; unknown cost no longer collapses to zero |
| `src/trajectory_aware_gym/storage/trajectory_db.py` | `experiment_runs` table; `provider`/`cost_type` columns on `llm_calls`; experiment_run FK on `episodes`; additive migration for `error_summary` and `logging_summary`; CRUD functions |
| `src/trajectory_aware_gym/storage/__init__.py` | Expanded public storage exports (DB models, CRUD, naming, and S3 helpers) |
| `src/trajectory_aware_gym/config/core.py` | `CostNormalizationModel`; wired into Settings |
| `src/trajectory_aware_gym/experiments/runner.py` | Experiment run DB lifecycle, local-first artifact flow, faithful cost summaries, RunReport generation, deprecated `--skip-s3-upload` handling |
| `scripts/run_experiment.py` | Deprecated `--skip-s3-upload` CLI flag help text |
| `src/trajectory_aware_gym/config/trajectory-aware-gym.yaml` | `cost_normalization` section with reference prices |
| `tests/unit/test_experiment_runner.py` | Regression tests for DB lifecycle, local-first flow, failure persistence, and RunReport write |
| `tests/unit/test_trajectory_db.py` | Provider/cost_type round-trip tests; 16 experiment_run CRUD tests |
| `tests/unit/test_gem_episode_runner.py` | Latency, cost_type, duration_ms assertions |
| `tests/unit/test_trajectory_logger.py` | Provider derivation tests |
| `AGENTS.md` | Updated "Trajectory Storage" section |
| `docs/02-architecture/configuration.md` | Added cost_normalization, retry, new Bedrock models, new fitness fields |
