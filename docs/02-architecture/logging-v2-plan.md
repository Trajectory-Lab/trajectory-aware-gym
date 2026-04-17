# Logging V2: Experiment Registry, Data Accuracy & Remote Storage

**Branch:** `feat/logging-v2`
**Base:** `development`
**Author:** Jinyu Han
**Created:** 2026-04-15

---

## Motivation

The current logging system has a solid per-episode trajectory DB but lacks:

1. **Experiment-level metadata** — no way to query "all episodes from Jinyu's orz57k heavy run on Bedrock Llama-8B"
2. **Data accuracy gaps** — `latency_ms` never populated, `ToolCall.duration_ms` never populated, Ollama cost silently reported as 0.0 instead of "unavailable"
3. **Remote accessibility** — SQLite is local-only; collaborators can't see each other's results
4. **Unified naming** — three unrelated ID schemes (UUID episode, `{config}-{ts}` run, filesystem path)

This plan addresses all four in two phases, with Phase 1 independent of PR #149.

---

## Terminology

| Term | Meaning |
|---|---|
| **experiment_run** | One invocation of the runner: a specific config + model + seed + operator combination |
| **episode** | One GEM environment episode (reset → terminal); many episodes per experiment_run |
| **artifact bundle** | The immutable set of files produced by one experiment_run (config YAML, summary, optimized prompt, cost report) |

---

## Phase 1 — Independent of PR #149 (start immediately)

These tasks touch `storage/`, `adapters/`, `config/`, and new files only. Zero overlap with `runner.py` or `models/experiment.py`.

### Task 1: Fix LLM call latency tracking

**Files:** `src/trajectory_aware_gym/adapters/gem_episode_runner.py`
**Why:** `LLMCallMetadata.latency_ms` exists in schema + DB but is never populated. Critical for H2 (compute efficiency comparison).

- [x] 1.1 In `generate_smoke_action()`, wrap `_completion_with_retry()` with `time.monotonic()` before/after. Compute `latency_ms = (after - before) * 1000`. Pass it to `LLMCallMetadata(latency_ms=latency_ms)`.
- [x] 1.2 Unit test: mock `_completion_with_retry`, assert `LLMCallMetadata.latency_ms` is a positive float (not None).
- [x] 1.3 Unit test: existing tests still pass (latency is additive, doesn't break existing assertions).

**Task 1 status: DONE.** Smooth. Added `time.monotonic()` wrap in `generate_smoke_action()` (3 lines), added `import time`. Extended existing `test_run_episode_records_trajectory_and_cost` to assert `latency_ms > 0`. All 4 tests pass.

### Task 2: Fix tool call duration tracking

**Files:** `src/trajectory_aware_gym/adapters/gem_episode_runner.py`
**Why:** `ToolCall.duration_ms` exists in schema + DB but is never populated.

- [x] 2.1 In `_run_agent_step()`, wrap `self._tool_runtime.execute(parsed_tool_call)` with `time.monotonic()` before/after. Compute `duration_ms = (after - before) * 1000`. Pass it to `ToolCall(duration_ms=duration_ms)`.
- [x] 2.2 Unit test: mock `ToolRuntime.execute`, assert `ToolCall.duration_ms` is a positive float (not None).

**Task 2 status: DONE.** Smooth. Added `time.monotonic()` wrap around `_tool_runtime.execute()` (3 lines). Extended existing `test_run_episode_executes_tool_call_before_final_action` to assert `duration_ms >= 0`. All 4 tests pass.

### Task 3: Add `provider` column to `llm_calls` table

**Files:** `src/trajectory_aware_gym/storage/trajectory_db.py`, `src/trajectory_aware_gym/adapters/trajectory_logger.py`
**Why:** Need to distinguish Ollama/Bedrock/SageMaker calls in DB for cost analysis.

- [x] 3.1 Add `provider: str | None = None` field to `LLMCallMetadata` in `trajectory_logger.py`. Derive provider from `model_id` prefix: `"ollama/"` → `"ollama"`, `"bedrock/"` → `"bedrock"`, `"sagemaker/"` → `"sagemaker"`, else `None`.
- [x] 3.2 Add `provider TEXT` column to `llm_calls` table in `_SCHEMA_SQL`. It goes after `model_id`. Nullable for backward compat.
- [x] 3.3 Update `save_trajectory()` INSERT for `llm_calls` to include `provider`.
- [x] 3.4 Update `_build_trajectory()` to read `provider` from row and pass to `LLMCallMetadata`.
- [x] 3.5 Bump `SCHEMA_VERSION` to `"1.2.0"` in `trajectory_logger.py`.
- [x] 3.6 Unit test: save a trajectory with `model_id="bedrock/llama-8b"`, load it back, assert `provider == "bedrock"`.
- [x] 3.7 Unit test: save a trajectory with `model_id="ollama/qwen3-1.7b"`, assert `provider == "ollama"`.
- [x] 3.8 Unit test: save with `model_id="some-unknown-model"`, assert `provider is None`.

**Task 3 status: DONE.** Smooth. Added `_derive_provider()` helper + `model_validator` on `LLMCallMetadata` for auto-derivation. Schema bumped to 1.2.0. DB read uses safe `"provider" in row.keys()` fallback for old databases. 161 tests pass (10 new).

### Task 4: Add `cost_type` column to `llm_calls` table

**Files:** `src/trajectory_aware_gym/storage/trajectory_db.py`, `src/trajectory_aware_gym/adapters/trajectory_logger.py`, `src/trajectory_aware_gym/adapters/gem_episode_runner.py`
**Why:** Distinguish "real API cost" from "no pricing available" — currently Ollama cost is silently 0.0 via `cost_usd or 0.0` aggregation.

- [x] 4.1 Add `cost_type: Literal["actual", "estimated", "unavailable"] | None = None` field to `LLMCallMetadata`.
- [x] 4.2 Add `cost_type TEXT` column to `llm_calls` table in `_SCHEMA_SQL`. Nullable.
- [x] 4.3 Update `save_trajectory()` and `_build_trajectory()` for the new column.
- [x] 4.4 In `generate_smoke_action()`, set `cost_type`:
    - If `completion_cost()` succeeds and returns > 0 → `"actual"`
    - If `completion_cost()` succeeds and returns 0.0 → `"actual"` (genuinely free/zero-cost)
    - If `completion_cost()` raises exception → `"unavailable"`, `cost_usd=None`
- [x] 4.5 Unit test: bedrock model → `cost_type == "actual"`.
- [x] 4.6 Unit test: ollama model (completion_cost raises) → `cost_type == "unavailable"`, `cost_usd is None`.

**Task 4 status: DONE.** Smooth. Added `cost_type` field to model, DB schema, save/load. Added cost_type setting in `generate_smoke_action()`. 166 tests pass (5 new: 1 runner + 4 DB round-trip parametrized).

### Task 5: Add `experiment_runs` table to DB schema

**Files:** `src/trajectory_aware_gym/storage/trajectory_db.py`
**Why:** Central experiment-level registry enabling cross-experiment queries.

- [x] 5.1 Design and add `experiment_runs` table to `_SCHEMA_SQL`:
    ```sql
    CREATE TABLE IF NOT EXISTS experiment_runs (
        experiment_run_id   TEXT PRIMARY KEY,
        config_name         TEXT NOT NULL,
        config_hash         TEXT NOT NULL,
        config_yaml         TEXT NOT NULL,        -- full YAML snapshot
        operator            TEXT NOT NULL,         -- git user.name
        git_commit          TEXT,
        git_branch          TEXT,
        provider            TEXT NOT NULL,         -- "ollama" / "bedrock" / "sagemaker"
        task_model_id       TEXT NOT NULL,
        reflection_model_id TEXT,
        environment_id      TEXT NOT NULL,
        gepa_budget_mode    TEXT,                  -- "light" / "medium" / "heavy"
        replication_seed    INTEGER,
        seed_prompt         TEXT,
        optimized_prompt    TEXT,
        started_at          TEXT NOT NULL,
        finished_at         TEXT,
        status              TEXT NOT NULL DEFAULT 'running',
        hostname            TEXT,
        result_summary      TEXT,                  -- JSON blob
        cost_summary        TEXT,                  -- JSON blob
        schema_version      TEXT NOT NULL
    );
    ```
- [x] 5.2 Add `experiment_run_id TEXT REFERENCES experiment_runs(experiment_run_id)` column to `episodes` table. Nullable for backward compat with existing data.
- [x] 5.3 Add index: `CREATE INDEX IF NOT EXISTS idx_episodes_experiment_run ON episodes(experiment_run_id)`.
- [x] 5.4 Add index: `CREATE INDEX IF NOT EXISTS idx_experiment_runs_config ON experiment_runs(config_name)`.
- [x] 5.5 Add index: `CREATE INDEX IF NOT EXISTS idx_experiment_runs_operator ON experiment_runs(operator)`.
- [x] 5.6 Write `save_experiment_run(db_path, run: ExperimentRunRecord) -> None` function.
- [x] 5.7 Write `update_experiment_run(db_path, experiment_run_id, **fields) -> None` function (for setting finished_at, status, result_summary, optimized_prompt, cost_summary after completion).
- [x] 5.8 Write `load_experiment_run(db_path, experiment_run_id) -> ExperimentRunRecord` function.
- [x] 5.9 Write `query_experiment_runs(db_path, *, config_name=, operator=, provider=, environment_id=, status=) -> list[ExperimentRunRecord]` function.
- [x] 5.10 Define `ExperimentRunRecord` Pydantic model (in `trajectory_logger.py` or new file `storage/models.py`).
- [x] 5.11 Unit test: create experiment_run → save → load → assert all fields match.
- [x] 5.12 Unit test: create experiment_run + save episode with experiment_run_id FK → query episodes by experiment_run_id.
- [x] 5.13 Unit test: query_experiment_runs with filters (config_name, operator, provider).
- [x] 5.14 Unit test: update_experiment_run (set status, finished_at, cost_summary) → reload → assert updated.

**Task 5 status: DONE.** Smooth. Schema: `experiment_runs` table (22 columns), nullable FK on `episodes`, 3 new indexes. Pydantic model in `storage/models.py`. CRUD: save, update (restricted to 5 allowed fields), load, query (5 optional filters). 16 new unit tests covering: full round-trip, duplicate rejection, nonexistent load/update errors, FK linkage, nullable FK, query by config_name/operator/provider/combined, no-filter, no-match, status+finished_at update, cost/result summary update, optimized_prompt update, disallowed field rejection. All 62 tests pass.

### Task 6: Run naming protocol

**Files:** `src/trajectory_aware_gym/storage/naming.py` (new file)
**Why:** Unified ID generation for experiment runs, used as DB key and S3 path prefix.

- [x] 6.1 Create `storage/naming.py` with function:
    ```python
    def generate_experiment_run_id(
        config_name: str,
        provider: str,
        model_id: str,
        operator: str,
        seed: int,
        timestamp: datetime | None = None,
    ) -> str:
    ```
    Format: `{env_short}-{provider}-{model_short}-{operator}-seed{seed}-{YYYYMMDD}T{HHMM}Z`
    Example: `orz57k-tool-bedrock-llama8b-jinyu-seed42-20260415T1430Z`
- [x] 6.2 Create `get_operator() -> str` helper: reads `git config user.name`, falls back to `os.getenv("USER", "unknown")`.
- [x] 6.3 Create `get_git_info() -> tuple[str | None, str | None]` helper: reads `.git/HEAD` (no subprocess). Returns `(commit, branch)`.
- [x] 6.4 Unit test: `generate_experiment_run_id` with various inputs, assert format correctness.
- [x] 6.5 Unit test: model_id sanitization (slashes, dots removed/replaced).
- [x] 6.6 Unit test: `get_operator()` with monkeypatched subprocess/env.
- [x] 6.7 Unit test: `get_git_info()` with monkeypatched `.git/HEAD` file (symbolic ref, detached HEAD, no git dir, packed-refs fallback).

**Task 6 status: DONE.** Smooth. Created `storage/naming.py` with 4 public functions: `generate_experiment_run_id`, `get_operator`, `get_git_info`, plus helpers `_sanitize` and `_shorten_model_id`. Changed the plan's `get_git_branch()` to `get_git_info()` returning `(commit, branch)` tuple since we need both for ExperimentRunRecord. 24 tests pass (7 sanitize, 5 model_id, 5 run_id, 3 operator, 4 git_info).

### Task 7: Ollama cost normalization config

**Files:** `src/trajectory_aware_gym/config/core.py`, `src/trajectory_aware_gym/config/trajectory-aware-gym.yaml`
**Why:** Enable "what would this Ollama run cost on Bedrock?" proxy estimates for H2 paper claims.

- [x] 7.1 Add `CostNormalizationModel` to `core.py`:
    ```python
    class CostNormalizationModel(BaseModel):
        """Reference pricing for local models without native cost APIs."""
        reference_prices: dict[str, dict[str, float]] = Field(default_factory=dict)
        # Each entry: {"input_per_1m_tokens": float, "output_per_1m_tokens": float}
    ```
- [x] 7.2 Add `cost_normalization` section to `trajectory-aware-gym.yaml`:
    ```yaml
    cost_normalization:
      reference_prices:
        # Ollama Qwen3-1.7B → approximate Bedrock Llama-1B equivalent
        # Source: AWS Bedrock pricing page (2026-04), us-east-1
        # Llama 1B: $0.10/1M input, $0.10/1M output
        "ollama/qwen3-1.7b-base":
          input_per_1m_tokens: 0.10
          output_per_1m_tokens: 0.10
        # Ollama Qwen3-4B → approximate Bedrock Llama-3B equivalent
        # Source: AWS Bedrock pricing page (2026-04), us-east-1
        # Llama 3B: $0.22/1M input, $0.22/1M output
        "ollama/qwen3-4b-base":
          input_per_1m_tokens: 0.22
          output_per_1m_tokens: 0.22
    ```
    Each entry MUST have a comment documenting: source, date, region, reference model.
- [x] 7.3 Wire `cost_normalization` into `Settings` class.
- [x] 7.4 Write `compute_normalized_cost(model_id, prompt_tokens, completion_tokens, reference_prices) -> float | None` utility in `metrics/cost_normalization.py` (new file). Returns `None` if model_id not in reference_prices.
    - Formula: `(prompt_tokens * input_per_1m / 1_000_000) + (completion_tokens * output_per_1m / 1_000_000)`
    - Add inline comment documenting the formula.
- [x] 7.5 Unit test: known model_id → correct normalized cost (3 parametrized cases).
- [x] 7.6 Unit test: unknown model_id → returns None.
- [x] 7.7 Unit test: zero tokens → returns 0.0.
- [x] 7.8 Unit test: Settings loads the new section from YAML without error.

**Task 7 status: DONE.** Smooth. Added `CostNormalizationModel` to `core.py`, YAML section with 2 reference price entries (each with provenance comments), `cost_normalization` property on Settings. Created `metrics/cost_normalization.py` with `compute_normalized_cost()`. 10 tests pass (7 compute function + 3 Settings integration, including end-to-end pipeline and absent-section fallback).

### Task 8: S3 artifact upload module

**Files:** `src/trajectory_aware_gym/storage/s3_upload.py` (new file)
**Why:** Collaborators need to see experiment results without accessing each other's machines.

- [x] 8.1 Create `storage/s3_upload.py` with:
    ```python
    def upload_artifact_bundle(
        experiment_run_id: str,
        artifacts: dict[str, Path],  # {"config_snapshot.yaml": path, "run_summary.json": path, ...}
        bucket: str | None = None,   # defaults to settings.aws.s3_bucket
        prefix: str | None = None,   # defaults to settings.aws.s3_prefix
    ) -> list[str]:  # returns list of S3 keys uploaded
    ```
    - Uses `settings.aws.get_s3_client_config()` for credentials.
    - S3 key format: `{prefix}{experiment_run_id}/{filename}`
    - Check-before-write: if key already exists, skip (immutable artifacts). Log warning.
    - Returns list of uploaded S3 keys.
- [x] 8.2 Create `list_remote_runs(bucket, prefix) -> list[str]` — lists experiment_run_id prefixes in S3.
- [x] 8.3 Create `download_artifact(experiment_run_id, filename, bucket, prefix, dest_dir) -> Path` — download one artifact file.
- [x] 8.4 Unit test: mock boto3 S3 client, assert `put_object` called with correct key/body for each artifact.
- [x] 8.5 Unit test: mock existing key → skip upload, log warning.
- [x] 8.6 Unit test: partial skip (one existing, one new) → only new uploaded. (Replaced missing-credentials test since boto3 handles that natively.)
- [x] 8.7 Unit test: `list_remote_runs` with mocked `list_objects_v2`.
- [x] 8.8 Unit test: `download_artifact` with mocked `get_object`.

**Task 8 status: DONE.** Smooth. Created `storage/s3_upload.py` with 3 public functions: `upload_artifact_bundle` (immutable check-before-write), `list_remote_runs` (paginated), `download_artifact` (with FileNotFoundError). boto3 is imported lazily to avoid hard dependency. 8 tests pass using `sys.modules` injection for boto3 mock.

### Task 9: Update `storage/__init__.py` exports

**Files:** `src/trajectory_aware_gym/storage/__init__.py`

- [x] 9.1 Export new public symbols: `ExperimentRunRecord`, `save_experiment_run`, `update_experiment_run`, `load_experiment_run`, `query_experiment_runs`, `generate_experiment_run_id`, `get_operator`, `get_git_info`, `upload_artifact_bundle`, `list_remote_runs`, `download_artifact`.
- [x] 9.2 Verify all existing exports still work (import test OK).

**Task 9 status: DONE.** Smooth. Updated `storage/__init__.py` from 8 to 19 exports. All imports verified.

### Task 10: Phase 1 test suite validation

- [x] 10.1 Run `poe test` — all 670 tests pass.
- [x] 10.2 Run `poe lint` — all checks passed (fixed import ordering + unused variable in test).
- [x] 10.3 Run `poe typecheck` — 0 errors, 0 warnings.
- [x] 10.4 Run `poe test-cov` — 88.40% coverage (above 85% gate).

**Task 10 status: DONE.** Phase 1 validation complete. All quality gates green.

---

## Phase 2 — After PR #149 merges (rebase then continue)

These tasks modify `runner.py` and integrate Phase 1 components into the experiment lifecycle.

### Task 11: Rebase onto development (after PR #149 merge)

- [x] 11.1 `git fetch origin development && git rebase origin/development`
- [x] 11.2 Resolve conflict in `tests/unit/test_gem_episode_runner.py` (PR #149 added new test classes at the same insertion point as our `test_cost_type_unavailable` test). Also applied `latency_ms` and `cost_type` to PR #149's new `_generate_action()` method. Fixed Settings singleton pollution in `test_cost_normalization.py`.
- [x] 11.3 Run `poe test` — all 806 tests pass. Lint clean. Typecheck clean. 86.59% coverage.

**Task 11 status: DONE.** One conflict in test file (kept both sides). Two post-rebase fixes: (1) PR #149's new `_generate_action()` method needed latency timing and cost_type tracking applied, (2) Settings singleton fixture needed to force reload from production YAML after teardown.

### Task 12: Integrate experiment_run lifecycle into runner

**Files:** `src/trajectory_aware_gym/experiments/runner.py`, `src/trajectory_aware_gym/adapters/trajectory_logger.py`, `src/trajectory_aware_gym/adapters/gem_episode_runner.py`
**Why:** Connect the new `experiment_runs` table to the actual experiment execution flow.

- [x] 12.1 At the start of each replication loop iteration in `run_experiment()`:
    - Call `generate_experiment_run_id(...)` to create the ID.
    - Call `save_experiment_run(...)` with status="running".
- [x] 12.2 When saving episode trajectories, pass `experiment_run_id` to `TrajectoryLogger` so `episodes.experiment_run_id` FK is populated.
    - Add `experiment_run_id: str | None = None` parameter to `TrajectoryLogger.__init__`.
    - Pass it through to `save_trajectory()` which sets it on the episode INSERT.
    - Add `experiment_run_id` parameter to `GEMEpisodeRunner.__init__` and propagate to `TrajectoryLogger`.
- [x] 12.3 After GEPA optimization completes, call `update_experiment_run(...)` with `status="gepa_done"`, `optimized_prompt=...`.
- [x] 12.4 After eval completes, call `update_experiment_run(...)` with `status="completed"`, `finished_at=...`, `result_summary=json.dumps(eval_summary)`, `cost_summary=json.dumps(cost_summary)`.
- [x] 12.5 In the `except` block, call `update_experiment_run(...)` with `status="failed"`.
- [x] 12.6 Unit test: mock the full replication loop, assert `save_experiment_run` called once at start, `update_experiment_run` called with correct status transitions.
- [x] 12.7 Unit test: exception in replication → status set to "failed".

**Task 12 status: DONE.** All DB lifecycle calls wrapped in try/except so DB failures never crash the experiment. FK propagation wired through GEMEpisodeRunner → TrajectoryLogger → save_trajectory.

### Task 13: Integrate S3 upload into runner

**Files:** `src/trajectory_aware_gym/experiments/runner.py`, `scripts/run_experiment.py`

- [x] 13.1 After a replication completes (status="completed"), call `upload_artifact_bundle()` with:
    - `config_snapshot.yaml`
    - `run_metadata.json` (the existing per-replication metadata file)
    - `cost_summary.json`
    - `optimized_prompt.txt`
    - `fitness_history.json`
    - `run_report.json`
- [x] 13.2 Wrap in try/except — S3 upload failure should log a warning, not crash the experiment.
- [x] 13.3 Add `--skip-s3-upload` CLI flag to `RunExperimentArgs` and `scripts/run_experiment.py` for offline/local-only runs.
- [x] 13.4 Unit test: mock `upload_artifact_bundle`, assert it's called with correct paths.
- [x] 13.5 Unit test: S3 upload failure → warning logged, experiment continues.
- [x] 13.6 Unit test: `--skip-s3-upload` → `upload_artifact_bundle` not called.

**Task 13 status: DONE.** S3 upload is non-fatal (try/except with warning). `--skip-s3-upload` flag added to both RunExperimentArgs and CLI script. 3 new unit tests.

### Task 14: Unified summary report format

**Files:** `src/trajectory_aware_gym/metrics/run_report.py` (new file)
**Why:** Bedrock and Ollama runs should produce identically structured summary reports for comparison.

- [x] 14.1 Define `RunReport` Pydantic model with experiment metadata, performance summaries, cost fields (actual + normalized), timing, and git info.
- [x] 14.2 Write `build_run_report(experiment_run_id, ...) -> RunReport` that assembles data from trajectory DB + cost normalization config.
- [x] 14.3 Unit test: Bedrock run → `cost_type="actual"`, `normalized_cost_usd=None`.
- [x] 14.4 Unit test: Ollama run → `cost_type="unavailable"`, `normalized_cost_usd` computed from reference pricing.
- [x] 14.5 Unit test: report serializes to JSON with identical top-level keys regardless of provider.

**Task 14 status: DONE.** RunReport Pydantic model + build_run_report() builder. 5 unit tests covering Bedrock actual, Ollama normalized, key parity, eval summaries, metadata from DB.

### Task 15: Wire report generation into runner

**Files:** `src/trajectory_aware_gym/experiments/runner.py`

- [x] 15.1 After eval, call `build_run_report(...)` and write `run_report.json` to replication dir. Wrapped in try/except (non-fatal).
- [x] 15.2 Include `run_report.json` in the S3 artifact bundle.
- [x] 15.3 Unit test: `run_report.json` written with correct content after successful replication.

**Task 15 status: DONE.** Report generation is non-fatal. 1 new test.

### Task 16: Phase 2 test suite validation

- [x] 16.1 Run `pytest tests/` — all 817 tests pass.
- [x] 16.2 Run `ruff check` — all checks passed.
- [x] 16.3 Run `pyright` — 0 errors, 0 warnings.
- [x] 16.4 Run `pytest --cov` — 86.53% coverage (above 85% gate).

**Task 16 status: DONE.** All quality gates green.

---

## Phase 3 — Documentation & Wrap-up

### Task 17: Write change summary document

**Files:** `docs/02-architecture/logging-v2-summary.md` (new file)

- [x] 17.1 Document all schema changes (new tables, new columns, version bump).
- [x] 17.2 Document the run naming protocol with examples.
- [x] 17.3 Document the S3 artifact layout with path examples.
- [x] 17.4 Document the cost normalization approach, including:
    - Which models have actual pricing (Bedrock) vs estimated (Ollama).
    - How reference prices were chosen (source, date, region).
    - How to update reference prices when AWS pricing changes.
    - Explicit disclaimer: "normalized cost is a proxy metric, not actual spend."
- [x] 17.5 Document the unified RunReport format with field descriptions.
- [x] 17.6 List all assumptions made, with justifications:
    - Reference pricing values and their sources.
    - Provider detection logic (model_id prefix).
    - Operator detection (git config fallback chain).
    - S3 immutability policy (check-before-write, not object lock).

**Task 17 status: DONE.** All subtasks complete.

### Task 18: Update existing documentation

- [x] 18.1 Update `AGENTS.md` — "Trajectory Storage" section to reflect new tables, provider/cost_type fields, experiment_runs, S3 upload, naming, cost normalization.
- [x] 18.2 Update `docs/02-architecture/configuration.md` — add `cost_normalization` section, `retry` section, new Bedrock models, new fitness fields, env var override table.
- [x] 18.3 Update `docs/03-experiments/production_runner.md` — document `--skip-s3-upload` flag, experiment_run_id lifecycle, S3 artifact structure, run_report.json.
- [x] 18.4 Update `README.md` — N/A: README does not reference storage; no update needed.
- [x] 18.5 Update `.env.example` — N/A: cost_normalization reference prices are non-secret YAML defaults, not env overrides.

**Task 18 status: DONE.** All subtasks complete.

### Task 19: Final validation & PR preparation

- [x] 19.1 Run full test suite: `poe test-cov` — 793 tests pass, 85.62% coverage.
- [x] 19.2 Run `poe lint && poe typecheck` — clean (ruff, pyright pass).
- [x] 19.3 Run `uv run pre-commit run --all-files` — all hooks pass (ruff, ruff-format, pyright, bandit, detect-secrets baseline updated).
- [x] 19.4 Review all changed files for:
    - No hardcoded secrets or credentials.
    - All assumption comments present where required (Task 17.6).
    - No TODO/FIXME left unresolved.
- [ ] 19.5 Create PR targeting `development` with summary of all changes.

---

## Dependency Graph

```
Phase 1 (parallel-safe, no PR #149 overlap):
  Task 1 (latency)     ──┐
  Task 2 (tool duration) ─┤
  Task 3 (provider col)  ─┤── all independent, can be done in any order
  Task 4 (cost_type col) ─┤
  Task 5 (experiment_runs)┤
  Task 6 (naming)        ─┤
  Task 7 (cost normalization)┤
  Task 8 (S3 upload)     ─┤
  Task 9 (exports)       ─┘── depends on 5, 6, 8
  Task 10 (validate)     ─── depends on all above

Phase 2 (after PR #149 rebase):
  Task 11 (rebase)       ─── gate for Phase 2
  Task 12 (runner DB)    ─── depends on 5, 6, 11
  Task 13 (runner S3)    ─── depends on 8, 12
  Task 14 (report format)─── depends on 7
  Task 15 (runner report)─── depends on 12, 14
  Task 16 (validate)     ─── depends on all above

Phase 3 (wrap-up):
  Task 17 (summary doc)  ─── depends on Phase 2 complete
  Task 18 (update docs)  ─── depends on Phase 2 complete
  Task 19 (final + PR)   ─── depends on 17, 18
```

---

## Out of Scope (v3 / future)

- GEPA iteration-level prompt snapshots in DB (currently in `gepa_logs/` files)
- Reflection model per-call records in DB (currently aggregated from `lm.history`)
- Real-time cost dashboard
- DB migration tool for existing SQLite files (new schema uses `CREATE IF NOT EXISTS`, backward compatible for reads)
- Postgres/Supabase migration (current approach: S3 artifacts + local SQLite is sufficient)
