"""Unit tests for production experiment runner."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from trajectory_aware_gym.adapters.gem_episode_runner import GEMEpisodeResult
from trajectory_aware_gym.config.core import Settings
from trajectory_aware_gym.experiments.runner import (
    DEFAULT_SEED_PROMPT,
    RunExperimentArgs,
    _budget_alert_fraction,
    _build_examples,
    _config_hash,
    _extract_fitness_history,
    _extract_pareto_frontier,
    _extract_reflection_usage,
    _extract_task_usage,
    _find_resumable_run,
    _git_commit_hash,
    _is_replication_completed,
    _model_replication_dir,
    _raw_metrics_summary,
    _safe_segment,
    _write_csv,
    _write_json,
    _write_jsonl,
    derive_max_metric_calls,
    run_experiment,
    select_replication_seeds,
    select_task_models,
)
from trajectory_aware_gym.metrics import EpisodeRawMetrics
from trajectory_aware_gym.models.experiment import ExperimentConfig, FitnessOverride

QUICK_TEST_CONFIG = Path("experiments/quick-test/config.yaml")


def _find_replication_dir(tmp_path: Path, config_name: str, model_name: str, seed: int) -> Path:
    """Locate the timestamped replication dir for a config/model/seed combo."""
    config_dir = tmp_path / config_name
    ts_dirs = sorted(d for d in config_dir.iterdir() if d.is_dir())
    assert len(ts_dirs) >= 1, f"No timestamp dirs found under {config_dir}"
    return ts_dirs[-1] / model_name / f"replication_{seed}"


def _make_metric(
    run_id: str,
    *,
    success: bool,
    cost: float,
    tokens: int,
    coverage: float = 1.0,
) -> EpisodeRawMetrics:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    finished = datetime(2026, 1, 1, tzinfo=UTC)
    return EpisodeRawMetrics(
        run_id=run_id,
        environment_id="math:Orz57K",
        seed=42,
        started_at=started,
        finished_at=finished,
        episode_latency_seconds=1.0,
        step_count=1,
        terminated=success,
        truncated=False,
        success=success,
        total_reward=1.0 if success else 0.0,
        reward_per_step=1.0 if success else 0.0,
        steps_per_second=1.0,
        reward_per_second=1.0 if success else 0.0,
        repeat_action_rate=0.0,
        llm_cost_usd=cost,
        prompt_tokens=max(0, tokens - 1),
        completion_tokens=1,
        total_tokens=tokens,
        mean_llm_latency_seconds=0.2,
        p95_llm_latency_seconds=0.2,
        cost_per_step_usd=cost,
        cost_per_success_usd=cost if success else None,
        tokens_per_step=float(tokens),
        cost_data_coverage=coverage,
        token_data_coverage=coverage,
        llm_latency_data_coverage=coverage,
    )


def _make_episode_result(
    run_id: str, *, success: bool, cost: float, tokens: int
) -> GEMEpisodeResult:
    metric = _make_metric(run_id, success=success, cost=cost, tokens=tokens)
    trajectory = SimpleNamespace(total_tokens=tokens, total_cost_usd=cost)
    return GEMEpisodeResult(trajectory=cast(Any, trajectory), log_path=None, raw_metrics=metric)


def test_derive_max_metric_calls_from_budget_and_override() -> None:
    config = ExperimentConfig.from_yaml(QUICK_TEST_CONFIG)
    expected = (
        config.gepa_budget.iterations
        * config.gepa_budget.population_size
        * config.gepa_budget.tasks_per_minibatch
    )
    assert derive_max_metric_calls(config, None) == expected
    assert derive_max_metric_calls(config, 77) == 77

    with pytest.raises(ValueError, match="max_metric_calls"):
        derive_max_metric_calls(config, 0)


def test_model_and_seed_selectors_validate_subset() -> None:
    config = ExperimentConfig.from_yaml(QUICK_TEST_CONFIG)

    models = select_task_models(config, ("Qwen3-1.7B-Base",))
    assert len(models) == 1
    assert models[0].name == "Qwen3-1.7B-Base"

    seeds = select_replication_seeds(config, (42,))
    assert seeds == (42,)

    with pytest.raises(ValueError, match="task_models"):
        select_task_models(config, ("does-not-exist",))

    with pytest.raises(ValueError, match="replication_seeds"):
        select_replication_seeds(config, (999,))


def test_settings_override_fitness_applies_and_restores() -> None:
    from trajectory_aware_gym.config import settings

    original_gamma = settings.fitness.gamma
    override = FitnessOverride(gamma=0.123)
    patch = override.model_dump(mode="json", by_alias=True, exclude_none=True)

    with Settings.override_fitness(patch):
        assert settings.fitness.gamma == pytest.approx(0.123)

    assert settings.fitness.gamma == pytest.approx(original_gamma)


def test_is_replication_completed_checks_run_metadata_status(tmp_path: Path) -> None:
    replication_dir = tmp_path / "replication_42"
    replication_dir.mkdir(parents=True)

    assert _is_replication_completed(replication_dir) is False

    (replication_dir / "run_metadata.json").write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )
    assert _is_replication_completed(replication_dir) is False

    (replication_dir / "run_metadata.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    assert _is_replication_completed(replication_dir) is True


def test_run_experiment_writes_full_replication_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module

    train_results = [
        _make_episode_result("train-1", success=True, cost=0.10, tokens=10),
        _make_episode_result("train-2", success=False, cost=0.20, tokens=20),
    ]
    eval_results = [_make_episode_result("eval-1", success=True, cost=0.05, tokens=5)]

    class FakeRunner:
        def __init__(self, **kwargs):
            self.episode_history = tuple(train_results)

    class FakeSolverModule:
        def __init__(self, runner, default_instructions: str):
            self.instructions = default_instructions

    class FakeGEPA:
        def __init__(self, **kwargs):
            return None

        def compile(self, student, trainset, valset):
            detailed = SimpleNamespace(
                best_idx=1,
                val_aggregate_scores=[0.2, 0.9],
                val_subscores=[{"a": 0.0}, {"a": 1.0}],
            )
            return SimpleNamespace(instructions="optimized prompt", detailed_results=detailed)

    monkeypatch.setattr(runner_module, "_build_trainset", lambda config: [SimpleNamespace()])
    monkeypatch.setattr(runner_module, "GEMEpisodeRunner", FakeRunner)
    monkeypatch.setattr(runner_module, "GEMSolverModule", FakeSolverModule)
    monkeypatch.setattr(runner_module.dspy, "GEPA", FakeGEPA)
    monkeypatch.setattr(runner_module.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(runner_module, "_build_task_lm", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        runner_module, "get_reflection_lm", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        runner_module,
        "_extract_reflection_usage",
        lambda reflection_lm: (7, 0.07),
    )
    monkeypatch.setattr(
        runner_module,
        "_run_heldout_eval",
        lambda **kwargs: (
            eval_results,
            _default_eval_summary(episodes=1, successes=1, correct=1),
        ),
    )

    summary = run_experiment(
        RunExperimentArgs(
            config_path=QUICK_TEST_CONFIG,
            max_metric_calls=8,
            results_root=tmp_path,
        )
    )

    replication_dir = _find_replication_dir(tmp_path, "quick-test", "Qwen3-1.7B-Base", 42)
    assert (replication_dir / "optimized_prompt.txt").exists()
    assert (replication_dir / "fitness_history.json").exists()
    assert (replication_dir / "pareto_frontier.json").exists()
    assert (replication_dir / "cost_summary.json").exists()
    assert (replication_dir / "run_metadata.json").exists()
    assert (replication_dir / "config_snapshot.yaml").exists()
    assert (replication_dir / "raw_metrics.csv").exists()
    assert (replication_dir / "raw_metrics.jsonl").exists()
    assert (replication_dir / "raw_metrics_summary.json").exists()
    assert (replication_dir / "gepa_logs").is_dir()

    cost_summary = json.loads((replication_dir / "cost_summary.json").read_text(encoding="utf-8"))
    # train(10+20) + baseline_eval(5) + eval(5) = 40
    assert cost_summary["task_model_tokens"] == 40
    assert cost_summary["reflection_tokens"] == 7
    assert cost_summary["total_tokens"] == 47
    # train(0.10+0.20) + baseline_eval(0.05) + eval(0.05) = 0.40
    assert cost_summary["task_model_cost"] == pytest.approx(0.40)
    assert cost_summary["reflection_cost"] == pytest.approx(0.07)
    assert cost_summary["total_cost"] == pytest.approx(0.47)
    assert cost_summary["baseline_eval_task_model_tokens"] == 5
    assert cost_summary["baseline_eval_task_model_cost"] == pytest.approx(0.05)

    metadata = json.loads((replication_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["result"]["final_fitness"] == pytest.approx(0.9)
    assert metadata["baseline_eval"]["accuracy"] == 1.0
    assert metadata["baseline_eval"]["episodes"] == 1

    raw_summary = json.loads(
        (replication_dir / "raw_metrics_summary.json").read_text(encoding="utf-8")
    )
    # 2 train + 1 baseline eval + 1 optimized eval = 4
    assert raw_summary["episodes"] == 4
    assert raw_summary["successes"] == 3
    assert raw_summary["mean_cost_data_coverage"] > 0
    assert raw_summary["mean_token_data_coverage"] > 0

    assert summary["models"]["Qwen3-1.7B-Base"]["42"]["status"] == "completed"


def test_run_experiment_skips_completed_replication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module

    # Pre-create an incomplete run (run_summary.json without finished_at) so
    # the runner resumes into this timestamp directory.
    ts = "20260101T000000Z"
    ts_dir = tmp_path / "quick-test" / ts
    ts_dir.mkdir(parents=True)
    (ts_dir / "run_summary.json").write_text(
        json.dumps({"finished_at": None}),
        encoding="utf-8",
    )
    replication_dir = ts_dir / "Qwen3-1.7B-Base" / "replication_42"
    replication_dir.mkdir(parents=True)
    (replication_dir / "run_metadata.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(runner_module, "_build_trainset", lambda config: [SimpleNamespace()])

    class FailGEPA:
        def __init__(self, **kwargs):
            raise AssertionError("GEPA should not initialize for completed replication")

    monkeypatch.setattr(runner_module.dspy, "GEPA", FailGEPA)

    summary = run_experiment(
        RunExperimentArgs(
            config_path=QUICK_TEST_CONFIG,
            results_root=tmp_path,
        )
    )

    assert summary["models"]["Qwen3-1.7B-Base"]["42"]["status"] == "skipped"


# ── Helper-function unit tests ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hello-world", "hello-world"),
        ("My_Model.v1", "My_Model.v1"),
        ("model/with/slashes", "model_with_slashes"),
        ("spaces here", "spaces_here"),
        ("a!b@c#d", "a_b_c_d"),
        ("", ""),
        ("---", "---"),
    ],
)
def test_safe_segment_sanitizes_special_chars(raw: str, expected: str) -> None:
    assert _safe_segment(raw) == expected


def test_git_commit_hash_returns_hash_for_detached_head(tmp_path: Path) -> None:
    """Detached HEAD: .git/HEAD contains a commit SHA directly."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("abc123def456\n", encoding="utf-8")
    with patch("trajectory_aware_gym.experiments.runner._PROJECT_ROOT", tmp_path):
        result = _git_commit_hash()
    assert result == "abc123def456"  # pragma: allowlist secret


def test_git_commit_hash_reads_head_and_ref(tmp_path: Path) -> None:
    """Named branch: .git/HEAD contains a ref pointer, resolved to a commit SHA."""
    git_dir = tmp_path / ".git"
    ref_dir = git_dir / "refs" / "heads"
    ref_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (ref_dir / "main").write_text("deadbeef1234\n", encoding="utf-8")
    with patch("trajectory_aware_gym.experiments.runner._PROJECT_ROOT", tmp_path):
        result = _git_commit_hash()
    assert result == "deadbeef1234"  # pragma: allowlist secret


def test_git_commit_hash_returns_none_when_no_git_dir(tmp_path: Path) -> None:
    """No .git directory present → returns None."""
    with patch("trajectory_aware_gym.experiments.runner._PROJECT_ROOT", tmp_path):
        result = _git_commit_hash()
    assert result is None


def test_git_commit_hash_returns_none_on_oserror(tmp_path: Path) -> None:
    """OSError reading .git files → returns None."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("abc123\n", encoding="utf-8")
    with (
        patch("trajectory_aware_gym.experiments.runner._PROJECT_ROOT", tmp_path),
        patch.object(Path, "read_text", side_effect=OSError("Permission denied")),
    ):
        result = _git_commit_hash()
    assert result is None


def test_git_commit_hash_returns_none_for_empty_head(tmp_path: Path) -> None:
    """Empty .git/HEAD (e.g. freshly initialised repo) → returns None."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("   \n", encoding="utf-8")
    with patch("trajectory_aware_gym.experiments.runner._PROJECT_ROOT", tmp_path):
        result = _git_commit_hash()
    assert result is None


def test_git_commit_hash_rejects_path_traversal(tmp_path: Path) -> None:
    """A malicious ref: with '..' must not read outside .git/."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: ../../etc/passwd\n", encoding="utf-8")
    with patch("trajectory_aware_gym.experiments.runner._PROJECT_ROOT", tmp_path):
        result = _git_commit_hash()
    assert result is None


def test_write_json_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "out.json"
    _write_json(out, {"key": "value", "n": 42})
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == {"key": "value", "n": 42}


def test_write_jsonl_writes_one_line_per_row(tmp_path: Path) -> None:
    rows = [
        _make_metric("r1", success=True, cost=0.1, tokens=5),
        _make_metric("r2", success=False, cost=0.2, tokens=10),
    ]
    out = tmp_path / "rows.jsonl"
    _write_jsonl(out, rows)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["run_id"] == "r1"


def test_write_csv_empty_writes_header_only(tmp_path: Path) -> None:
    out = tmp_path / "empty.csv"
    _write_csv(out, [])
    with out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows_read = list(reader)
    assert rows_read == []
    # Header row must exist (DictReader.fieldnames is populated)
    with out.open(encoding="utf-8") as fh:
        header_line = fh.readline().strip()
    assert len(header_line) > 0


def test_write_csv_with_rows(tmp_path: Path) -> None:
    rows = [_make_metric("x", success=True, cost=0.05, tokens=3)]
    out = tmp_path / "metrics.csv"
    _write_csv(out, rows)
    with out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        data = list(reader)
    assert len(data) == 1
    assert data[0]["run_id"] == "x"


# ── _raw_metrics_summary ───────────────────────────────────────────────────


def test_raw_metrics_summary_empty() -> None:
    summary = _raw_metrics_summary([])
    assert summary["episodes"] == 0
    assert summary["total_cost"] is None
    assert summary["total_tokens"] is None


def test_raw_metrics_summary_no_cost_data() -> None:
    rows = [
        _make_metric("r1", success=True, cost=0.0, tokens=5, coverage=1.0),
    ]
    # Manually set llm_cost_usd to None to exercise the None-filtering branch
    rows[0] = rows[0].model_copy(update={"llm_cost_usd": None})
    summary = _raw_metrics_summary(rows)
    assert summary["episodes"] == 1
    assert summary["total_cost"] is None


def test_raw_metrics_summary_no_token_data() -> None:
    rows = [_make_metric("r1", success=False, cost=0.01, tokens=5)]
    rows[0] = rows[0].model_copy(update={"total_tokens": None})
    summary = _raw_metrics_summary(rows)
    assert summary["total_tokens"] is None
    assert summary["total_cost"] == pytest.approx(0.01)


def test_raw_metrics_summary_aggregates_correctly() -> None:
    rows = [
        _make_metric("a", success=True, cost=0.10, tokens=10),
        _make_metric("b", success=False, cost=0.20, tokens=20),
    ]
    summary = _raw_metrics_summary(rows)
    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["total_cost"] == pytest.approx(0.30)
    assert summary["total_tokens"] == 30
    assert summary["mean_latency_seconds"] == pytest.approx(1.0)


# ── _extract_reflection_usage ──────────────────────────────────────────────


def test_extract_reflection_usage_none_lm() -> None:
    tokens, cost = _extract_reflection_usage(None)
    assert tokens == 0
    assert cost == 0.0


def test_extract_reflection_usage_no_history_attr() -> None:
    lm = SimpleNamespace()  # no `history` attribute
    tokens, cost = _extract_reflection_usage(lm)
    assert tokens == 0
    assert cost == 0.0


def test_extract_reflection_usage_empty_history() -> None:
    lm = SimpleNamespace(history=[])
    tokens, cost = _extract_reflection_usage(lm)
    assert tokens == 0
    assert cost == 0.0


def test_extract_reflection_usage_non_dict_entry_ignored() -> None:
    lm = SimpleNamespace(history=["not a dict", 42, None])
    tokens, cost = _extract_reflection_usage(lm)
    assert tokens == 0
    assert cost == 0.0


def test_extract_reflection_usage_with_tokens_only() -> None:
    history = [{"usage": {"total_tokens": 150}, "response": None}]
    lm = SimpleNamespace(history=history)
    tokens, cost = _extract_reflection_usage(lm)
    assert tokens == 150
    assert cost == 0.0


def test_extract_reflection_usage_missing_usage_key() -> None:
    history = [{"response": None}]
    lm = SimpleNamespace(history=history)
    tokens, cost = _extract_reflection_usage(lm)
    assert tokens == 0


def test_extract_reflection_usage_completion_cost_exception() -> None:
    """completion_cost raising an exception should be swallowed."""
    history = [{"usage": {"total_tokens": 10}, "response": object()}]
    lm = SimpleNamespace(history=history)
    with patch(
        "trajectory_aware_gym.experiments.runner.completion_cost",
        side_effect=ValueError("fail"),
    ):
        tokens, cost = _extract_reflection_usage(lm)
    assert tokens == 10
    assert cost == 0.0


def test_extract_reflection_usage_accumulates_multiple_entries() -> None:
    history = [
        {"usage": {"total_tokens": 50}, "response": None},
        {"usage": {"total_tokens": 100}, "response": None},
    ]
    lm = SimpleNamespace(history=history)
    tokens, cost = _extract_reflection_usage(lm)
    assert tokens == 150


def test_extract_reflection_usage_adds_numeric_completion_cost() -> None:
    history = [{"usage": {"total_tokens": 11}, "response": object()}]
    lm = SimpleNamespace(history=history)
    with patch(
        "trajectory_aware_gym.experiments.runner.completion_cost",
        return_value=0.123,
    ):
        tokens, cost = _extract_reflection_usage(lm)
    assert tokens == 11
    assert cost == pytest.approx(0.123)


# ── dataset/eval helpers ───────────────────────────────────────────────────


def _fake_gem_env(seen_seeds: list[int], env_closed: dict[str, bool], prefix: str = "obs"):
    """Create a fake GEM environment and gem module for testing _build_examples."""

    class FakeEnv:
        def reset(self, *, seed: int):
            seen_seeds.append(seed)
            return (f"{prefix}-{seed}", {})

        def close(self) -> None:
            env_closed["value"] = True

    fake_env = FakeEnv()

    class FakeGem:
        @staticmethod
        def make(_env_id: str):
            return fake_env

    def fake_import_module(name: str):
        if name == "gem":
            return FakeGem
        if name == "gem.envs":
            return SimpleNamespace()
        raise ModuleNotFoundError(name)

    return fake_import_module


def test_build_examples_uses_expected_seeds_and_closes_env() -> None:
    import importlib

    config = ExperimentConfig.from_yaml(QUICK_TEST_CONFIG)
    seen_seeds: list[int] = []
    env_closed = {"value": False}
    fake_import = _fake_gem_env(seen_seeds, env_closed, "train")

    with patch.object(importlib, "import_module", side_effect=fake_import):
        trainset = _build_examples(config, config.seeds.data_seed, config.environment.train_size)

    assert len(trainset) == config.environment.train_size
    assert seen_seeds[0] == config.seeds.data_seed
    assert seen_seeds[-1] == config.seeds.data_seed + config.environment.train_size - 1
    assert env_closed["value"] is True


def test_build_examples_eval_uses_offset_seed_and_closes_env() -> None:
    import importlib

    config = ExperimentConfig.from_yaml(QUICK_TEST_CONFIG)
    seen_seeds: list[int] = []
    env_closed = {"value": False}
    fake_import = _fake_gem_env(seen_seeds, env_closed, "eval")

    start_seed = config.seeds.data_seed + config.environment.train_size
    with patch.object(importlib, "import_module", side_effect=fake_import):
        examples = _build_examples(config, start_seed, config.environment.effective_eval_size)

    assert len(examples) == config.environment.effective_eval_size
    assert seen_seeds[0] == start_seed
    assert seen_seeds[-1] == start_seed + config.environment.effective_eval_size - 1
    assert env_closed["value"] is True


def test_run_heldout_eval_rollouts_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module

    config = ExperimentConfig.from_yaml(QUICK_TEST_CONFIG)
    config = config.model_copy(
        update={"eval_protocol": config.eval_protocol.model_copy(update={"rollouts_per_task": 2})}
    )

    eval_examples = [
        SimpleNamespace(seed=100, problem="p1"),
        SimpleNamespace(seed=200, problem="p2"),
    ]
    run_calls: list[tuple[int, int, str]] = []

    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_episode(
            self,
            instructions: str,
            *,
            episode_index: int,
            seed_override: int,
            expected_observation: str,
            persist: bool,
        ) -> GEMEpisodeResult:
            run_calls.append((episode_index, seed_override, expected_observation))
            return _make_episode_result(
                f"eval-{episode_index}",
                success=(episode_index % 2 == 0),
                cost=0.0,
                tokens=1,
            )

    monkeypatch.setattr(runner_module, "GEMEpisodeRunner", FakeRunner)
    monkeypatch.setattr(runner_module, "_eval_examples", lambda cfg: eval_examples)

    results, summary = runner_module._run_heldout_eval(
        config=config,
        task_model_id="ollama/qwen3-1.7b-base",
        instructions="optimized prompt",
    )

    assert len(results) == 4
    assert summary["episodes_attempted"] == 4
    assert summary["episodes"] == 4
    assert summary["failed"] == 0
    assert summary["successes"] == 2
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["correct"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    assert run_calls == [
        (0, 100, "p1"),
        (1, 101, None),
        (2, 200, "p2"),
        (3, 201, None),
    ]


# ── _extract_task_usage ────────────────────────────────────────────────────


def test_extract_task_usage_empty() -> None:
    tokens, cost = _extract_task_usage([])
    assert tokens == 0
    assert cost == 0.0


def test_extract_task_usage_accumulates() -> None:
    results = [
        _make_episode_result("a", success=True, cost=0.10, tokens=10),
        _make_episode_result("b", success=False, cost=0.20, tokens=20),
    ]
    tokens, cost = _extract_task_usage(results)
    assert tokens == 30
    assert cost == pytest.approx(0.30)


# ── _extract_fitness_history ───────────────────────────────────────────────


def test_extract_fitness_history_no_detailed_results() -> None:
    module = SimpleNamespace()  # no detailed_results attribute
    assert _extract_fitness_history(module) == []


def test_extract_fitness_history_with_all_fields() -> None:
    detailed = SimpleNamespace(
        val_aggregate_scores=[0.3, 0.7],
        val_subscores=[{0: 0.0, 1: 0.95}, {0: 0.95, 1: 0.95}],
        discovery_eval_counts=[0, 53],
    )
    module = SimpleNamespace(detailed_results=detailed)
    result = _extract_fitness_history(module)
    assert result == [
        {"index": 0, "val_aggregate_score": 0.3, "accuracy": 0.5, "metric_calls": 0},
        {"index": 1, "val_aggregate_score": 0.7, "accuracy": 1.0, "metric_calls": 53},
    ]


def test_extract_fitness_history_aggregate_only() -> None:
    detailed = SimpleNamespace(val_aggregate_scores=[0.2, 0.8, 0.9])
    module = SimpleNamespace(detailed_results=detailed)
    result = _extract_fitness_history(module)
    assert result == [
        {"index": 0, "val_aggregate_score": 0.2},
        {"index": 1, "val_aggregate_score": 0.8},
        {"index": 2, "val_aggregate_score": 0.9},
    ]


def test_extract_fitness_history_aggregate_skips_non_numeric() -> None:
    detailed = SimpleNamespace(val_aggregate_scores=[0.5, "bad", None, 0.9])
    module = SimpleNamespace(detailed_results=detailed)
    result = _extract_fitness_history(module)
    assert len(result) == 2
    assert result[0]["val_aggregate_score"] == 0.5
    assert result[1]["val_aggregate_score"] == 0.9


def test_extract_fitness_history_fallthrough_returns_empty() -> None:
    detailed = SimpleNamespace()
    module = SimpleNamespace(detailed_results=detailed)
    assert _extract_fitness_history(module) == []


# ── _extract_pareto_frontier ───────────────────────────────────────────────


def test_extract_pareto_frontier_no_detailed_results() -> None:
    module = SimpleNamespace()
    assert _extract_pareto_frontier(module) == []


def test_extract_pareto_frontier_no_frontier_attr() -> None:
    module = SimpleNamespace(detailed_results=SimpleNamespace())
    assert _extract_pareto_frontier(module) == []


def test_extract_pareto_frontier_with_list() -> None:
    frontier = [{"fitness": 0.9, "tokens": 100}, {"fitness": 0.7, "tokens": 80}]
    module = SimpleNamespace(detailed_results=SimpleNamespace(pareto_frontier=frontier))
    result = _extract_pareto_frontier(module)
    assert result == frontier


def test_extract_pareto_frontier_filters_non_dicts() -> None:
    frontier = [{"fitness": 0.5}, "not-a-dict", None]
    module = SimpleNamespace(detailed_results=SimpleNamespace(pareto_frontier=frontier))
    result = _extract_pareto_frontier(module)
    assert result == [{"fitness": 0.5}]


# ── _completed edge cases ──────────────────────────────────────────────────


def test_is_replication_completed_invalid_json_returns_false(tmp_path: Path) -> None:
    d = tmp_path / "rep"
    d.mkdir()
    (d / "run_metadata.json").write_text("{{ not valid json }", encoding="utf-8")
    assert _is_replication_completed(d) is False


# ── _find_resumable_run ────────────────────────────────────────────────────


def test_find_resumable_run_returns_none_when_no_config_dir(tmp_path: Path) -> None:
    assert _find_resumable_run(tmp_path, "nonexistent") is None


def test_find_resumable_run_returns_none_when_all_complete(tmp_path: Path) -> None:
    ts_dir = tmp_path / "my-config" / "20260101T000000Z"
    ts_dir.mkdir(parents=True)
    (ts_dir / "run_summary.json").write_text(
        json.dumps({"finished_at": "2026-01-01T00:01:00Z"}),
        encoding="utf-8",
    )
    assert _find_resumable_run(tmp_path, "my-config") is None


def test_find_resumable_run_returns_incomplete(tmp_path: Path) -> None:
    ts_dir = tmp_path / "my-config" / "20260101T000000Z"
    ts_dir.mkdir(parents=True)
    (ts_dir / "run_summary.json").write_text(
        json.dumps({"finished_at": None}),
        encoding="utf-8",
    )
    assert _find_resumable_run(tmp_path, "my-config") == "20260101T000000Z"


def test_find_resumable_run_picks_most_recent(tmp_path: Path) -> None:
    for ts in ("20260101T000000Z", "20260102T000000Z"):
        d = tmp_path / "my-config" / ts
        d.mkdir(parents=True)
        (d / "run_summary.json").write_text(
            json.dumps({"finished_at": None}),
            encoding="utf-8",
        )
    assert _find_resumable_run(tmp_path, "my-config") == "20260102T000000Z"


def test_find_resumable_run_ignores_corrupt_json(tmp_path: Path) -> None:
    ts_dir = tmp_path / "my-config" / "20260101T000000Z"
    ts_dir.mkdir(parents=True)
    (ts_dir / "run_summary.json").write_text("{{ bad json", encoding="utf-8")
    assert _find_resumable_run(tmp_path, "my-config") is None


# ── _build_task_lm ─────────────────────────────────────────────────────────


def test_build_task_lm_ollama_passes_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module
    from trajectory_aware_gym.models.experiment import ExperimentConfig, TaskModelConfig

    config = ExperimentConfig.from_yaml(QUICK_TEST_CONFIG)
    model = TaskModelConfig(
        name="Qwen3-1.7B",
        model_id="ollama/qwen3-1.7b-base",
        provider="ollama",
        parameter_count="1.7B",
    )

    captured: dict = {}

    def fake_lm(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(runner_module.dspy, "LM", fake_lm)
    runner_module._build_task_lm(config, model)

    assert "api_base" in captured
    assert captured["model"] == "ollama/qwen3-1.7b-base"


def test_build_task_lm_sagemaker_passes_region(monkeypatch: pytest.MonkeyPatch) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module
    from trajectory_aware_gym.models.experiment import ExperimentConfig, TaskModelConfig

    config = ExperimentConfig.from_yaml(QUICK_TEST_CONFIG)
    model = TaskModelConfig(
        name="Qwen3-4B-Base",
        model_id="sagemaker/qwen3-4b-base",
        provider="sagemaker",
        parameter_count="4B",
    )

    captured: dict = {}

    def fake_lm(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(runner_module.dspy, "LM", fake_lm)
    runner_module._build_task_lm(config, model)

    assert captured["model"] == "sagemaker/qwen3-4b-base"
    assert "api_base" not in captured


# ── _budget_alert_fraction ─────────────────────────────────────────────────


def test_budget_alert_fraction_raw_above_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threshold > 1 is treated as a percentage."""
    import trajectory_aware_gym.experiments.runner as runner_module
    from trajectory_aware_gym.config import settings

    monkeypatch.setattr(
        type(settings),
        "_cost_tracking",
        settings.cost_tracking.model_copy(update={"alert_threshold": 80.0}),
    )
    assert runner_module._budget_alert_fraction() == pytest.approx(0.80)


def test_budget_alert_fraction_raw_at_or_below_1(monkeypatch: pytest.MonkeyPatch) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module
    from trajectory_aware_gym.config import settings

    monkeypatch.setattr(
        type(settings),
        "_cost_tracking",
        settings.cost_tracking.model_copy(update={"alert_threshold": 0.75}),
    )
    assert runner_module._budget_alert_fraction() == pytest.approx(0.75)


def test_budget_alert_fraction_clamped_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module
    from trajectory_aware_gym.config import settings

    monkeypatch.setattr(
        type(settings),
        "_cost_tracking",
        settings.cost_tracking.model_copy(update={"alert_threshold": -0.5}),
    )
    assert runner_module._budget_alert_fraction() == pytest.approx(0.0)


# ── _model_replication_dir ─────────────────────────────────────────────────


def test_model_replication_dir_path_structure(tmp_path: Path) -> None:
    from trajectory_aware_gym.models.experiment import ExperimentConfig, TaskModelConfig

    config = ExperimentConfig.from_yaml(QUICK_TEST_CONFIG)
    model = config.task_models[0]
    args = RunExperimentArgs(config_path=QUICK_TEST_CONFIG, results_root=tmp_path)
    ts = "20260101T000000Z"
    result = _model_replication_dir(args, config, model, 42, ts)
    assert result == tmp_path / "quick-test" / ts / _safe_segment(model.name) / "replication_42"


# ── _config_hash ───────────────────────────────────────────────────────────


def test_config_hash_is_stable() -> None:
    config = ExperimentConfig.from_yaml(QUICK_TEST_CONFIG)
    h1 = _config_hash(config)
    h2 = _config_hash(config)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


# ── _fitness_override_context no-op path ──────────────────────────────────


def test_settings_override_fitness_no_op_with_empty_override() -> None:
    """An empty override dict must not modify settings."""
    from trajectory_aware_gym.config import settings

    original_gamma = settings.fitness.gamma
    with Settings.override_fitness({}):
        assert settings.fitness.gamma == pytest.approx(original_gamma)
    assert settings.fitness.gamma == pytest.approx(original_gamma)


# ── run_experiment: fresh / purge ──────────────────────────────────────────


def _default_eval_summary(
    episodes: int = 1,
    successes: int = 1,
    correct: int = 1,
) -> dict[str, Any]:
    total = episodes
    return {
        "episodes_attempted": total,
        "episodes": total,
        "failed": 0,
        "successes": successes,
        "success_rate": (successes / total) if total else 0.0,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "temperature_eval": 0.0,
    }


def _setup_fake_runner(
    monkeypatch: pytest.MonkeyPatch,
    runner_module: Any,
    *,
    task_cost: float = 0.01,
    task_tokens: int = 1,
    reflection_cost: float = 0.0,
    reflection_tokens: int = 0,
    eval_episodes: int = 1,
    eval_successes: int = 1,
) -> None:
    """Patch runner_module dependencies for lightweight integration tests."""
    train_result = _make_episode_result("t1", success=True, cost=task_cost, tokens=task_tokens)
    eval_result = _make_episode_result("e1", success=True, cost=task_cost, tokens=task_tokens)

    class FakeRunner:
        def __init__(self, **kwargs):
            self.episode_history = (train_result,)

    class FakeSolverModule:
        def __init__(self, runner, default_instructions: str):
            self.instructions = default_instructions

    class FakeGEPA:
        def __init__(self, **kwargs):
            pass

        def compile(self, student, trainset, valset):
            detailed = SimpleNamespace(
                best_idx=0,
                val_aggregate_scores=[0.5],
                val_subscores=[{"a": 1.0}],
            )
            return SimpleNamespace(instructions="prompt", detailed_results=detailed)

    monkeypatch.setattr(runner_module, "_build_trainset", lambda config: [SimpleNamespace()])
    monkeypatch.setattr(runner_module, "GEMEpisodeRunner", FakeRunner)
    monkeypatch.setattr(runner_module, "GEMSolverModule", FakeSolverModule)
    monkeypatch.setattr(runner_module.dspy, "GEPA", FakeGEPA)
    monkeypatch.setattr(runner_module.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(runner_module, "_build_task_lm", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(runner_module, "get_reflection_lm", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(
        runner_module,
        "_extract_reflection_usage",
        lambda lm: (reflection_tokens, reflection_cost),
    )
    monkeypatch.setattr(
        runner_module,
        "_run_heldout_eval",
        lambda **kwargs: (
            [eval_result] * eval_episodes,
            _default_eval_summary(eval_episodes, eval_successes, eval_successes),
        ),
    )


def test_run_experiment_purge_removes_existing_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module

    existing = tmp_path / "quick-test" / "should-be-deleted"
    existing.mkdir(parents=True)
    sentinel = existing / "old_file.txt"
    sentinel.write_text("old content", encoding="utf-8")

    _setup_fake_runner(monkeypatch, runner_module)

    run_experiment(
        RunExperimentArgs(
            config_path=QUICK_TEST_CONFIG,
            results_root=tmp_path,
            purge=True,
        )
    )

    assert not sentinel.exists()


def test_run_experiment_fresh_skips_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """fresh=True creates a new timestamp dir even when a resumable run exists."""
    import trajectory_aware_gym.experiments.runner as runner_module

    # Create a resumable (incomplete) run directory.
    old_ts = "20260101T000000Z"
    old_dir = tmp_path / "quick-test" / old_ts
    old_dir.mkdir(parents=True)
    (old_dir / "run_summary.json").write_text(
        json.dumps({"finished_at": None}),
        encoding="utf-8",
    )

    _setup_fake_runner(monkeypatch, runner_module)

    run_experiment(
        RunExperimentArgs(
            config_path=QUICK_TEST_CONFIG,
            results_root=tmp_path,
            fresh=True,
        )
    )

    # Should have two timestamp dirs: the old one and a new one.
    config_dir = tmp_path / "quick-test"
    ts_dirs = sorted(d.name for d in config_dir.iterdir() if d.is_dir())
    assert len(ts_dirs) == 2
    assert old_ts in ts_dirs


# ── run_experiment: budget alert and halt ─────────────────────────────────


def test_run_experiment_logs_cost_alert(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When total_cost >= alert_level a WARNING is emitted."""
    import logging

    import trajectory_aware_gym.experiments.runner as runner_module

    _setup_fake_runner(
        monkeypatch, runner_module, task_cost=100.0, eval_episodes=0, eval_successes=0
    )
    monkeypatch.setattr(runner_module, "_budget_alert_fraction", lambda: 0.5)

    with caplog.at_level(logging.WARNING, logger="trajectory_aware_gym.experiments.runner"):
        run_experiment(RunExperimentArgs(config_path=QUICK_TEST_CONFIG, results_root=tmp_path))

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Cost alert" in m for m in warning_messages)


def test_run_experiment_cost_budget_exceeded_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    import trajectory_aware_gym.experiments.runner as runner_module

    _setup_fake_runner(
        monkeypatch, runner_module, task_cost=100.0, eval_episodes=0, eval_successes=0
    )

    with caplog.at_level(logging.WARNING, logger="trajectory_aware_gym.experiments.runner"):
        run_experiment(RunExperimentArgs(config_path=QUICK_TEST_CONFIG, results_root=tmp_path))

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Cost budget exceeded" in m for m in warning_messages)


def test_run_experiment_halt_on_budget_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module

    _setup_fake_runner(
        monkeypatch, runner_module, task_cost=100.0, eval_episodes=0, eval_successes=0
    )

    with pytest.raises(RuntimeError, match="Cost budget exceeded"):
        run_experiment(
            RunExperimentArgs(
                config_path=QUICK_TEST_CONFIG,
                results_root=tmp_path,
                halt_on_budget_exceeded=True,
            )
        )


def test_run_experiment_exception_writes_failed_status_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trajectory_aware_gym.experiments.runner as runner_module

    class FakeRunner:
        def __init__(self, **kwargs):
            self.episode_history = ()

    class FakeSolverModule:
        def __init__(self, runner, default_instructions: str):
            self.instructions = default_instructions

    class BrokenGEPA:
        def __init__(self, **kwargs):
            pass

        def compile(self, student, trainset, valset):
            raise RuntimeError("GEPA exploded")

    monkeypatch.setattr(runner_module, "_build_trainset", lambda config: [SimpleNamespace()])
    monkeypatch.setattr(runner_module, "GEMEpisodeRunner", FakeRunner)
    monkeypatch.setattr(runner_module, "GEMSolverModule", FakeSolverModule)
    monkeypatch.setattr(runner_module.dspy, "GEPA", BrokenGEPA)
    monkeypatch.setattr(runner_module.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(runner_module, "_build_task_lm", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(runner_module, "get_reflection_lm", lambda *a, **k: SimpleNamespace())

    with pytest.raises(RuntimeError, match="GEPA exploded"):
        run_experiment(RunExperimentArgs(config_path=QUICK_TEST_CONFIG, results_root=tmp_path))

    replication_dir = _find_replication_dir(tmp_path, "quick-test", "Qwen3-1.7B-Base", 42)
    metadata = json.loads((replication_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["finished_at"] is not None


def test_run_experiment_result_none_when_no_detailed_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When optimized module has no detailed_results the run still completes."""
    import trajectory_aware_gym.experiments.runner as runner_module

    episode = _make_episode_result("t1", success=True, cost=0.01, tokens=1)

    class FakeRunner:
        def __init__(self, **kwargs):
            self.episode_history = (episode,)

    class FakeSolverModule:
        def __init__(self, runner, default_instructions: str):
            self.instructions = default_instructions

    class FakeGEPA:
        def __init__(self, **kwargs):
            pass

        def compile(self, student, trainset, valset):
            # No detailed_results → GEPARunResult.from_module returns None
            return SimpleNamespace(instructions="fallback-prompt")

    monkeypatch.setattr(runner_module, "_build_trainset", lambda config: [SimpleNamespace()])
    monkeypatch.setattr(runner_module, "GEMEpisodeRunner", FakeRunner)
    monkeypatch.setattr(runner_module, "GEMSolverModule", FakeSolverModule)
    monkeypatch.setattr(runner_module.dspy, "GEPA", FakeGEPA)
    monkeypatch.setattr(runner_module.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(runner_module, "_build_task_lm", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(runner_module, "get_reflection_lm", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(runner_module, "_extract_reflection_usage", lambda lm: (0, 0.0))
    monkeypatch.setattr(
        runner_module,
        "_run_heldout_eval",
        lambda **kwargs: (
            [],
            {
                "episodes": 0,
                "successes": 0,
                "success_rate": 0.0,
                "correct": 0,
                "accuracy": 0.0,
                "temperature_eval": 0.0,
            },
        ),
    )

    summary = run_experiment(
        RunExperimentArgs(config_path=QUICK_TEST_CONFIG, results_root=tmp_path)
    )

    assert summary["models"]["Qwen3-1.7B-Base"]["42"]["status"] == "completed"
    replication_dir = _find_replication_dir(tmp_path, "quick-test", "Qwen3-1.7B-Base", 42)
    prompt_text = (replication_dir / "optimized_prompt.txt").read_text(encoding="utf-8")
    assert prompt_text == "fallback-prompt"
    metadata = json.loads((replication_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["result"] is None


# ── CLI script (scripts/run_experiment.py) ─────────────────────────────────


def _import_run_experiment_script():
    import importlib.util
    import sys

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cli_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _import_run_experiment_script()
    monkeypatch.setattr(
        "sys.argv",
        ["run_experiment.py", "--config", "experiments/quick-test/config.yaml"],
    )
    args = mod.parse_args()
    assert args.config == Path("experiments/quick-test/config.yaml")
    assert args.max_metric_calls is None
    assert args.seed_prompt == DEFAULT_SEED_PROMPT
    assert args.models is None
    assert args.seeds is None
    assert args.fresh is False
    assert args.resume is None
    assert args.danger_purge is False
    assert args.results_root == Path("results")
    assert args.halt_on_budget_exceeded is False


def test_cli_parse_args_all_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _import_run_experiment_script()
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_experiment.py",
            "--config",
            "experiments/orz57k/config.yaml",
            "--max-metric-calls",
            "50",
            "--seed-prompt",
            "custom prompt",
            "--models",
            "Qwen3-1.7B-Base",
            "--seeds",
            "42",
            "123",
            "--fresh",
            "--danger-purge",
            "--results-root",
            "/tmp/results",
            "--halt-on-budget-exceeded",
        ],
    )
    args = mod.parse_args()
    assert args.config == Path("experiments/orz57k/config.yaml")
    assert args.max_metric_calls == 50
    assert args.seed_prompt == "custom prompt"
    assert args.models == ["Qwen3-1.7B-Base"]
    assert args.seeds == [42, 123]
    assert args.fresh is True
    assert args.danger_purge is True
    assert args.results_root == Path("/tmp/results")
    assert args.halt_on_budget_exceeded is True


def test_cli_main_calls_run_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _import_run_experiment_script()
    monkeypatch.setattr(
        "sys.argv",
        ["run_experiment.py", "--config", "experiments/quick-test/config.yaml"],
    )
    captured: list = []
    monkeypatch.setattr(mod, "run_experiment", lambda args: captured.append(args) or {})
    mod.main()
    assert len(captured) == 1
    assert captured[0].config_path == Path("experiments/quick-test/config.yaml")


# ── GEPARunResult ──────────────────────────────────────────────────────────


def test_gepa_run_result_from_module_no_detailed_results() -> None:
    from trajectory_aware_gym.models.gepa_result import GEPARunResult

    module = SimpleNamespace()
    assert GEPARunResult.from_module(module, "seed") is None


def test_gepa_run_result_from_module_extracts_correctly() -> None:
    from trajectory_aware_gym.models.gepa_result import GEPARunResult

    detailed = SimpleNamespace(
        best_idx=1,
        val_aggregate_scores=[0.3, 0.9],
        val_subscores=[{"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 0.0}],
    )
    module = SimpleNamespace(detailed_results=detailed, instructions="best prompt")
    result = GEPARunResult.from_module(module, "seed-prompt")

    assert result is not None
    assert result.baseline_fitness == pytest.approx(0.3)
    assert result.final_fitness == pytest.approx(0.9)
    assert result.baseline_accuracy == pytest.approx(0.0)
    assert result.final_accuracy == pytest.approx(0.5)
    assert result.best_program_index == 1
    assert result.optimized_instructions == "best prompt"


def test_gepa_run_result_falls_back_to_seed_when_no_instructions() -> None:
    from trajectory_aware_gym.models.gepa_result import GEPARunResult

    detailed = SimpleNamespace(
        best_idx=0,
        val_aggregate_scores=[0.5],
        val_subscores=[{"x": 1.0}],
    )
    module = SimpleNamespace(detailed_results=detailed)  # no `instructions`
    result = GEPARunResult.from_module(module, "fallback-seed")

    assert result is not None
    assert result.optimized_instructions == "fallback-seed"


def test_accuracy_from_subscores_empty() -> None:
    from trajectory_aware_gym.models.gepa_result import accuracy_from_subscores

    assert accuracy_from_subscores({}) == 0.0


@pytest.mark.parametrize(
    ("subscores", "expected_accuracy"),
    [
        ({"a": 1.0, "b": 0.0, "c": 1.0}, pytest.approx(2 / 3)),
        ({"a": 0.5, "b": 0.5}, pytest.approx(1.0)),
        ({"a": 0.5}, pytest.approx(1.0)),  # 0.5 > 0 → success
        ({"a": 0.0}, pytest.approx(0.0)),
        ({"a": -1.0}, pytest.approx(0.0)),
    ],
)
def test_accuracy_from_subscores_parametrized(subscores, expected_accuracy) -> None:
    from trajectory_aware_gym.models.gepa_result import accuracy_from_subscores

    assert accuracy_from_subscores(subscores) == expected_accuracy
