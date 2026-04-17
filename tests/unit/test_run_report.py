"""Tests for unified run report format."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import (
    SCHEMA_VERSION,
    LLMCallMetadata,
    TrajectoryLog,
    TrajectoryStep,
)
from trajectory_aware_gym.metrics.run_report import RunReport, build_run_report
from trajectory_aware_gym.storage import save_experiment_run, save_trajectory
from trajectory_aware_gym.storage.models import ExperimentRunRecord


def _make_run_record(
    *,
    provider: str = "bedrock",
    model_id: str = "bedrock/us.meta.llama3-1-8b-instruct-v1:0",
) -> ExperimentRunRecord:
    return ExperimentRunRecord(
        experiment_run_id=f"test-{provider}-run-1",
        config_name="quick-test",
        config_hash="abc123",
        config_yaml="config: test",
        operator="test-user",
        git_commit="deadbeef",
        git_branch="development",
        provider=provider,
        task_model_id=model_id,
        reflection_model_id="openai.gpt-oss-120b-1:0",
        environment_id="math:Orz57K",
        gepa_budget_mode="light",
        replication_seed=42,
        seed_prompt="Solve the problem.",
        started_at=datetime(2026, 4, 15, 14, 30, tzinfo=UTC),
        status="completed",
        schema_version=SCHEMA_VERSION,
    )


_COST_SUMMARY = {
    "task_model_tokens": 1000,
    "task_model_cost": 0.01,
    "reflection_tokens": 200,
    "reflection_cost": 0.005,
    "total_tokens": 1200,
    "total_cost": 0.015,
}

_EVAL_SUMMARY = {
    "episodes": 5,
    "successes": 3,
    "accuracy": 0.6,
}


class TestRunReportModel:
    def test_bedrock_report_has_actual_cost_type(self, tmp_path: Path) -> None:
        record = _make_run_record(provider="bedrock")
        db = tmp_path / "test.db"
        save_experiment_run(db, record)

        report = build_run_report(
            experiment_run_id=record.experiment_run_id,
            db_path=db,
            cost_summary=_COST_SUMMARY,
            eval_summary=_EVAL_SUMMARY,
        )

        assert report.cost_type == "actual"
        assert report.normalized_cost_usd is None
        assert report.normalization_reference is None
        assert report.total_cost_usd == 0.015
        assert report.total_tokens == 1200

    def test_ollama_report_has_normalized_cost(self, tmp_path: Path) -> None:
        record = _make_run_record(
            provider="ollama",
            model_id="ollama/qwen3-1.7b-base",
        )
        db = tmp_path / "test.db"
        save_experiment_run(db, record)

        ref_prices = {
            "ollama/qwen3-1.7b-base": {
                "input_per_1m_tokens": 0.10,
                "output_per_1m_tokens": 0.10,
            }
        }

        report = build_run_report(
            experiment_run_id=record.experiment_run_id,
            db_path=db,
            cost_summary=_COST_SUMMARY,
            reference_prices=ref_prices,
        )

        assert report.cost_type == "unavailable"
        assert report.normalized_cost_usd is not None
        assert report.normalized_cost_usd > 0
        assert report.normalization_reference is not None
        assert "qwen3-1.7b-base" in report.normalization_reference

    def test_prompt_token_ratio_can_be_overridden(self, tmp_path: Path) -> None:
        record = _make_run_record(provider="ollama", model_id="ollama/qwen3-1.7b-base")
        db = tmp_path / "test.db"
        save_experiment_run(db, record)

        report = build_run_report(
            experiment_run_id=record.experiment_run_id,
            db_path=db,
            cost_summary={"task_model_tokens": 1000},
            reference_prices={
                "ollama/qwen3-1.7b-base": {
                    "input_per_1m_tokens": 1.0,
                    "output_per_1m_tokens": 3.0,
                }
            },
            prompt_token_ratio=0.25,
        )

        assert report.normalized_cost_usd == pytest.approx(0.0025)

    def test_report_json_has_identical_keys_regardless_of_provider(
        self,
        tmp_path: Path,
    ) -> None:
        db = tmp_path / "test.db"

        bedrock_record = _make_run_record(provider="bedrock")
        bedrock_record = bedrock_record.model_copy(
            update={"experiment_run_id": "test-bedrock-run-keys"}
        )
        save_experiment_run(db, bedrock_record)

        ollama_record = _make_run_record(
            provider="ollama",
            model_id="ollama/qwen3-1.7b-base",
        )
        ollama_record = ollama_record.model_copy(
            update={"experiment_run_id": "test-ollama-run-keys"}
        )
        save_experiment_run(db, ollama_record)

        bedrock_report = build_run_report(
            experiment_run_id=bedrock_record.experiment_run_id,
            db_path=db,
            cost_summary=_COST_SUMMARY,
        )
        ollama_report = build_run_report(
            experiment_run_id=ollama_record.experiment_run_id,
            db_path=db,
            cost_summary=_COST_SUMMARY,
        )

        bedrock_keys = set(json.loads(bedrock_report.model_dump_json()).keys())
        ollama_keys = set(json.loads(ollama_report.model_dump_json()).keys())
        assert bedrock_keys == ollama_keys

    def test_report_includes_eval_summaries(self, tmp_path: Path) -> None:
        record = _make_run_record()
        db = tmp_path / "test.db"
        save_experiment_run(db, record)

        validation_baseline = {"accuracy": 0.1, "episodes": 5, "correct": 0}
        validation_optimized = {"accuracy": 0.4, "episodes": 5, "correct": 2}
        baseline = {"accuracy": 0.3, "episodes": 10}
        optimized = {"accuracy": 0.7, "episodes": 10}

        report = build_run_report(
            experiment_run_id=record.experiment_run_id,
            db_path=db,
            cost_summary=_COST_SUMMARY,
            baseline_validation_summary=validation_baseline,
            optimized_validation_summary=validation_optimized,
            baseline_eval_summary=baseline,
            eval_summary=optimized,
            wall_clock_seconds=120.5,
        )

        assert report.baseline_validation == validation_baseline
        assert report.optimized_validation == validation_optimized
        assert report.baseline_eval == baseline
        assert report.eval_summary == optimized
        assert report.wall_clock_seconds == 120.5

    def test_report_metadata_from_db(self, tmp_path: Path) -> None:
        record = _make_run_record()
        db = tmp_path / "test.db"
        save_experiment_run(db, record)

        report = build_run_report(
            experiment_run_id=record.experiment_run_id,
            db_path=db,
            cost_summary=_COST_SUMMARY,
        )

        assert report.config_name == "quick-test"
        assert report.operator == "test-user"
        assert report.provider == "bedrock"
        assert report.environment_id == "math:Orz57K"
        assert report.seed == 42
        assert report.git_commit == "deadbeef"

    def test_partial_cost_and_logging_summary_round_trip(self, tmp_path: Path) -> None:
        record = _make_run_record(provider="ollama", model_id="ollama/qwen3-1.7b-base")
        db = tmp_path / "test.db"
        save_experiment_run(db, record)

        logging_summary = {
            "status": "partial",
            "trajectory_persisted_episodes": 1,
            "trajectory_failed_episodes": 0,
            "metrics_unavailable_episodes": 1,
            "numeric_anomaly_count": 1,
            "events": [],
            "events_truncated": False,
        }
        cost_summary = {
            "task_model_tokens": None,
            "task_model_tokens_known": 1000,
            "task_model_token_data_coverage": 0.5,
            "task_model_cost": None,
            "task_model_cost_known": 0.01,
            "task_model_cost_data_coverage": 0.5,
            "reflection_cost": 0.005,
            "total_tokens": None,
            "total_tokens_known": 1200,
            "total_cost": None,
            "total_cost_known": 0.015,
            "cost_type": "partial",
        }

        report = build_run_report(
            experiment_run_id=record.experiment_run_id,
            db_path=db,
            cost_summary=cost_summary,
            logging_summary=logging_summary,
            reference_prices={
                "ollama/qwen3-1.7b-base": {
                    "input_per_1m_tokens": 0.10,
                    "output_per_1m_tokens": 0.10,
                }
            },
        )

        assert report.cost_type == "partial"
        assert report.total_cost_usd is None
        assert report.total_cost_known_usd == pytest.approx(0.015)
        assert report.logging_summary == logging_summary
        assert report.normalized_cost_usd is None

    def test_mean_latency_is_scoped_to_experiment_run(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        record_a = _make_run_record().model_copy(update={"experiment_run_id": "run-a"})
        record_b = _make_run_record().model_copy(update={"experiment_run_id": "run-b"})
        save_experiment_run(db, record_a)
        save_experiment_run(db, record_b)

        def make_trajectory(latency_ms: float) -> TrajectoryLog:
            step = TrajectoryStep(
                step_index=1,
                action="answer",
                observation="done",
                reward=1.0,
                terminated=True,
                truncated=False,
                info={},
                llm_calls=[
                    LLMCallMetadata(
                        model_id="bedrock/us.meta.llama3-1-8b-instruct-v1:0",
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        latency_ms=latency_ms,
                    )
                ],
            )
            return TrajectoryLog(
                environment_id="math:Orz57K",
                started_at=datetime(2026, 4, 15, 14, 30, tzinfo=UTC),
                finished_at=datetime(2026, 4, 15, 14, 31, tzinfo=UTC),
                initial_observation="start",
                steps=[step],
                total_reward=1.0,
            )

        save_trajectory(db, make_trajectory(100.0), experiment_run_id="run-a")
        save_trajectory(db, make_trajectory(300.0), experiment_run_id="run-b")

        report = build_run_report(
            experiment_run_id="run-a",
            db_path=db,
            cost_summary=_COST_SUMMARY,
        )

        assert report.mean_llm_latency_ms == pytest.approx(100.0)
