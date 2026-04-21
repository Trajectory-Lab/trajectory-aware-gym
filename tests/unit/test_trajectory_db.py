"""Tests for SQLite-backed trajectory storage."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import (
    SCHEMA_VERSION,
    LLMCallMetadata,
    ToolCall,
    TrajectoryLog,
    TrajectoryLogger,
    TrajectoryStep,
    load_all_trajectories,
    load_trajectory,
)
from trajectory_aware_gym.config import ProjectPaths
from trajectory_aware_gym.storage.models import ExperimentRunRecord
from trajectory_aware_gym.storage.trajectory_db import (
    close_connection,
    episode_exists,
    get_connection,
    load_experiment_run,
    load_trajectory_by_id,
    query_experiment_runs,
    query_trajectories,
    save_experiment_run,
    save_tool_call_entry,
    save_trajectory,
    update_experiment_run,
)
from trajectory_aware_gym.storage.trajectory_db import (
    load_all_trajectories as db_load_all,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    step_index: int = 1,
    *,
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    info: dict[str, Any] | None = None,
    tool_calls: list[ToolCall] | None = None,
    llm_calls: list[LLMCallMetadata] | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=step_index,
        action="act",
        observation="obs",
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info or {},
        tool_calls=tool_calls or [],
        llm_calls=llm_calls or [],
        timestamp=datetime.now(UTC),
    )


def _make_log(
    steps: list[TrajectoryStep] | None = None,
    **overrides: Any,
) -> TrajectoryLog:
    now = datetime.now(UTC)
    steps = steps or []
    defaults: dict[str, Any] = {
        "environment_id": "game:GuessTheNumber-v0-easy",
        "seed": 1,
        "started_at": now,
        "finished_at": now + timedelta(seconds=1),
        "initial_observation": "start",
        "steps": steps,
        "total_reward": sum(s.reward for s in steps),
    }
    defaults.update(overrides)
    return TrajectoryLog(**defaults)


def _create_legacy_v11_db(db_path) -> None:
    """Create a pre-v1.2 SQLite schema to exercise migration-on-open."""

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE episodes (
            run_id              TEXT PRIMARY KEY,
            schema_version      TEXT NOT NULL,
            environment_id      TEXT NOT NULL,
            seed                INTEGER,
            system_prompt       TEXT,
            started_at          TEXT NOT NULL,
            finished_at         TEXT NOT NULL,
            initial_observation TEXT NOT NULL,
            initial_info        TEXT NOT NULL,
            total_reward        REAL NOT NULL,
            episode_outcome     TEXT
        );

        CREATE TABLE steps (
            run_id      TEXT    NOT NULL,
            step_index  INTEGER NOT NULL,
            action      TEXT    NOT NULL,
            observation TEXT    NOT NULL,
            reward      REAL    NOT NULL,
            terminated  INTEGER NOT NULL,
            truncated   INTEGER NOT NULL,
            info        TEXT    NOT NULL,
            timestamp   TEXT,
            PRIMARY KEY (run_id, step_index)
        );

        CREATE TABLE llm_calls (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id            TEXT    NOT NULL,
            step_index        INTEGER NOT NULL,
            model_id          TEXT    NOT NULL,
            prompt_tokens     INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens      INTEGER NOT NULL,
            cost_usd          REAL,
            latency_ms        REAL,
            FOREIGN KEY (run_id, step_index) REFERENCES steps(run_id, step_index)
        );

        CREATE TABLE tool_calls (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT    NOT NULL,
            step_index  INTEGER NOT NULL,
            tool_name   TEXT    NOT NULL,
            tool_input  TEXT    NOT NULL,
            tool_output TEXT    NOT NULL,
            success     INTEGER NOT NULL,
            FOREIGN KEY (run_id, step_index) REFERENCES steps(run_id, step_index)
        );

        CREATE TABLE tool_call_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            tool      TEXT NOT NULL,
            args      TEXT NOT NULL,
            result    TEXT NOT NULL
        );
        """
    )
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    """Yield a fresh SQLite database path and clean up the connection after."""
    path = tmp_path / "test.db"
    yield path
    close_connection(path)


# ===========================================================================
# Save and load round-trip
# ===========================================================================


class TestSaveAndLoad:
    """Round-trip: save a TrajectoryLog, load it back, verify equality."""

    def test_empty_episode(self, db_path):
        log = _make_log()
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)
        assert loaded.run_id == log.run_id
        assert loaded.environment_id == log.environment_id
        assert loaded.steps == []
        assert loaded.total_reward == 0.0

    def test_single_step_round_trip(self, db_path):
        step = _make_step(reward=1.0, terminated=True, info={"correct": True})
        log = _make_log(steps=[step])
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)

        assert loaded.num_steps == 1
        assert loaded.total_reward == 1.0
        assert loaded.steps[0].reward == 1.0
        assert loaded.steps[0].info == {"correct": True}
        assert loaded.steps[0].terminated is True

    def test_multi_step_with_tool_and_llm_calls(self, db_path):
        tc = ToolCall(
            tool_name="python_exec",
            tool_input='print("hi")',
            tool_output="hi",
            success=True,
            duration_ms=42.0,
        )
        lc = LLMCallMetadata(
            model_id="bedrock/qwen3-1.7b",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            latency_ms=150.0,
        )
        steps = [
            _make_step(step_index=1, reward=0.0, tool_calls=[tc], llm_calls=[lc]),
            _make_step(step_index=2, reward=1.0, terminated=True, llm_calls=[lc, lc]),
        ]
        log = _make_log(steps=steps, system_prompt="Be helpful")
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)

        assert loaded.num_steps == 2
        assert loaded.system_prompt == "Be helpful"
        assert len(loaded.steps[0].tool_calls) == 1
        assert loaded.steps[0].tool_calls[0].tool_name == "python_exec"
        assert loaded.steps[0].tool_calls[0].duration_ms == 42.0
        assert len(loaded.steps[0].llm_calls) == 1
        assert loaded.steps[0].llm_calls[0].provider == "bedrock"
        assert loaded.steps[0].llm_calls[0].token_usage_known is True
        assert len(loaded.steps[1].llm_calls) == 2
        assert loaded.steps[1].llm_calls[0].cost_usd == 0.001

    @pytest.mark.parametrize(
        ("model_id", "expected_provider"),
        [
            ("bedrock/llama-8b", "bedrock"),
            ("ollama/qwen3-1.7b", "ollama"),
            ("sagemaker/qwen3-4b", "sagemaker"),
            ("unknown-model", None),
        ],
    )
    def test_llm_call_provider_round_trip(self, db_path, model_id, expected_provider):
        lc = LLMCallMetadata(
            model_id=model_id,
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
        )
        step = _make_step(reward=0.0, terminated=True, llm_calls=[lc])
        log = _make_log(steps=[step])
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)
        assert loaded.steps[0].llm_calls[0].provider == expected_provider

    @pytest.mark.parametrize(
        ("cost_usd", "cost_type"),
        [
            (0.005, "actual"),
            (None, "unavailable"),
            (0.0, "actual"),
            (None, None),
        ],
    )
    def test_llm_call_cost_type_round_trip(self, db_path, cost_usd, cost_type):
        lc = LLMCallMetadata(
            model_id="bedrock/llama-8b",
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
            cost_usd=cost_usd,
            cost_type=cost_type,
        )
        step = _make_step(reward=0.0, terminated=True, llm_calls=[lc])
        log = _make_log(steps=[step])
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)
        loaded_lc = loaded.steps[0].llm_calls[0]
        assert loaded_lc.cost_usd == cost_usd
        assert loaded_lc.cost_type == cost_type

    def test_llm_call_unknown_token_usage_round_trip(self, db_path):
        lc = LLMCallMetadata(
            model_id="bedrock/llama-8b",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            token_usage_known=False,
        )
        step = _make_step(reward=0.0, terminated=True, llm_calls=[lc])
        log = _make_log(steps=[step])
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)

        assert loaded.steps[0].llm_calls[0].token_usage_known is False

    def test_seed_none_preserved(self, db_path):
        log = _make_log(seed=None)
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)
        assert loaded.seed is None

    def test_episode_outcome_preserved(self, db_path):
        step = _make_step(reward=1.0, terminated=True, info={"correct": True})
        log = _make_log(steps=[step], episode_outcome="success")
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)
        assert loaded.episode_outcome == "success"

    def test_timestamps_round_trip(self, db_path):
        now = datetime.now(UTC)
        step = _make_step(terminated=True)
        log = _make_log(steps=[step], started_at=now, finished_at=now + timedelta(seconds=5))
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)

        assert abs((loaded.started_at - log.started_at).total_seconds()) < 0.01
        assert abs((loaded.finished_at - log.finished_at).total_seconds()) < 0.01


# ===========================================================================
# Load all and query
# ===========================================================================


class TestLoadAllAndQuery:
    def test_load_all_empty(self, db_path):
        assert db_load_all(db_path) == []

    def test_load_all_multiple(self, db_path):
        now = datetime.now(UTC)
        for i in range(3):
            step = _make_step(reward=float(i), terminated=True)
            log = _make_log(
                steps=[step],
                started_at=now + timedelta(seconds=i),
                finished_at=now + timedelta(seconds=i + 1),
            )
            save_trajectory(db_path, log)

        all_logs = db_load_all(db_path)
        assert len(all_logs) == 3
        # Ordered by started_at
        assert all_logs[0].started_at <= all_logs[1].started_at <= all_logs[2].started_at

    def test_query_by_outcome(self, db_path):
        step_ok = _make_step(reward=1.0, terminated=True, info={"correct": True})
        step_fail = _make_step(reward=0.0, terminated=True, info={"correct": False})
        save_trajectory(db_path, _make_log(steps=[step_ok], episode_outcome="success"))
        save_trajectory(db_path, _make_log(steps=[step_fail], episode_outcome="failure"))

        successes = query_trajectories(db_path, outcome="success")
        assert len(successes) == 1
        assert successes[0].episode_outcome == "success"

    def test_query_by_environment(self, db_path):
        step = _make_step(terminated=True)
        save_trajectory(db_path, _make_log(steps=[step], environment_id="math:Orz57K"))
        save_trajectory(db_path, _make_log(steps=[step], environment_id="qa:HotpotQA"))

        results = query_trajectories(db_path, environment_id="math:Orz57K")
        assert len(results) == 1
        assert results[0].environment_id == "math:Orz57K"

    def test_query_combined_filters(self, db_path):
        step_ok = _make_step(reward=1.0, terminated=True, info={"correct": True})
        step_fail = _make_step(reward=0.0, terminated=True, info={"correct": False})
        save_trajectory(
            db_path,
            _make_log(steps=[step_ok], environment_id="math:Orz57K", episode_outcome="success"),
        )
        save_trajectory(
            db_path,
            _make_log(steps=[step_fail], environment_id="math:Orz57K", episode_outcome="failure"),
        )
        save_trajectory(
            db_path,
            _make_log(steps=[step_ok], environment_id="qa:HotpotQA", episode_outcome="success"),
        )

        results = query_trajectories(db_path, outcome="success", environment_id="math:Orz57K")
        assert len(results) == 1


# ===========================================================================
# episode_exists
# ===========================================================================


class TestEpisodeExists:
    def test_exists_after_save(self, db_path):
        log = _make_log()
        assert not episode_exists(db_path, log.run_id)
        save_trajectory(db_path, log)
        assert episode_exists(db_path, log.run_id)

    def test_not_exists(self, db_path):
        assert not episode_exists(db_path, "nonexistent-id")


# ===========================================================================
# Tool call log
# ===========================================================================


class TestToolCallLog:
    def test_save_tool_call(self, db_path):
        from trajectory_aware_gym.storage.trajectory_db import get_connection

        save_tool_call_entry(
            db_path,
            timestamp="2026-03-26T12:00:00+00:00",
            tool="python_exec",
            args={"code": "print(1)"},
            result={"status": "success", "output": "1"},
        )
        conn = get_connection(db_path)
        rows = conn.execute("SELECT * FROM tool_call_log").fetchall()
        assert len(rows) == 1
        assert rows[0]["tool"] == "python_exec"


# ===========================================================================
# load_trajectory dispatch (JSON vs DB)
# ===========================================================================


class TestLoadTrajectoryDispatch:
    def test_load_from_db(self, db_path):
        log = _make_log()
        save_trajectory(db_path, log)
        loaded = load_trajectory(db_path, run_id=log.run_id)
        assert loaded.run_id == log.run_id

    def test_load_from_db_requires_run_id(self, db_path):
        with pytest.raises(ValueError, match="run_id is required"):
            load_trajectory(db_path)

    def test_load_from_json(self, tmp_path):
        log = _make_log()
        json_path = tmp_path / "trajectory.json"
        json_path.write_text(log.model_dump_json(indent=2), encoding="utf-8")
        loaded = load_trajectory(json_path)
        assert loaded.run_id == log.run_id

    def test_missing_run_id_raises(self, db_path):
        # Ensure the DB exists so it doesn't fail on connection
        save_trajectory(db_path, _make_log())
        with pytest.raises(KeyError):
            load_trajectory(db_path, run_id="nonexistent")


# ===========================================================================
# load_all_trajectories (directory-level dispatch)
# ===========================================================================


class TestLoadAllTrajectories:
    def test_prefers_sqlite(self, tmp_path):
        """When trajectories.db exists, loads from it instead of JSON files."""
        db = tmp_path / "trajectories.db"
        log = _make_log()
        save_trajectory(db, log)

        # Also create a JSON file that would be found by glob
        other_log = _make_log(environment_id="other:env")
        json_path = tmp_path / f"trajectory_20260326T000000Z_{other_log.run_id}.json"
        json_path.write_text(other_log.model_dump_json(), encoding="utf-8")

        results = load_all_trajectories(tmp_path)
        # Should only get the SQLite entry, not the JSON
        assert len(results) == 1
        assert results[0].run_id == log.run_id
        close_connection(db)

    def test_falls_back_to_json(self, tmp_path):
        """Without a .db file, falls back to JSON glob."""
        log = _make_log()
        json_path = tmp_path / f"trajectory_20260326T000000Z_{log.run_id}.json"
        json_path.write_text(log.model_dump_json(), encoding="utf-8")

        results = load_all_trajectories(tmp_path)
        assert len(results) == 1
        assert results[0].run_id == log.run_id


# ===========================================================================
# TrajectoryLogger.save() integration
# ===========================================================================


class TestTrajectoryLoggerSave:
    def test_save_returns_db_path(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        logger = TrajectoryLogger(environment_id="test:env", seed=42)
        logger.set_initial_state("start")
        logger.add_step(
            action="go", observation="end", reward=1.0, terminated=True, truncated=False
        )
        result_path = logger.save(project_paths=paths)
        assert result_path.suffix == ".db"
        assert result_path.name == "trajectories.db"

    def test_last_run_id_set_after_save(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        logger = TrajectoryLogger(environment_id="test:env", seed=42)
        logger.set_initial_state("start")
        logger.add_step(
            action="go", observation="end", reward=1.0, terminated=True, truncated=False
        )
        assert logger.last_run_id is None
        db_path = logger.save(project_paths=paths)
        assert logger.last_run_id is not None

        # Verify we can load it back
        loaded = load_trajectory(db_path, run_id=logger.last_run_id)
        assert loaded.environment_id == "test:env"
        close_connection(db_path)

    def test_multiple_saves_to_same_db(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        for i in range(3):
            logger = TrajectoryLogger(environment_id="test:env", seed=i)
            logger.set_initial_state("start")
            logger.add_step(
                action="go", observation="end", reward=1.0, terminated=True, truncated=False
            )
            logger.save(project_paths=paths)

        db = tmp_path / "logs" / "trajectories.db"
        all_logs = db_load_all(db)
        assert len(all_logs) == 3
        close_connection(db)


# ===========================================================================
# Duplicate run_id
# ===========================================================================


class TestDuplicateRunId:
    def test_save_duplicate_raises_value_error(self, db_path):
        log = _make_log()
        save_trajectory(db_path, log)
        with pytest.raises(ValueError, match=log.run_id):
            save_trajectory(db_path, log)

    def test_duplicate_does_not_corrupt_existing(self, db_path):
        log = _make_log(steps=[_make_step(reward=1.0, terminated=True)])
        save_trajectory(db_path, log)
        try:
            save_trajectory(db_path, log)
        except ValueError:
            pass
        loaded = load_trajectory_by_id(db_path, log.run_id)
        assert loaded.run_id == log.run_id
        assert loaded.total_reward == 1.0


# ===========================================================================
# None timestamp on step
# ===========================================================================


class TestNoneTimestamp:
    def test_step_with_none_timestamp_round_trips(self, db_path):
        step = TrajectoryStep(
            step_index=1,
            action="act",
            observation="obs",
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            timestamp=None,
            tool_calls=[],
            llm_calls=[],
        )
        log = _make_log(steps=[step])
        save_trajectory(db_path, log)
        loaded = load_trajectory_by_id(db_path, log.run_id)
        assert loaded.steps[0].timestamp is None


# ===========================================================================
# query_trajectories edge cases
# ===========================================================================


class TestQueryEdgeCases:
    def test_query_no_filters_returns_all(self, db_path):
        """query_trajectories with no filters behaves like load_all_trajectories."""
        save_trajectory(db_path, _make_log(episode_outcome="success"))
        save_trajectory(db_path, _make_log(episode_outcome="failure"))
        results = query_trajectories(db_path)
        assert len(results) == 2

    def test_query_filter_no_matches_returns_empty(self, db_path):
        save_trajectory(db_path, _make_log(episode_outcome="success"))
        results = query_trajectories(db_path, outcome="failure")
        assert results == []

    def test_query_by_experiment_run_id(self, db_path):
        run_a = _make_experiment_run(experiment_run_id="exp-a")
        run_b = _make_experiment_run(experiment_run_id="exp-b")
        save_experiment_run(db_path, run_a)
        save_experiment_run(db_path, run_b)

        save_trajectory(
            db_path, _make_log(steps=[_make_step(terminated=True)]), experiment_run_id="exp-a"
        )
        save_trajectory(
            db_path, _make_log(steps=[_make_step(terminated=True)]), experiment_run_id="exp-b"
        )

        results = query_trajectories(db_path, experiment_run_id="exp-a")

        assert len(results) == 1


# ===========================================================================
# Concurrent WAL access
# ===========================================================================


class TestConcurrentAccess:
    def test_concurrent_write_and_read(self, db_path):
        """Two threads can write and read simultaneously under WAL mode."""
        import threading

        errors: list[Exception] = []
        written_ids: list[str] = []

        def writer() -> None:
            try:
                for _ in range(5):
                    log = _make_log()
                    save_trajectory(db_path, log)
                    written_ids.append(log.run_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                close_connection(db_path)

        def reader() -> None:
            try:
                for _ in range(5):
                    db_load_all(db_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                close_connection(db_path)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        writer_thread.start()
        reader_thread.start()
        writer_thread.join()
        reader_thread.join()

        assert errors == [], f"Concurrent access errors: {errors}"
        assert len(written_ids) == 5


class TestSchemaMigration:
    def test_legacy_db_is_migrated_on_first_write(self, db_path):
        _create_legacy_v11_db(db_path)

        run = _make_experiment_run(experiment_run_id="migrated-run")
        save_experiment_run(db_path, run)

        step = _make_step(
            reward=1.0,
            terminated=True,
            llm_calls=[
                LLMCallMetadata(
                    model_id="bedrock/llama-8b",
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cost_type="actual",
                )
            ],
        )
        log = _make_log(steps=[step])
        save_trajectory(db_path, log, experiment_run_id="migrated-run")

        from trajectory_aware_gym.storage.trajectory_db import get_connection

        conn = get_connection(db_path)
        episode_columns = {row["name"] for row in conn.execute("PRAGMA table_info(episodes)")}
        llm_call_columns = {row["name"] for row in conn.execute("PRAGMA table_info(llm_calls)")}
        tool_call_columns = {row["name"] for row in conn.execute("PRAGMA table_info(tool_calls)")}

        assert "experiment_run_id" in episode_columns
        assert "provider" in llm_call_columns
        assert "cost_type" in llm_call_columns
        assert "token_usage_known" in llm_call_columns
        assert "duration_ms" in tool_call_columns
        assert (
            conn.execute("SELECT version FROM schema_meta").fetchone()["version"] == SCHEMA_VERSION
        )

        loaded = load_trajectory_by_id(db_path, log.run_id)
        assert loaded.steps[0].llm_calls[0].provider == "bedrock"
        assert loaded.steps[0].llm_calls[0].cost_type == "actual"
        assert loaded.steps[0].llm_calls[0].token_usage_known is True

    def test_new_db_records_schema_version(self, db_path):
        save_experiment_run(db_path, _make_experiment_run(experiment_run_id="schema-meta-run"))

        conn = get_connection(db_path)
        row = conn.execute("SELECT version FROM schema_meta").fetchone()

        assert row is not None
        assert row["version"] == SCHEMA_VERSION

    def test_newer_schema_version_fails_fast(self, db_path):
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE schema_meta (version TEXT PRIMARY KEY);
            INSERT INTO schema_meta (version) VALUES ('9.9.9');
            """
        )
        conn.close()

        with pytest.raises(RuntimeError, match="newer than supported"):
            get_connection(db_path)


# ===========================================================================
# Migration script
# ===========================================================================


class TestMigrateTrajectories:
    def test_migrates_json_files(self, tmp_path):
        from scripts.migrate_json_to_sqlite import migrate_trajectories

        log = _make_log()
        json_path = tmp_path / f"trajectory_{log.run_id}.json"
        json_path.write_text(log.model_dump_json(), encoding="utf-8")

        db = tmp_path / "trajectories.db"
        migrated, skipped, migrated_files = migrate_trajectories(tmp_path, db)

        assert migrated == 1
        assert skipped == 0
        assert len(migrated_files) == 1
        assert episode_exists(db, log.run_id)
        close_connection(db)

    def test_skips_already_in_db(self, tmp_path):
        from scripts.migrate_json_to_sqlite import migrate_trajectories

        log = _make_log()
        json_path = tmp_path / f"trajectory_{log.run_id}.json"
        json_path.write_text(log.model_dump_json(), encoding="utf-8")
        db = tmp_path / "trajectories.db"
        save_trajectory(db, log)  # pre-populate

        migrated, skipped, migrated_files = migrate_trajectories(tmp_path, db)

        assert migrated == 0
        assert skipped == 1
        assert len(migrated_files) == 1  # still included for potential cleanup
        close_connection(db)

    def test_skips_corrupt_json(self, tmp_path):
        from scripts.migrate_json_to_sqlite import migrate_trajectories

        corrupt = tmp_path / "trajectory_bad.json"
        corrupt.write_text("{not valid json", encoding="utf-8")
        db = tmp_path / "trajectories.db"

        migrated, skipped, migrated_files = migrate_trajectories(tmp_path, db)

        assert migrated == 0
        assert skipped == 1
        assert migrated_files == []
        close_connection(db)


class TestMigrateToolCalls:
    def test_migrates_valid_lines(self, tmp_path):
        import json as _json

        from scripts.migrate_json_to_sqlite import migrate_tool_calls

        jsonl = tmp_path / "tool_calls.jsonl"
        entry = {
            "timestamp": "2026-01-01T00:00:00Z",
            "tool": "python_exec",
            "args": {},
            "result": {},
        }
        jsonl.write_text(_json.dumps(entry) + "\n", encoding="utf-8")
        db = tmp_path / "trajectories.db"

        migrated, skipped = migrate_tool_calls(jsonl, db)
        assert migrated == 1
        assert skipped == 0
        close_connection(db)

    def test_skips_corrupt_lines(self, tmp_path):
        from scripts.migrate_json_to_sqlite import migrate_tool_calls

        jsonl = tmp_path / "tool_calls.jsonl"
        jsonl.write_text("{corrupt\n", encoding="utf-8")
        db = tmp_path / "trajectories.db"

        migrated, skipped = migrate_tool_calls(jsonl, db)
        assert migrated == 0
        assert skipped == 1
        close_connection(db)

    def test_returns_zero_when_file_absent(self, tmp_path):
        from scripts.migrate_json_to_sqlite import migrate_tool_calls

        db = tmp_path / "trajectories.db"
        migrated, skipped = migrate_tool_calls(tmp_path / "tool_calls.jsonl", db)
        assert migrated == 0
        assert skipped == 0


class TestVerifyAndClean:
    def test_deletes_verified_files(self, tmp_path):
        from scripts.migrate_json_to_sqlite import verify_and_clean

        log = _make_log()
        db = tmp_path / "trajectories.db"
        save_trajectory(db, log)

        json_path = tmp_path / f"trajectory_{log.run_id}.json"
        json_path.write_text(log.model_dump_json(), encoding="utf-8")

        deleted, failed = verify_and_clean(
            db, [(json_path, log.run_id)], tmp_path / "tool_calls.jsonl", tmp_path
        )

        assert deleted == 1
        assert failed == 0
        assert not json_path.exists()
        close_connection(db)

    def test_keeps_file_not_in_db(self, tmp_path):
        from scripts.migrate_json_to_sqlite import verify_and_clean

        db = tmp_path / "trajectories.db"
        # Create the DB (empty)
        save_trajectory(db, _make_log())  # need at least one row so DB is initialized
        close_connection(db)

        json_path = tmp_path / "trajectory_missing.json"
        json_path.write_text("{}", encoding="utf-8")
        db2 = tmp_path / "trajectories.db"

        deleted, failed = verify_and_clean(
            db2, [(json_path, "nonexistent-id")], tmp_path / "tool_calls.jsonl", tmp_path
        )

        assert deleted == 0
        assert failed == 1
        assert json_path.exists()
        close_connection(db2)

    def test_rejects_path_outside_input_dir(self, tmp_path):
        from scripts.migrate_json_to_sqlite import verify_and_clean

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        outside_file = other_dir / "trajectory_x.json"
        outside_file.write_text("{}", encoding="utf-8")

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        db = input_dir / "trajectories.db"

        deleted, failed = verify_and_clean(
            db, [(outside_file, "some-id")], input_dir / "tool_calls.jsonl", input_dir
        )

        assert deleted == 0
        assert failed == 1
        assert outside_file.exists()  # was not deleted


# ===========================================================================
# Experiment run CRUD
# ===========================================================================


def _make_experiment_run(**overrides: Any) -> ExperimentRunRecord:
    """Build a minimal ExperimentRunRecord for testing."""
    now = datetime.now(UTC)
    defaults: dict[str, Any] = {
        "experiment_run_id": f"test-run-{now.timestamp()}",
        "config_name": "quick-test",
        "config_hash": "abc123",
        "config_yaml": "model: ollama/qwen3-1.7b\n",
        "operator": "test-user",
        "provider": "ollama",
        "task_model_id": "ollama/qwen3-1.7b",
        "environment_id": "math:Orz57K",
        "started_at": now,
        "schema_version": SCHEMA_VERSION,
    }
    defaults.update(overrides)
    return ExperimentRunRecord(**defaults)


class TestExperimentRunCRUD:
    """Tests for save/load/query/update of experiment_runs."""

    def test_save_and_load_round_trip(self, db_path):
        """5.11: save → load → assert all fields match."""
        run = _make_experiment_run(
            experiment_run_id="run-rt-001",
            config_name="orz57k-gepa-light",
            config_hash="sha256abc",
            config_yaml="budget: light\nmodel: ollama/qwen3\n",
            operator="jinyuhan",
            git_commit="abc1234",
            git_branch="feat/logging-v2",
            provider="ollama",
            task_model_id="ollama/qwen3-1.7b",
            reflection_model_id="bedrock/gpt-oss-120b",
            environment_id="math:Orz57K",
            gepa_budget_mode="light",
            replication_seed=42,
            seed_prompt="Solve the problem step by step.",
            hostname="macbook-pro.local",
            schema_version=SCHEMA_VERSION,
        )
        save_experiment_run(db_path, run)
        loaded = load_experiment_run(db_path, "run-rt-001")

        assert loaded.experiment_run_id == run.experiment_run_id
        assert loaded.config_name == run.config_name
        assert loaded.config_hash == run.config_hash
        assert loaded.config_yaml == run.config_yaml
        assert loaded.operator == run.operator
        assert loaded.git_commit == "abc1234"
        assert loaded.git_branch == "feat/logging-v2"
        assert loaded.provider == run.provider
        assert loaded.task_model_id == run.task_model_id
        assert loaded.reflection_model_id == "bedrock/gpt-oss-120b"
        assert loaded.environment_id == run.environment_id
        assert loaded.gepa_budget_mode == "light"
        assert loaded.replication_seed == 42
        assert loaded.seed_prompt == "Solve the problem step by step."
        assert loaded.optimized_prompt is None
        assert loaded.status == "running"
        assert loaded.hostname == "macbook-pro.local"
        assert loaded.result_summary is None
        assert loaded.cost_summary is None
        assert loaded.schema_version == SCHEMA_VERSION
        assert abs((loaded.started_at - run.started_at).total_seconds()) < 0.01

    def test_save_duplicate_raises(self, db_path):
        run = _make_experiment_run(experiment_run_id="dup-run")
        save_experiment_run(db_path, run)
        with pytest.raises(ValueError, match="dup-run"):
            save_experiment_run(db_path, run)

    def test_load_nonexistent_raises(self, db_path):
        # Initialise the DB so it doesn't fail on missing file
        save_experiment_run(db_path, _make_experiment_run())
        with pytest.raises(KeyError, match="no-such-run"):
            load_experiment_run(db_path, "no-such-run")

    def test_episode_linked_to_experiment_run(self, db_path):
        """5.12: save experiment_run + episode with FK → verify linkage."""
        run = _make_experiment_run(experiment_run_id="run-fk-001")
        save_experiment_run(db_path, run)

        step = _make_step(reward=1.0, terminated=True)
        log = _make_log(steps=[step], environment_id="math:Orz57K")
        save_trajectory(db_path, log, experiment_run_id="run-fk-001")

        # Verify the FK was stored by querying the raw row
        from trajectory_aware_gym.storage.trajectory_db import get_connection

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT experiment_run_id FROM episodes WHERE run_id = ?",
            (log.run_id,),
        ).fetchone()
        assert row["experiment_run_id"] == "run-fk-001"

    def test_episode_without_experiment_run_fk(self, db_path):
        """Episodes can exist without an experiment_run_id (nullable FK)."""
        step = _make_step(reward=0.5, terminated=True)
        log = _make_log(steps=[step])
        save_trajectory(db_path, log)

        from trajectory_aware_gym.storage.trajectory_db import get_connection

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT experiment_run_id FROM episodes WHERE run_id = ?",
            (log.run_id,),
        ).fetchone()
        assert row["experiment_run_id"] is None

    def test_query_by_config_name(self, db_path):
        """5.13a: query_experiment_runs with config_name filter."""
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-cfg-1",
                config_name="orz57k-light",
            ),
        )
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-cfg-2",
                config_name="hotpotqa-medium",
            ),
        )
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-cfg-3",
                config_name="orz57k-light",
            ),
        )

        results = query_experiment_runs(db_path, config_name="orz57k-light")
        assert len(results) == 2
        assert all(r.config_name == "orz57k-light" for r in results)

    def test_query_by_operator(self, db_path):
        """5.13b: query_experiment_runs with operator filter."""
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-op-1",
                operator="alice",
            ),
        )
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-op-2",
                operator="bob",
            ),
        )

        results = query_experiment_runs(db_path, operator="alice")
        assert len(results) == 1
        assert results[0].operator == "alice"

    def test_query_by_provider(self, db_path):
        """5.13c: query_experiment_runs with provider filter."""
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-prov-1",
                provider="ollama",
            ),
        )
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-prov-2",
                provider="bedrock",
            ),
        )

        results = query_experiment_runs(db_path, provider="bedrock")
        assert len(results) == 1
        assert results[0].provider == "bedrock"

    def test_query_combined_filters(self, db_path):
        """5.13d: query with multiple filters ANDed together."""
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-combo-1",
                config_name="orz57k",
                operator="alice",
                provider="ollama",
            ),
        )
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-combo-2",
                config_name="orz57k",
                operator="bob",
                provider="ollama",
            ),
        )
        save_experiment_run(
            db_path,
            _make_experiment_run(
                experiment_run_id="q-combo-3",
                config_name="hotpotqa",
                operator="alice",
                provider="bedrock",
            ),
        )

        results = query_experiment_runs(
            db_path,
            config_name="orz57k",
            operator="alice",
        )
        assert len(results) == 1
        assert results[0].experiment_run_id == "q-combo-1"

    def test_query_no_filters_returns_all(self, db_path):
        save_experiment_run(db_path, _make_experiment_run(experiment_run_id="q-all-1"))
        save_experiment_run(db_path, _make_experiment_run(experiment_run_id="q-all-2"))

        results = query_experiment_runs(db_path)
        assert len(results) == 2

    def test_query_no_matches_returns_empty(self, db_path):
        save_experiment_run(db_path, _make_experiment_run(experiment_run_id="q-empty"))
        results = query_experiment_runs(db_path, operator="nonexistent")
        assert results == []

    def test_update_status_and_finished_at(self, db_path):
        """5.14a: update status and finished_at → reload → verify."""
        now = datetime.now(UTC)
        run = _make_experiment_run(experiment_run_id="upd-001", started_at=now)
        save_experiment_run(db_path, run)

        finished = now + timedelta(hours=1)
        update_experiment_run(
            db_path,
            "upd-001",
            status="completed",
            finished_at=finished,
        )

        loaded = load_experiment_run(db_path, "upd-001")
        assert loaded.status == "completed"
        assert loaded.finished_at is not None
        assert abs((loaded.finished_at - finished).total_seconds()) < 0.01

    def test_update_cost_and_result_summary(self, db_path):
        """5.14b: update cost_summary and result_summary dicts."""
        run = _make_experiment_run(experiment_run_id="upd-002")
        save_experiment_run(db_path, run)

        update_experiment_run(
            db_path,
            "upd-002",
            result_summary={"accuracy": 0.85, "total_episodes": 50},
            cost_summary={"total_usd": 1.23, "total_tokens": 500000},
        )

        loaded = load_experiment_run(db_path, "upd-002")
        assert loaded.result_summary == {"accuracy": 0.85, "total_episodes": 50}
        assert loaded.cost_summary == {"total_usd": 1.23, "total_tokens": 500000}

    def test_update_optimized_prompt(self, db_path):
        """5.14c: update optimized_prompt after GEPA completes."""
        run = _make_experiment_run(experiment_run_id="upd-003")
        save_experiment_run(db_path, run)

        update_experiment_run(
            db_path,
            "upd-003",
            optimized_prompt="Think carefully, use Python.",
        )

        loaded = load_experiment_run(db_path, "upd-003")
        assert loaded.optimized_prompt == "Think carefully, use Python."

    def test_update_disallowed_field_raises(self, db_path):
        run = _make_experiment_run(experiment_run_id="upd-bad")
        save_experiment_run(db_path, run)

        with pytest.raises(ValueError, match="Cannot update"):
            update_experiment_run(db_path, "upd-bad", config_name="hacked")

    def test_update_nonexistent_run_raises(self, db_path):
        # Ensure DB is initialised
        save_experiment_run(db_path, _make_experiment_run())
        with pytest.raises(KeyError, match="no-such-run"):
            update_experiment_run(db_path, "no-such-run", status="failed")
