"""Tests for SQLite-backed trajectory storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import (
    LLMCallMetadata,
    ToolCall,
    TrajectoryLog,
    TrajectoryLogger,
    TrajectoryStep,
    load_all_trajectories,
    load_trajectory,
)
from trajectory_aware_gym.config import ProjectPaths
from trajectory_aware_gym.storage.trajectory_db import (
    close_connection,
    episode_exists,
    load_trajectory_by_id,
    query_trajectories,
    save_tool_call_entry,
    save_trajectory,
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
        assert len(loaded.steps[1].llm_calls) == 2
        assert loaded.steps[1].llm_calls[0].cost_usd == 0.001

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
        assert rows[0][2] == "python_exec"


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
        with pytest.raises(FileNotFoundError):
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
