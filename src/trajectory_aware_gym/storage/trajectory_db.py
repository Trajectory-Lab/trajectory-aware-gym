"""SQLite-backed storage for trajectory logs and tool call records.

Replaces per-episode JSON files and the append-only tool_calls.jsonl with a
single ``trajectories.db`` database.  WAL mode is enabled for concurrent
read access during writes.

Threading model: each thread gets its own connection via ``threading.local()``.
WAL mode handles concurrent file access across threads/processes at the OS level.

Public API
----------
- ``get_connection``           -- open (or reuse) a WAL-mode connection
- ``save_trajectory``          -- persist one TrajectoryLog atomically
- ``load_trajectory_by_id``    -- fetch a single episode by run_id
- ``load_all_trajectories``    -- fetch all episodes, ordered by started_at
- ``query_trajectories``       -- fetch episodes with optional SQL filters
- ``save_tool_call_entry``     -- append one tool invocation record
- ``episode_exists``           -- check if a run_id is already stored
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trajectory_aware_gym.adapters.trajectory_logger import (
    LLMCallMetadata,
    ToolCall,
    TrajectoryLog,
    TrajectoryStep,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS episodes (
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

CREATE TABLE IF NOT EXISTS steps (
    run_id      TEXT    NOT NULL REFERENCES episodes(run_id),
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

CREATE TABLE IF NOT EXISTS llm_calls (
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

CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL,
    step_index  INTEGER NOT NULL,
    tool_name   TEXT    NOT NULL,
    tool_input  TEXT    NOT NULL,
    tool_output TEXT    NOT NULL,
    success     INTEGER NOT NULL,
    duration_ms REAL,
    FOREIGN KEY (run_id, step_index) REFERENCES steps(run_id, step_index)
);

CREATE TABLE IF NOT EXISTS tool_call_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    tool      TEXT NOT NULL,
    args      TEXT NOT NULL,
    result    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_env     ON episodes(environment_id);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(episode_outcome);
CREATE INDEX IF NOT EXISTS idx_steps_run        ON steps(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_step   ON llm_calls(run_id, step_index);
CREATE INDEX IF NOT EXISTS idx_tool_calls_step  ON tool_calls(run_id, step_index);
"""

_SQLITE_CONNECT_TIMEOUT_SECONDS = 10

# ---------------------------------------------------------------------------
# Connection management (one connection per thread via threading.local)
# ---------------------------------------------------------------------------

_local = threading.local()
_initialized_dbs: set[str] = set()
_initialized_dbs_lock = threading.Lock()


def _thread_connections() -> dict[str, sqlite3.Connection]:
    """Return the per-thread connection cache, initializing it if needed."""
    if not hasattr(_local, "connections"):
        _local.connections = {}
    return _local.connections


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Return a WAL-mode connection for the current thread.

    Creates the database schema on first access per db_path (not per thread).
    executescript() uses sqlite3_exec() internally and does not honour the
    connection busy-timeout, so running it from multiple threads against the
    same file causes spurious "database is locked" errors even under WAL mode.
    Tracking initialised paths ensures DDL runs exactly once.
    """
    key = str(db_path.resolve())
    connections = _thread_connections()
    if key in connections:
        return connections[key]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=_SQLITE_CONNECT_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    with _initialized_dbs_lock:
        if key not in _initialized_dbs:
            # journal_mode is persistent on the db file — set once during init.
            # Setting it outside the lock races with executescript's EXCLUSIVE
            # lock on slow CI runners.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA_SQL)
            _initialized_dbs.add(key)
    connections[key] = conn
    return conn


def close_connection(db_path: Path) -> None:
    """Close and remove the current thread's cached connection (useful in tests)."""
    key = str(db_path.resolve())
    conn = _thread_connections().pop(key, None)
    if conn is not None:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dt_to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _iso_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def save_trajectory(db_path: Path, log: TrajectoryLog) -> None:
    """Insert one episode atomically (single transaction).

    Raises ``ValueError`` if a trajectory with the same ``run_id`` already exists.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO episodes
                    (run_id, schema_version, environment_id, seed, system_prompt,
                     started_at, finished_at, initial_observation, initial_info,
                     total_reward, episode_outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.run_id,
                    log.schema_version,
                    log.environment_id,
                    log.seed,
                    log.system_prompt,
                    _dt_to_iso(log.started_at),
                    _dt_to_iso(log.finished_at),
                    log.initial_observation,
                    json.dumps(log.initial_info),
                    log.total_reward,
                    log.episode_outcome,
                ),
            )

            for step in log.steps:
                conn.execute(
                    """
                    INSERT INTO steps
                        (run_id, step_index, action, observation, reward,
                         terminated, truncated, info, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log.run_id,
                        step.step_index,
                        step.action,
                        step.observation,
                        step.reward,
                        int(step.terminated),
                        int(step.truncated),
                        json.dumps(step.info),
                        _dt_to_iso(step.timestamp) if step.timestamp else None,
                    ),
                )

                if step.tool_calls:
                    conn.executemany(
                        """
                        INSERT INTO tool_calls
                            (run_id, step_index, tool_name, tool_input,
                             tool_output, success, duration_ms)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                log.run_id,
                                step.step_index,
                                tc.tool_name,
                                tc.tool_input,
                                tc.tool_output,
                                int(tc.success),
                                tc.duration_ms,
                            )
                            for tc in step.tool_calls
                        ],
                    )

                if step.llm_calls:
                    conn.executemany(
                        """
                        INSERT INTO llm_calls
                            (run_id, step_index, model_id, prompt_tokens,
                             completion_tokens, total_tokens, cost_usd, latency_ms)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                log.run_id,
                                step.step_index,
                                lc.model_id,
                                lc.prompt_tokens,
                                lc.completion_tokens,
                                lc.total_tokens,
                                lc.cost_usd,
                                lc.latency_ms,
                            )
                            for lc in step.llm_calls
                        ],
                    )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Episode {log.run_id!r} already exists in {db_path}") from exc


def save_tool_call_entry(
    db_path: Path,
    *,
    timestamp: str,
    tool: str,
    args: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Append one tool invocation to the ``tool_call_log`` table."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "INSERT INTO tool_call_log (timestamp, tool, args, result) VALUES (?, ?, ?, ?)",
            (timestamp, tool, json.dumps(args), json.dumps(result)),
        )


def episode_exists(db_path: Path, run_id: str) -> bool:
    """Return True if *run_id* is already stored."""
    conn = get_connection(db_path)
    row = conn.execute("SELECT 1 FROM episodes WHERE run_id = ?", (run_id,)).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def _build_trajectory(
    episode_row: sqlite3.Row,
    step_rows: list[sqlite3.Row],
    tool_call_rows: list[sqlite3.Row],
    llm_call_rows: list[sqlite3.Row],
) -> TrajectoryLog:
    """Reconstruct a TrajectoryLog from sqlite3.Row objects."""
    tc_by_step: dict[int, list[ToolCall]] = {}
    for row in tool_call_rows:
        idx = row["step_index"]
        tc_by_step.setdefault(idx, []).append(
            ToolCall(
                tool_name=row["tool_name"],
                tool_input=row["tool_input"],
                tool_output=row["tool_output"],
                success=bool(row["success"]),
                duration_ms=row["duration_ms"],
            )
        )

    lc_by_step: dict[int, list[LLMCallMetadata]] = {}
    for row in llm_call_rows:
        idx = row["step_index"]
        lc_by_step.setdefault(idx, []).append(
            LLMCallMetadata(
                model_id=row["model_id"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
                cost_usd=row["cost_usd"],
                latency_ms=row["latency_ms"],
            )
        )

    steps: list[TrajectoryStep] = []
    for srow in step_rows:
        si = srow["step_index"]
        steps.append(
            TrajectoryStep(
                step_index=si,
                action=srow["action"],
                observation=srow["observation"],
                reward=srow["reward"],
                terminated=bool(srow["terminated"]),
                truncated=bool(srow["truncated"]),
                info=json.loads(srow["info"]),
                timestamp=_iso_to_dt(srow["timestamp"]) if srow["timestamp"] else None,
                tool_calls=tc_by_step.get(si, []),
                llm_calls=lc_by_step.get(si, []),
            )
        )

    return TrajectoryLog(
        schema_version=episode_row["schema_version"],
        run_id=episode_row["run_id"],
        environment_id=episode_row["environment_id"],
        seed=episode_row["seed"],
        system_prompt=episode_row["system_prompt"],
        started_at=_iso_to_dt(episode_row["started_at"]),
        finished_at=_iso_to_dt(episode_row["finished_at"]),
        initial_observation=episode_row["initial_observation"],
        initial_info=json.loads(episode_row["initial_info"]),
        steps=steps,
        total_reward=episode_row["total_reward"],
        episode_outcome=episode_row["episode_outcome"],
    )


def _load_trajectories_for_episodes(
    conn: sqlite3.Connection,
    episode_rows: list[sqlite3.Row],
) -> list[TrajectoryLog]:
    """Fetch all child rows for the given episode rows and reconstruct TrajectoryLogs."""
    if not episode_rows:
        return []

    run_ids = [r["run_id"] for r in episode_rows]
    # placeholders are "?,?,?..." computed from len(run_ids) — no user input interpolated
    placeholders = ",".join("?" for _ in run_ids)

    step_rows = conn.execute(
        f"SELECT * FROM steps WHERE run_id IN ({placeholders}) ORDER BY run_id, step_index",  # noqa: S608  # nosec B608
        run_ids,
    ).fetchall()
    tc_rows = conn.execute(
        f"SELECT * FROM tool_calls WHERE run_id IN ({placeholders}) ORDER BY run_id, step_index",  # noqa: S608  # nosec B608
        run_ids,
    ).fetchall()
    lc_rows = conn.execute(
        f"SELECT * FROM llm_calls WHERE run_id IN ({placeholders}) ORDER BY run_id, step_index",  # noqa: S608  # nosec B608
        run_ids,
    ).fetchall()

    steps_by_run: dict[str, list[sqlite3.Row]] = {rid: [] for rid in run_ids}
    for row in step_rows:
        steps_by_run[row["run_id"]].append(row)
    tc_by_run: dict[str, list[sqlite3.Row]] = {rid: [] for rid in run_ids}
    for row in tc_rows:
        tc_by_run[row["run_id"]].append(row)
    lc_by_run: dict[str, list[sqlite3.Row]] = {rid: [] for rid in run_ids}
    for row in lc_rows:
        lc_by_run[row["run_id"]].append(row)

    return [
        _build_trajectory(
            ep, steps_by_run[ep["run_id"]], tc_by_run[ep["run_id"]], lc_by_run[ep["run_id"]]
        )
        for ep in episode_rows
    ]


def load_trajectory_by_id(db_path: Path, run_id: str) -> TrajectoryLog:
    """Load a single trajectory by its ``run_id``.

    Raises ``KeyError`` if no episode with that ``run_id`` exists.
    """
    conn = get_connection(db_path)
    episode_row = conn.execute("SELECT * FROM episodes WHERE run_id = ?", (run_id,)).fetchone()
    if episode_row is None:
        raise KeyError(f"No episode with run_id={run_id!r} in {db_path}")

    step_rows = conn.execute(
        "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", (run_id,)
    ).fetchall()
    tc_rows = conn.execute(
        "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY step_index", (run_id,)
    ).fetchall()
    lc_rows = conn.execute(
        "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY step_index", (run_id,)
    ).fetchall()

    return _build_trajectory(episode_row, step_rows, tc_rows, lc_rows)


def load_all_trajectories(db_path: Path) -> list[TrajectoryLog]:
    """Load every stored trajectory, ordered by ``started_at``."""
    conn = get_connection(db_path)
    episode_rows = conn.execute("SELECT * FROM episodes ORDER BY started_at").fetchall()
    return _load_trajectories_for_episodes(conn, episode_rows)


def query_trajectories(
    db_path: Path,
    *,
    outcome: str | None = None,
    environment_id: str | None = None,
) -> list[TrajectoryLog]:
    """Load trajectories with optional SQL-level filtering.

    Both ``outcome`` and ``environment_id`` are exact-match filters combined with AND.
    Pass neither to return all trajectories (equivalent to ``load_all_trajectories``).
    """
    conn = get_connection(db_path)

    clauses: list[str] = []
    params: list[str] = []
    if outcome is not None:
        clauses.append("episode_outcome = ?")
        params.append(outcome)
    if environment_id is not None:
        clauses.append("environment_id = ?")
        params.append(environment_id)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    episode_rows = conn.execute(
        f"SELECT * FROM episodes{where} ORDER BY started_at",  # noqa: S608  # nosec B608 — clauses contain only hardcoded literals; values are ? params
        params,
    ).fetchall()
    return _load_trajectories_for_episodes(conn, episode_rows)
