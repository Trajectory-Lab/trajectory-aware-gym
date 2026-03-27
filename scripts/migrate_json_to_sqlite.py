"""Migrate existing trajectory JSON files and tool_calls.jsonl into SQLite.

Usage::

    uv run python scripts/migrate_json_to_sqlite.py
    uv run python scripts/migrate_json_to_sqlite.py --input-dir logs --db logs/trajectories.db
    uv run python scripts/migrate_json_to_sqlite.py --clean   # delete source files after verified migration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog
from trajectory_aware_gym.storage.trajectory_db import (
    episode_exists,
    load_trajectory_by_id,
    save_tool_call_entry,
    save_trajectory,
)


def migrate_trajectories(input_dir: Path, db_path: Path) -> tuple[int, int, list[tuple[Path, str]]]:
    """Ingest trajectory_*.json files into SQLite.

    Returns (migrated_count, skipped_count, migrated_files) where
    *migrated_files* is a list of ``(path, run_id)`` pairs for files
    that are now verified in the database.
    """
    migrated = 0
    skipped = 0
    migrated_files: list[tuple[Path, str]] = []

    for path in sorted(input_dir.glob("trajectory_*.json")):
        try:
            log = TrajectoryLog.model_validate_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValidationError) as exc:
            print(f"  SKIP (corrupt): {path.name} — {exc}")
            skipped += 1
            continue

        if episode_exists(db_path, log.run_id):
            # Already in DB — safe to clean up.
            migrated_files.append((path, log.run_id))
            skipped += 1
            continue

        save_trajectory(db_path, log)
        migrated_files.append((path, log.run_id))
        migrated += 1

    return migrated, skipped, migrated_files


def migrate_tool_calls(jsonl_path: Path, db_path: Path) -> int:
    """Ingest tool_calls.jsonl entries into SQLite.

    Returns the number of entries migrated.
    """
    if not jsonl_path.exists():
        return 0

    count = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        save_tool_call_entry(
            db_path,
            timestamp=entry.get("timestamp", ""),
            tool=entry.get("tool", ""),
            args=entry.get("args", {}),
            result=entry.get("result", {}),
        )
        count += 1

    return count


def verify_and_clean(
    db_path: Path,
    migrated_files: list[tuple[Path, str]],
    jsonl_path: Path,
) -> tuple[int, int]:
    """Verify each migrated file exists in the DB, then delete the source.

    Returns (deleted_count, failed_count).
    """
    deleted = 0
    failed = 0

    for path, run_id in migrated_files:
        try:
            load_trajectory_by_id(db_path, run_id)
        except FileNotFoundError:
            print(f"  KEEP (not in DB): {path.name}")
            failed += 1
            continue
        path.unlink()
        deleted += 1

    if jsonl_path.exists():
        jsonl_path.unlink()
        print(f"  Deleted: {jsonl_path.name}")

    return deleted, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate JSON trajectory logs to SQLite")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing trajectory_*.json and tool_calls.jsonl",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite database path (default: <input-dir>/trajectories.db)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete source JSON/JSONL files after verifying they exist in the database",
    )
    args = parser.parse_args()

    db_path = args.db or (args.input_dir / "trajectories.db")

    print(f"Source:      {args.input_dir}")
    print(f"Destination: {db_path}\n")

    migrated, skipped, migrated_files = migrate_trajectories(args.input_dir, db_path)
    print(f"\nTrajectories: {migrated} migrated, {skipped} skipped")

    jsonl_path = args.input_dir / "tool_calls.jsonl"
    tool_count = migrate_tool_calls(jsonl_path, db_path)
    print(f"Tool calls:   {tool_count} migrated")

    if args.clean and migrated_files:
        print(f"\nVerifying and cleaning {len(migrated_files)} source files...")
        deleted, failed = verify_and_clean(db_path, migrated_files, jsonl_path)
        print(f"Cleaned: {deleted} deleted, {failed} kept (verification failed)")

    print(f"\nDone. Database at: {db_path}")


if __name__ == "__main__":
    main()
