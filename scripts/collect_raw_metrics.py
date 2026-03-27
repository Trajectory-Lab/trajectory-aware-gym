"""Collect raw cost, latency, and efficiency metrics from trajectory logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from trajectory_aware_gym.metrics import (
    EpisodeRawMetrics,
    collect_raw_metrics,
    extract_episode_raw_metrics,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract raw per-episode metrics from trajectory logs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing trajectory_*.json files",
    )
    parser.add_argument(
        "--glob",
        default="trajectory_*.json",
        help="Glob pattern used to locate trajectory logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/raw_metrics"),
        help="Directory where CSV and JSONL outputs are written",
    )
    parser.add_argument(
        "--output-prefix",
        default="phase3_raw_metrics",
        help="Output filename prefix",
    )
    return parser.parse_args()


def _write_csv(file_path: Path, rows: list[EpisodeRawMetrics]) -> None:
    """Write raw metric rows to CSV."""
    if not rows:
        field_names = list(EpisodeRawMetrics.model_fields.keys())
        with file_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=field_names).writeheader()
        return

    field_names = list(rows[0].model_dump(mode="json").keys())
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _write_jsonl(file_path: Path, rows: list[EpisodeRawMetrics]) -> None:
    """Write raw metric rows to JSONL."""
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")


def _summary(rows: list[EpisodeRawMetrics]) -> dict[str, float | int | None]:
    """Build compact summary for quick validation of collected metrics."""
    if not rows:
        return {
            "episodes": 0,
            "successes": 0,
            "mean_episode_latency_seconds": None,
            "total_llm_cost_usd": None,
            "total_tokens": None,
        }

    episodes = len(rows)
    successes = sum(1 for row in rows if row.success)
    mean_episode_latency_seconds = sum(row.episode_latency_seconds for row in rows) / episodes

    known_costs = [row.llm_cost_usd for row in rows if row.llm_cost_usd is not None]
    known_tokens = [row.total_tokens for row in rows if row.total_tokens is not None]

    return {
        "episodes": episodes,
        "successes": successes,
        "mean_episode_latency_seconds": round(mean_episode_latency_seconds, 6),
        "total_llm_cost_usd": round(sum(known_costs), 6) if known_costs else None,
        "total_tokens": int(sum(known_tokens)) if known_tokens else None,
    }


def _load_rows(args: argparse.Namespace) -> tuple[list[EpisodeRawMetrics], int]:
    """Load metrics from SQLite (preferred) or JSON files.

    Returns (rows, source_count) where source_count is the number of
    episodes/files found in the source.
    """
    db_path = args.input_dir / "trajectories.db"
    if db_path.exists():
        from trajectory_aware_gym.storage import load_all_trajectories

        trajectories = load_all_trajectories(db_path)
        return [extract_episode_raw_metrics(t) for t in trajectories], len(trajectories)

    log_paths = sorted(args.input_dir.glob(args.glob))
    return collect_raw_metrics(log_paths), len(log_paths)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    rows, source_count = _load_rows(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    jsonl_path = args.output_dir / f"{args.output_prefix}.jsonl"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"

    _write_csv(csv_path, rows)
    _write_jsonl(jsonl_path, rows)
    summary_payload = _summary(rows)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Found logs: {source_count}")
    print(f"Metric rows: {len(rows)}")
    print(f"CSV output: {csv_path}")
    print(f"JSONL output: {jsonl_path}")
    print(f"Summary output: {summary_path}")


if __name__ == "__main__":
    main()
