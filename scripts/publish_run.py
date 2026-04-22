"""Publish a completed experiment run to docs/07-results.

The production runner writes results to ``results/{config}/{timestamp}/{model}/replication_{seed}/``.
This script promotes a run into ``docs/07-results/{config}/{model}/{timestamp}/`` so finished
replications can be reviewed alongside the other capstone documentation.

Default behaviour: publish the most recent completed run across the production configs
(``hotpotqa-tool``, ``hotpotqa-notool``, ``orz57k-tool``, ``orz57k-notool``). Pass
``--run-dir`` to override with an explicit run, or ``--config`` to pick the most recent
run for a specific config.

A warning is emitted when the published run was optimised with ``gepa_budget.mode`` set
to anything other than ``medium``; the publish still proceeds so ablations can be
captured.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
DEFAULT_DEST_ROOT = REPO_ROOT / "docs" / "07-results"
PRODUCTION_CONFIGS = ("hotpotqa-tool", "hotpotqa-notool", "orz57k-tool", "orz57k-notool")
EXPECTED_BUDGET_MODE = "medium"

logger = logging.getLogger("publish_run")


@dataclass(frozen=True)
class RunLocation:
    config: str
    timestamp: str
    run_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--run-dir",
        type=Path,
        help="Publish this exact run directory (overrides --config and the default most-recent lookup).",
    )
    scope.add_argument(
        "--config",
        choices=PRODUCTION_CONFIGS,
        help="Publish the most recent run for this config (default: most recent across all production configs).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help=f"Destination root directory (default: {DEFAULT_DEST_ROOT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination folders if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without touching the filesystem.",
    )
    return parser.parse_args()


def _iter_timestamp_dirs(config_dir: Path) -> list[Path]:
    if not config_dir.is_dir():
        return []
    return sorted((p for p in config_dir.iterdir() if p.is_dir()), key=lambda p: p.name)


def _latest_run_for_config(config: str) -> RunLocation | None:
    config_dir = RESULTS_ROOT / config
    timestamp_dirs = _iter_timestamp_dirs(config_dir)
    if not timestamp_dirs:
        return None
    latest = timestamp_dirs[-1]
    return RunLocation(config=config, timestamp=latest.name, run_dir=latest)


def _latest_run_across_configs() -> RunLocation | None:
    candidates = [run for cfg in PRODUCTION_CONFIGS if (run := _latest_run_for_config(cfg))]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.timestamp)


def _resolve_run_from_args(args: argparse.Namespace) -> RunLocation:
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory not found: {run_dir}")
        try:
            rel = run_dir.relative_to(RESULTS_ROOT)
        except ValueError as exc:
            raise SystemExit(
                f"--run-dir must live under {RESULTS_ROOT.relative_to(REPO_ROOT)}; got {run_dir}"
            ) from exc
        if len(rel.parts) != 2:
            raise SystemExit(
                f"--run-dir must be shaped results/<config>/<timestamp>; got results/{rel}"
            )
        config, timestamp = rel.parts
        return RunLocation(config=config, timestamp=timestamp, run_dir=run_dir)

    if args.config is not None:
        location = _latest_run_for_config(args.config)
        if location is None:
            raise SystemExit(f"No runs found for config '{args.config}' under {RESULTS_ROOT}.")
        return location

    location = _latest_run_across_configs()
    if location is None:
        raise SystemExit(f"No production runs found under {RESULTS_ROOT}.")
    return location


def _discover_model_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and any(child.name.startswith("replication_") for child in path.iterdir())
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _detect_budget_mode(run_dir: Path, model_dirs: list[Path]) -> str | None:
    summary = _load_json(run_dir / "run_summary.json")
    if summary and isinstance(summary.get("gepa_budget_mode"), str):
        return summary["gepa_budget_mode"]
    for model_dir in model_dirs:
        for replication in sorted(model_dir.glob("replication_*")):
            metadata = _load_json(replication / "run_metadata.json")
            if metadata and isinstance(metadata.get("gepa_budget_mode"), str):
                return metadata["gepa_budget_mode"]
    return None


def _copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] would copy %s → %s", src, dst)
        return
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] would copy %s → %s", src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _publish_model(
    model_dir: Path,
    *,
    config: str,
    timestamp: str,
    run_dir: Path,
    dest_root: Path,
    dry_run: bool,
) -> Path:
    model_name = model_dir.name
    dest_dir = dest_root / config / model_name / timestamp

    if dest_dir.exists():
        if dry_run:
            logger.info("[dry-run] would remove existing %s", dest_dir)
        else:
            shutil.rmtree(dest_dir)

    if dry_run:
        logger.info("[dry-run] would create %s", dest_dir)
    else:
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        dest_dir.mkdir(parents=True)

    for replication_dir in sorted(model_dir.glob("replication_*")):
        _copy_tree(
            replication_dir,
            dest_dir / replication_dir.name,
            dry_run=dry_run,
        )

    run_summary = run_dir / "run_summary.json"
    if run_summary.exists():
        _copy_file(run_summary, dest_dir / "run_summary.json", dry_run=dry_run)

    gitkeep = dest_dir.parent / ".gitkeep"
    if gitkeep.exists():
        if dry_run:
            logger.info("[dry-run] would remove %s", gitkeep)
        else:
            gitkeep.unlink()

    return dest_dir


def publish(location: RunLocation, *, dest_root: Path, force: bool, dry_run: bool) -> list[Path]:
    model_dirs = _discover_model_dirs(location.run_dir)
    if not model_dirs:
        raise SystemExit(f"No model directories with replications found under {location.run_dir}.")

    budget_mode = _detect_budget_mode(location.run_dir, model_dirs)
    if budget_mode is None:
        logger.warning("Could not determine gepa_budget.mode for %s", location.run_dir)
    elif budget_mode != EXPECTED_BUDGET_MODE:
        logger.warning(
            "gepa_budget.mode is '%s' (expected '%s'). Publishing anyway.",
            budget_mode,
            EXPECTED_BUDGET_MODE,
        )

    planned_dests = [
        dest_root / location.config / model_dir.name / location.timestamp
        for model_dir in model_dirs
    ]
    existing = [dest for dest in planned_dests if dest.exists()]
    if existing and not force:
        for dest in existing:
            logger.warning("Destination already published: %s", dest.relative_to(REPO_ROOT))
        logger.warning(
            "Refusing to overwrite. Re-run with --force to replace the existing publish."
        )
        raise SystemExit(1)

    published: list[Path] = []
    for model_dir in model_dirs:
        dest_dir = _publish_model(
            model_dir,
            config=location.config,
            timestamp=location.timestamp,
            run_dir=location.run_dir,
            dest_root=dest_root,
            dry_run=dry_run,
        )
        published.append(dest_dir)
        logger.info(
            "Published %s → %s", model_dir.relative_to(REPO_ROOT), dest_dir.relative_to(REPO_ROOT)
        )
    return published


def _load_config_models(config: str) -> list[str]:
    config_path = EXPERIMENTS_ROOT / config / "config.yaml"
    if not config_path.exists():
        return []
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    task_models = payload.get("task_models") or []
    return [entry["name"] for entry in task_models if isinstance(entry, dict) and "name" in entry]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    location = _resolve_run_from_args(args)
    logger.info(
        "Publishing run results/%s/%s to %s",
        location.config,
        location.timestamp,
        args.dest.relative_to(REPO_ROOT)
        if args.dest.is_absolute() and args.dest.is_relative_to(REPO_ROOT)
        else args.dest,
    )

    published = publish(
        location,
        dest_root=args.dest.resolve(),
        force=args.force,
        dry_run=args.dry_run,
    )

    print(
        json.dumps(
            {
                "config": location.config,
                "timestamp": location.timestamp,
                "source_run_dir": str(location.run_dir.relative_to(REPO_ROOT)),
                "published": [str(p.relative_to(REPO_ROOT)) for p in published],
                "expected_models": _load_config_models(location.config),
                "dry_run": args.dry_run,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — surface unexpected errors cleanly for the CLI
        logger.error("Publish failed: %s", exc)
        sys.exit(1)
