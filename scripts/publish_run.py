"""Publish a completed experiment run to docs/07-results.

The production runner writes results to ``results/{config}/{timestamp}/{model}/replication_{seed}/``.
This script promotes a run into ``docs/07-results/{config}/{model}/{timestamp}/`` so finished
replications can be reviewed alongside the other capstone documentation, and regenerates a
top-level ``docs/07-results/summary.md`` aggregate table.

Default behaviour: publish the most recent completed run across the production configs
(``hotpotqa-tool``, ``hotpotqa-notool``, ``orz57k-tool``, ``orz57k-notool``). Pass
``--run-dir`` to override with an explicit run, ``--config`` to pick the most recent run for
a specific config, or ``--update-summary-only`` to rebuild the summary without copying.

Publish guards:
- Refuses to publish when the current branch is behind ``origin/development``.
  Pass ``--skip-branch-check`` for offline / CI runs.
- Refuses to publish a run that contains replication errors/failures.
- Refuses to publish a run whose replication ``config_snapshot.yaml`` drifts from the
  canonical per-config protocol (``val_size``, held-out ``eval_size``,
  ``tasks_per_minibatch``). There is no ``--force`` override — a config tweak must be
  reverted (or the run re-executed with the correct config) before it can be published.
- Warns and still publishes when ``gepa_budget.mode != "medium"`` (needed for ablations).
- When the destination already has a publish for the same ``{config}/{model}/{timestamp}``,
  the user must type ``yes`` to overwrite. Pass ``--force`` to skip the prompt in CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import subprocess  # noqa: S404 — calling git with fixed arg lists, no shell
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
DEFAULT_DEST_ROOT = REPO_ROOT / "docs" / "07-results"
PRODUCTION_CONFIGS = ("hotpotqa-tool", "hotpotqa-notool", "orz57k-tool", "orz57k-notool")
CONFIG_DISPLAY_NAMES = {
    "hotpotqa-tool": "HotpotQA (with tool)",
    "hotpotqa-notool": "HotpotQA (no tool)",
    "orz57k-tool": "Orz57K (with tool)",
    "orz57k-notool": "Orz57K (no tool)",
}
EXPECTED_BUDGET_MODE = "medium"
ALLOWED_SEEDS = frozenset({42, 123, 456, 789, 101112})
SUMMARY_FILENAME = "summary.md"
UPSTREAM_BRANCH = "development"
UPSTREAM_REMOTE = "origin"


@dataclass(frozen=True)
class ProtocolSpec:
    """Canonical protocol values that every publishable replication must match."""

    val_size: int
    eval_size: int
    eval_rationale: str
    tasks_per_minibatch: int = 3
    val_rationale: str = "GEM paper validation subsample"
    minibatch_rationale: str = "GEPA paper reflection_minibatch_size"


PROTOCOL_EXPECTATIONS: dict[str, ProtocolSpec] = {
    "hotpotqa-tool": ProtocolSpec(
        val_size=300,
        eval_size=512,
        eval_rationale="full axon-rl/search-eval hotpotqa split",
    ),
    "hotpotqa-notool": ProtocolSpec(
        val_size=300,
        eval_size=512,
        eval_rationale="full axon-rl/search-eval hotpotqa split",
    ),
    "orz57k-tool": ProtocolSpec(
        val_size=300,
        eval_size=500,
        eval_rationale="full MATH500 held-out set",
    ),
    "orz57k-notool": ProtocolSpec(
        val_size=300,
        eval_size=500,
        eval_rationale="full MATH500 held-out set",
    ),
}

logger = logging.getLogger("publish_run")


@dataclass(frozen=True)
class RunLocation:
    config: str
    timestamp: str
    run_dir: Path


@dataclass(frozen=True)
class ReplicationRecord:
    config: str
    model: str
    timestamp: str
    seed: int
    baseline_accuracy: float
    optimized_accuracy: float


def _parse_seed(replication_dir: Path) -> int | None:
    raw = replication_dir.name.removeprefix("replication_")
    try:
        return int(raw)
    except ValueError:
        return None


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed arg lists, no shell
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _check_branch_up_to_date() -> None:
    """Abort publish when origin/development has commits not reachable from HEAD.

    Fetches the upstream branch, then counts any commits that the current
    branch is missing. When git is unavailable, the remote can't be reached,
    or the ref doesn't exist we log a warning and proceed rather than block
    legitimate offline / CI runs. Pass ``--skip-branch-check`` to skip this
    check entirely.
    """
    fetch = _run_git(["fetch", UPSTREAM_REMOTE, UPSTREAM_BRANCH])
    if fetch.returncode != 0:
        logger.warning(
            "Could not fetch %s/%s (%s). Skipping branch-freshness check.",
            UPSTREAM_REMOTE,
            UPSTREAM_BRANCH,
            fetch.stderr.strip() or "unknown error",
        )
        return

    upstream_ref = f"{UPSTREAM_REMOTE}/{UPSTREAM_BRANCH}"
    count = _run_git(["rev-list", "--count", f"HEAD..{upstream_ref}"])
    if count.returncode != 0:
        logger.warning(
            "Could not compare HEAD against %s (%s). Skipping branch-freshness check.",
            upstream_ref,
            count.stderr.strip() or "unknown error",
        )
        return

    try:
        missing = int(count.stdout.strip())
    except ValueError:
        logger.warning("Unexpected rev-list output: %r", count.stdout)
        return

    if missing == 0:
        return

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip() or "<detached>"
    logger.error(
        "Refusing to publish: %r is behind %s by %d commit(s). "
        "Merge development first, or pass --skip-branch-check for offline/CI runs.",
        branch,
        upstream_ref,
        missing,
    )
    logger.error(
        "  git fetch %s %s && git merge %s", UPSTREAM_REMOTE, UPSTREAM_BRANCH, upstream_ref
    )
    raise SystemExit(1)


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
    scope.add_argument(
        "--update-summary-only",
        action="store_true",
        help="Skip copying; rebuild docs/07-results/summary.md from already-published runs.",
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
        help="Skip the interactive overwrite confirmation (for CI / non-interactive use).",
    )
    parser.add_argument(
        "--skip-branch-check",
        action="store_true",
        help=(
            f"Skip the check that {UPSTREAM_REMOTE}/{UPSTREAM_BRANCH} is fully "
            "merged into HEAD (for offline or CI runs without fetch access)."
        ),
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


def _load_config_snapshot(replication_dir: Path) -> dict[str, Any] | None:
    path = replication_dir / "config_snapshot.yaml"
    if not path.exists():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _protocol_issues(config: str, snapshot: dict[str, Any] | None) -> list[str]:
    """Check a replication's config snapshot against the canonical protocol.

    Returns an empty list when the replication matches, otherwise a list of
    human-readable explanations naming each drifted field, the expected value,
    and why that value is canonical.
    """
    expected = PROTOCOL_EXPECTATIONS.get(config)
    if expected is None:
        return []
    if snapshot is None:
        return [f"config_snapshot.yaml missing; cannot verify {config} protocol"]

    env = snapshot.get("environment") or {}
    dataset = env.get("dataset") or {}
    gepa_budget = snapshot.get("gepa_budget") or {}

    checks: list[tuple[str, Any, int, str]] = [
        ("val_size", env.get("val_size"), expected.val_size, expected.val_rationale),
        ("eval_size", dataset.get("eval_size"), expected.eval_size, expected.eval_rationale),
        (
            "tasks_per_minibatch",
            gepa_budget.get("tasks_per_minibatch"),
            expected.tasks_per_minibatch,
            expected.minibatch_rationale,
        ),
    ]
    issues: list[str] = []
    for field, actual, required, rationale in checks:
        if actual != required:
            issues.append(f"{field}={actual!r} but {config} requires {required} ({rationale})")
    return issues


def _scan_protocol_compliance(config: str, model_dirs: list[Path]) -> list[tuple[Path, list[str]]]:
    """Return [(replication_dir, issues), ...] for every replication that drifted."""
    faulty: list[tuple[Path, list[str]]] = []
    for model_dir in model_dirs:
        for replication_dir in sorted(model_dir.glob("replication_*")):
            snapshot = _load_config_snapshot(replication_dir)
            issues = _protocol_issues(config, snapshot)
            if issues:
                faulty.append((replication_dir, issues))
    return faulty


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


def _replication_health_issues(metadata: dict[str, Any]) -> list[str]:
    """Return a list of error strings for a replication; empty list means clean."""
    issues: list[str] = []
    status = metadata.get("status")
    if status != "completed":
        issues.append(f"status={status!r} (expected 'completed')")

    for phase in ("baseline_eval", "eval"):
        phase_data = metadata.get(phase) or {}
        for key in ("failed", "timed_out", "metrics_unavailable"):
            value = phase_data.get(key, 0) or 0
            if value:
                issues.append(f"{phase}.{key}={value}")

    logging_summary = metadata.get("logging_summary") or {}
    if logging_summary.get("status") not in (None, "complete"):
        issues.append(f"logging_summary.status={logging_summary.get('status')!r}")
    for key in (
        "trajectory_failed_episodes",
        "metrics_unavailable_episodes",
        "numeric_anomaly_count",
    ):
        value = logging_summary.get(key, 0) or 0
        if value:
            issues.append(f"logging_summary.{key}={value}")
    return issues


def _eval_manifest_count(replication_dir: Path) -> int:
    manifest = replication_dir / "eval_failure_manifest.jsonl"
    if not manifest.exists():
        return 0
    try:
        with manifest.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _scan_replication_health(model_dirs: list[Path]) -> list[tuple[Path, list[str]]]:
    """Return [(replication_dir, issues), ...] for every replication with a non-empty issue list."""
    faulty: list[tuple[Path, list[str]]] = []
    for model_dir in model_dirs:
        for replication_dir in sorted(model_dir.glob("replication_*")):
            metadata = _load_json(replication_dir / "run_metadata.json")
            if metadata is None:
                faulty.append((replication_dir, ["run_metadata.json missing or unreadable"]))
                continue
            issues = _replication_health_issues(metadata)
            manifest_failures = _eval_manifest_count(replication_dir)
            if manifest_failures:
                issues.append(f"eval_failure_manifest.jsonl has {manifest_failures} rows")
            if issues:
                faulty.append((replication_dir, issues))
    return faulty


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


def _scan_source_seeds(model_dirs: list[Path]) -> list[tuple[Path, int | None]]:
    """Return [(replication_dir, seed or None), ...] for every source replication."""
    return [
        (replication_dir, _parse_seed(replication_dir))
        for model_dir in model_dirs
        for replication_dir in sorted(model_dir.glob("replication_*"))
    ]


def _find_seed_conflicts(
    location: RunLocation,
    model_dirs: list[Path],
    dest_root: Path,
) -> dict[tuple[str, int], list[Path]]:
    """Map (model_name, seed) → existing destination replication dirs for that seed."""
    conflicts: dict[tuple[str, int], list[Path]] = {}
    for model_dir in model_dirs:
        model_root = dest_root / location.config / model_dir.name
        if not model_root.is_dir():
            continue
        for replication_dir in sorted(model_dir.glob("replication_*")):
            seed = _parse_seed(replication_dir)
            if seed is None:
                continue
            matches = sorted(model_root.glob(f"*/replication_{seed}"))
            if matches:
                conflicts[(model_dir.name, seed)] = matches
    return conflicts


def _confirm_overwrite(
    conflicts: dict[tuple[str, int], list[Path]], *, force: bool, dry_run: bool
) -> None:
    for (model, seed), paths in conflicts.items():
        for path in paths:
            logger.warning(
                "Seed %d already published for %s: %s",
                seed,
                model,
                path.relative_to(REPO_ROOT),
            )
    if force:
        logger.info("--force set, skipping overwrite confirmation.")
        return
    if dry_run:
        logger.info("[dry-run] would prompt to overwrite before proceeding.")
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            "Seed conflicts found and stdin is not a TTY. "
            "Re-run with --force to overwrite non-interactively."
        )
    prompt = "Overwrite the existing seed directories above? Type 'yes' to continue: "
    try:
        answer = input(prompt).strip().lower()
    except EOFError as exc:
        raise SystemExit("Aborted: no input received.") from exc
    if answer != "yes":
        raise SystemExit("Aborted by user.")


def _remove_conflicting_paths(
    conflicts: dict[tuple[str, int], list[Path]], *, dry_run: bool
) -> None:
    for paths in conflicts.values():
        for path in paths:
            if dry_run:
                logger.info("[dry-run] would remove existing %s", path.relative_to(REPO_ROOT))
                continue
            shutil.rmtree(path)
            parent = path.parent
            if not parent.exists():
                continue
            remaining_replications = [
                p for p in parent.iterdir() if p.name.startswith("replication_")
            ]
            if not remaining_replications:
                shutil.rmtree(parent)


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

    if dry_run:
        logger.info("[dry-run] would create %s", dest_dir)
    else:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
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

    protocol_faulty = _scan_protocol_compliance(location.config, model_dirs)
    if protocol_faulty:
        logger.error(
            "Refusing to publish: replication config drifted from the %s protocol "
            "(no --force override).",
            location.config,
        )
        for replication_dir, issues in protocol_faulty:
            logger.error("  %s", replication_dir.relative_to(REPO_ROOT))
            for issue in issues:
                logger.error("    - %s", issue)
        raise SystemExit(1)

    faulty = _scan_replication_health(model_dirs)
    if faulty:
        logger.error("Refusing to publish: run has replication errors/failures.")
        for replication_dir, issues in faulty:
            logger.error("  %s", replication_dir.relative_to(REPO_ROOT))
            for issue in issues:
                logger.error("    - %s", issue)
        raise SystemExit(1)

    source_seeds = _scan_source_seeds(model_dirs)
    disallowed = [
        (replication_dir, seed)
        for replication_dir, seed in source_seeds
        if seed is None or seed not in ALLOWED_SEEDS
    ]
    if disallowed:
        logger.error(
            "Refusing to publish: replication(s) use seeds outside %s.",
            sorted(ALLOWED_SEEDS),
        )
        for replication_dir, seed in disallowed:
            logger.error("  %s (seed=%s)", replication_dir.relative_to(REPO_ROOT), seed)
        raise SystemExit(1)

    budget_mode = _detect_budget_mode(location.run_dir, model_dirs)
    if budget_mode is None:
        logger.warning("Could not determine gepa_budget.mode for %s", location.run_dir)
    elif budget_mode != EXPECTED_BUDGET_MODE:
        logger.warning(
            "gepa_budget.mode is '%s' (expected '%s'). Publishing anyway.",
            budget_mode,
            EXPECTED_BUDGET_MODE,
        )

    conflicts = _find_seed_conflicts(location, model_dirs, dest_root)
    if conflicts:
        _confirm_overwrite(conflicts, force=force, dry_run=dry_run)
        _remove_conflicting_paths(conflicts, dry_run=dry_run)

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


def _collect_eligible_records(dest_root: Path) -> list[ReplicationRecord]:
    records: list[ReplicationRecord] = []
    for config in PRODUCTION_CONFIGS:
        config_root = dest_root / config
        if not config_root.is_dir():
            continue
        for model_dir in sorted(p for p in config_root.iterdir() if p.is_dir()):
            for timestamp_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                for replication_dir in sorted(timestamp_dir.glob("replication_*")):
                    record = _build_replication_record(
                        config, model_dir.name, timestamp_dir.name, replication_dir
                    )
                    if record is not None:
                        records.append(record)
    return records


def _build_replication_record(
    config: str, model: str, timestamp: str, replication_dir: Path
) -> ReplicationRecord | None:
    metadata = _load_json(replication_dir / "run_metadata.json")
    if metadata is None:
        return None
    if metadata.get("gepa_budget_mode") != EXPECTED_BUDGET_MODE:
        return None
    if _replication_health_issues(metadata):
        return None
    if _eval_manifest_count(replication_dir):
        return None
    if _protocol_issues(config, _load_config_snapshot(replication_dir)):
        return None

    baseline = (metadata.get("baseline_eval") or {}).get("accuracy")
    optimized = (metadata.get("eval") or {}).get("accuracy")
    if not isinstance(baseline, int | float) or not isinstance(optimized, int | float):
        return None

    seed = _parse_seed(replication_dir)
    if seed is None or seed not in ALLOWED_SEEDS:
        return None
    return ReplicationRecord(
        config=config,
        model=model,
        timestamp=timestamp,
        seed=seed,
        baseline_accuracy=float(baseline),
        optimized_accuracy=float(optimized),
    )


def _format_aggregate(values: list[float]) -> str:
    if not values:
        return "—"
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:.3f}"
    stdev = statistics.stdev(values)
    return f"{mean:.3f} ± {stdev:.3f}"


def _render_aligned_table(
    headers: list[str], aligns: list[str], rows: list[list[str]]
) -> list[str]:
    """Render a GFM table with cells space-padded so columns line up in raw source."""
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt_cell(cell: str, width: int, align: str) -> str:
        return cell.rjust(width) if align == "right" else cell.ljust(width)

    def _fmt_row(cells: list[str]) -> str:
        padded = [_fmt_cell(c, w, a) for c, w, a in zip(cells, widths, aligns, strict=True)]
        return "| " + " | ".join(padded) + " |"

    def _fmt_sep() -> str:
        parts = []
        for width, align in zip(widths, aligns, strict=True):
            inner = width + 2
            if align == "right":
                parts.append("-" * (inner - 1) + ":")
            else:
                parts.append(":" + "-" * (inner - 1))
        return "|" + "|".join(parts) + "|"

    return [_fmt_row(headers), _fmt_sep(), *(_fmt_row(row) for row in rows)]


def _render_summary(dest_root: Path, records: list[ReplicationRecord]) -> str:
    by_config: dict[str, dict[str, list[ReplicationRecord]]] = {
        cfg: {model: [] for model in _load_config_models(cfg)} for cfg in PRODUCTION_CONFIGS
    }
    for record in records:
        by_config.setdefault(record.config, {}).setdefault(record.model, []).append(record)

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%MZ")
    lines: list[str] = [
        "# Results Summary",
        "",
        "_Auto-generated by `poe publish`. Do not edit by hand — re-run "
        "`poe publish --update-summary-only` to refresh._",
        "",
        f"_Last updated: {now}. Only replications with "
        f"`gepa_budget.mode = {EXPECTED_BUDGET_MODE}` and no errors are counted._",
        "",
    ]

    for config in PRODUCTION_CONFIGS:
        display = CONFIG_DISPLAY_NAMES.get(config, config)
        lines.append(f"## {display}  `{config}`")
        lines.append("")
        models = by_config.get(config) or {}
        if not models:
            lines.extend(["_No models configured._", ""])
            continue
        headers = ["Model", "Runs", "Seeds", "Baseline accuracy", "Optimized accuracy"]
        aligns = ["left", "right", "left", "right", "right"]
        rows: list[list[str]] = []
        for model in _load_config_models(config):
            entries = models.get(model, [])
            runs = len(entries)
            baseline_values = [r.baseline_accuracy for r in entries]
            optimized_values = [r.optimized_accuracy for r in entries]
            runs_cell = str(runs) if runs else "—"
            seeds_cell = (
                ", ".join(str(s) for s in sorted({r.seed for r in entries})) if entries else "—"
            )
            rows.append(
                [
                    model,
                    runs_cell,
                    seeds_cell,
                    _format_aggregate(baseline_values),
                    _format_aggregate(optimized_values),
                ]
            )
        lines.extend(_render_aligned_table(headers, aligns, rows))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_summary(dest_root: Path, content: str, *, dry_run: bool) -> Path:
    path = dest_root / SUMMARY_FILENAME
    if dry_run:
        logger.info(
            "[dry-run] would write %s (%d bytes)", path.relative_to(REPO_ROOT), len(content)
        )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", path.relative_to(REPO_ROOT))
    return path


def update_summary(dest_root: Path, *, dry_run: bool) -> tuple[Path, list[ReplicationRecord]]:
    records = _collect_eligible_records(dest_root)
    content = _render_summary(dest_root, records)
    path = _write_summary(dest_root, content, dry_run=dry_run)
    return path, records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    dest_root = args.dest.resolve()

    if args.skip_branch_check:
        logger.warning(
            "--skip-branch-check set; not verifying HEAD is up to date with development."
        )
    else:
        _check_branch_up_to_date()

    if args.update_summary_only:
        summary_path, records = update_summary(dest_root, dry_run=args.dry_run)
        print(
            json.dumps(
                {
                    "mode": "update-summary-only",
                    "summary_path": str(summary_path.relative_to(REPO_ROOT)),
                    "eligible_replications": len(records),
                    "dry_run": args.dry_run,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

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
        dest_root=dest_root,
        force=args.force,
        dry_run=args.dry_run,
    )
    summary_path, records = update_summary(dest_root, dry_run=args.dry_run)

    print(
        json.dumps(
            {
                "config": location.config,
                "timestamp": location.timestamp,
                "source_run_dir": str(location.run_dir.relative_to(REPO_ROOT)),
                "published": [str(p.relative_to(REPO_ROOT)) for p in published],
                "expected_models": _load_config_models(location.config),
                "summary_path": str(summary_path.relative_to(REPO_ROOT)),
                "eligible_replications": len(records),
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
