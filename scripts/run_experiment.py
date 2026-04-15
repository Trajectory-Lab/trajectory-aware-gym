"""Run production GEPA experiments with replication and cost tracking."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trajectory_aware_gym.config import settings
from trajectory_aware_gym.experiments.runner import (
    RunExperimentArgs,
    run_experiment,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run production GEPA experiments")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to ExperimentConfig YAML",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Optional override for GEPA max_metric_calls (mutually exclusive with --budget-mode)",
    )
    parser.add_argument(
        "--budget-mode",
        type=str,
        choices=["light", "medium", "heavy"],
        default=None,
        help="Override gepa_budget.mode from config (mutually exclusive with --max-metric-calls)",
    )
    parser.add_argument(
        "--seed-prompt",
        type=str,
        default=None,
        help=(
            "Override seed prompt for ad-hoc runs. "
            "When omitted, uses seed_prompt from the experiment YAML."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of task model names from config.task_models",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of replication seeds from config.seeds.replication_seeds",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Skip resuming an incomplete run; start a new one",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume a specific run by its timestamp (e.g. 20260408T150000Z)",
    )
    parser.add_argument(
        "--danger-purge",
        action="store_true",
        help="Delete ALL prior results for this config before running (requires confirmation)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory for experiment artifacts",
    )
    parser.add_argument(
        "--halt-on-budget-exceeded",
        action="store_true",
        help="Stop the run if cost exceeds config.cost_budget.effective_budget_usd",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()

    purge = args.danger_purge
    if purge:
        config_dir = args.results_root / args.config.stem
        print(
            f"WARNING: This will permanently delete ALL results under {config_dir}/\n"
            f"Type 'yes' to confirm: ",
            end="",
        )
        confirmation = input().strip()
        if confirmation != "yes":
            print("Aborted.")
            sys.exit(1)

    if args.max_metric_calls is not None and args.budget_mode is not None:
        print("Error: --max-metric-calls and --budget-mode are mutually exclusive.")
        sys.exit(1)

    run_experiment(
        RunExperimentArgs(
            config_path=args.config,
            max_metric_calls=args.max_metric_calls,
            budget_mode=args.budget_mode,
            seed_prompt_override=args.seed_prompt,
            models=tuple(args.models) if args.models else None,
            seeds=tuple(args.seeds) if args.seeds else None,
            fresh=args.fresh,
            purge=purge,
            resume=args.resume,
            results_root=args.results_root,
            halt_on_budget_exceeded=args.halt_on_budget_exceeded,
        )
    )


if __name__ == "__main__":
    main()
