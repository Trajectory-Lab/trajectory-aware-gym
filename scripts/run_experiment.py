"""Run production GEPA experiments with replication and cost tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

from trajectory_aware_gym.experiments.runner import (
    DEFAULT_SEED_PROMPT,
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
        help="Optional override for GEPA max_metric_calls",
    )
    parser.add_argument(
        "--seed-prompt",
        type=str,
        default=DEFAULT_SEED_PROMPT,
        help="Seed prompt to initialize GEPA",
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
        help="Remove existing results for this config.name before running",
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
    args = parse_args()
    run_experiment(
        RunExperimentArgs(
            config_path=args.config,
            max_metric_calls=args.max_metric_calls,
            seed_prompt=args.seed_prompt,
            models=tuple(args.models) if args.models else None,
            seeds=tuple(args.seeds) if args.seeds else None,
            fresh=args.fresh,
            results_root=args.results_root,
            halt_on_budget_exceeded=args.halt_on_budget_exceeded,
        )
    )


if __name__ == "__main__":
    main()
