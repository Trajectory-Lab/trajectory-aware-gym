"""Verify baseline algorithm config setup for target environments."""

from __future__ import annotations

import argparse
from pathlib import Path

from trajectory_aware_gym.config import load_baseline_configs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate baseline config files for GRPO/PPO target environments."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/configs/baselines"),
        help="Directory containing baseline *.toml config files",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    configs = load_baseline_configs(args.config_dir)

    coverage: dict[tuple[str, str], Path] = {}
    for config_path, config in configs.items():
        coverage[(config.algorithm, config.environment)] = config_path

    required_algorithms = ["grpo", "ppo"]
    required_environments = ["math12k", "codecontest", "hotpotqa"]

    missing_pairs: list[str] = []
    for algorithm in required_algorithms:
        for environment in required_environments:
            if (algorithm, environment) not in coverage:
                missing_pairs.append(f"{algorithm}:{environment}")

    if missing_pairs:
        missing_text = ", ".join(missing_pairs)
        raise ValueError(f"Missing required baseline configs: {missing_text}")

    print(f"Validated baseline configs: {len(configs)}")
    for (algorithm, environment), config_path in sorted(coverage.items()):
        print(f"- {algorithm}:{environment} -> {config_path}")


if __name__ == "__main__":
    main()
