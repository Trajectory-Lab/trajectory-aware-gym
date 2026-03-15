"""Tests for baseline experiment config setup and loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from trajectory_aware_gym.config import (
    BaselineExperimentConfig,
    load_baseline_config,
    load_baseline_configs,
)


def test_load_baseline_config_from_toml(tmp_path: Path) -> None:
    """Single baseline config TOML should load into validated model."""
    config_path = tmp_path / "baseline.toml"
    config_path.write_text(
        "\n".join(
            [
                'config_name = "grpo_math12k_baseline"',
                'algorithm = "grpo"',
                'environment = "math12k"',
                'environment_id = "math12k"',
                'output_subdir = "baseline/grpo/math12k"',
                "",
                "[model]",
                'provider = "bedrock"',
                'model_ref = "env:BEDROCK_QWEN3_4B"',
                "",
                "[runtime]",
                "max_steps = 8",
                "max_response_tokens = 4096",
                "temperature_train = 1.0",
                "temperature_eval = 0.0",
                "top_p = 1.0",
                "",
                "[budget]",
                "train_episodes = 5000",
                "eval_rollouts_per_task = 5",
                "num_replications = 3",
            ]
        ),
        encoding="utf-8",
    )

    config = load_baseline_config(config_path)

    assert isinstance(config, BaselineExperimentConfig)
    assert config.algorithm == "grpo"
    assert config.environment == "math12k"
    assert config.runtime.max_response_tokens == 4096


def test_load_baseline_configs_raises_for_empty_directory(tmp_path: Path) -> None:
    """Bulk loader should raise when no baseline TOML files are present."""
    with pytest.raises(FileNotFoundError, match="No baseline config files found"):
        load_baseline_configs(tmp_path)


def test_repository_baseline_configs_cover_target_matrix() -> None:
    """Repository configs should include GRPO/PPO for all target environments."""
    config_dir = Path(__file__).resolve().parents[2] / "experiments" / "configs" / "baselines"
    configs = load_baseline_configs(config_dir)

    assert len(configs) == 6

    actual_pairs = {
        (config.algorithm, config.environment)
        for config in configs.values()
    }
    expected_pairs = {
        ("grpo", "math12k"),
        ("grpo", "codecontest"),
        ("grpo", "hotpotqa"),
        ("ppo", "math12k"),
        ("ppo", "codecontest"),
        ("ppo", "hotpotqa"),
    }

    assert actual_pairs == expected_pairs
