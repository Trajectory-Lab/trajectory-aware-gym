"""Tests for scripts/verify_baseline_setup.py."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_baseline_setup.py"


@pytest.fixture
def script_module():
    """Load the setup verification script as a module for testing."""
    spec = importlib.util.spec_from_file_location("verify_baseline_setup_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_prints_coverage_for_valid_matrix(
    script_module, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Main should print success output when all required configs are present."""
    baseline_dir = Path(__file__).resolve().parents[2] / "experiments" / "configs" / "baselines"
    monkeypatch.setattr(
        script_module,
        "parse_args",
        lambda: argparse.Namespace(config_dir=baseline_dir),
    )

    script_module.main()

    output = capsys.readouterr().out
    assert "Validated baseline configs: 6" in output
    assert "grpo:math12k" in output
    assert "ppo:hotpotqa" in output


def test_main_raises_when_required_pair_missing(
    script_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Main should raise when any required algorithm/environment pair is missing."""
    partial_dir = tmp_path / "baselines"
    partial_dir.mkdir(parents=True)
    (partial_dir / "only_one.toml").write_text(
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

    monkeypatch.setattr(
        script_module,
        "parse_args",
        lambda: argparse.Namespace(config_dir=partial_dir),
    )

    with pytest.raises(ValueError, match="Missing required baseline configs"):
        script_module.main()
