"""Tests for the GEPA dry-run script wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import dspy
import pytest
from scripts.run_gepa_dry_run import build_runner, build_trainset, load_config

MATH_DRY_RUN_CONFIG = Path("experiments/math-dry-run/config.yaml")


class TestLoadConfig:
    def test_loads_math_dry_run(self):
        config = load_config(MATH_DRY_RUN_CONFIG)
        assert config.name == "math-dry-run"
        assert config.environment.gem_env_id == "math:Orz57K"
        assert config.gepa_budget.mode == "light"


class TestBuildRunner:
    def test_creates_runner_from_config(self):
        config = load_config(MATH_DRY_RUN_CONFIG)
        runner = build_runner(config)
        assert runner._environment_id == "math:Orz57K"
        assert runner._model_id == "bedrock/us.meta.llama3-1-8b-instruct-v1:0"
        assert runner._max_steps == 5


class TestBuildTrainset:
    def test_builds_examples_from_gem_resets(self):
        mock_env = MagicMock()
        mock_env.reset.return_value = ("Solve: 2+2", {})

        mock_gem = MagicMock()
        mock_gem.make.return_value = mock_env

        config = load_config(MATH_DRY_RUN_CONFIG)

        with patch.dict(sys.modules, {"gem": mock_gem, "gem.envs": MagicMock()}):
            trainset = build_trainset(config)

        assert len(trainset) == config.environment.train_size
        assert all(isinstance(ex, dspy.Example) for ex in trainset)
        assert trainset[0].problem == "Solve: 2+2"

    def test_examples_have_inputs_set(self):
        mock_env = MagicMock()
        mock_env.reset.return_value = ("problem text", {})
        mock_gem = MagicMock()
        mock_gem.make.return_value = mock_env

        config = load_config(MATH_DRY_RUN_CONFIG)

        with patch.dict(sys.modules, {"gem": mock_gem, "gem.envs": MagicMock()}):
            trainset = build_trainset(config)

        assert "problem" in trainset[0].inputs().keys()
        assert "seed" in trainset[0].inputs().keys()
