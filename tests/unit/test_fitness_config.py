"""Tests for fitness function configuration."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from trajectory_aware_gym.config import FitnessConfig


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clear environment variables that might interfere with config tests."""
    for key in list(os.environ.keys()):
        if key.startswith("FITNESS_"):
            monkeypatch.delenv(key, raising=False)


class TestFitnessConfig:
    """Tests for FitnessConfig defaults and validation."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("fitness_gamma", 0.99),
            ("fitness_lambda", 0.1),
            ("fitness_loop_penalty_weight", 1.0),
            ("fitness_step_efficiency_weight", 1.0),
            ("fitness_max_steps", 50),
            ("fitness_loop_window", 3),
        ],
    )
    def test_default_values(self, field, expected):
        config = FitnessConfig(_env_file=None)
        assert getattr(config, field) == expected

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("fitness_gamma", 0.0),
            ("fitness_gamma", 0.5),
            ("fitness_gamma", 1.0),
            ("fitness_lambda", 0.0),
            ("fitness_lambda", 10.0),
            ("fitness_loop_penalty_weight", 0.0),
            ("fitness_step_efficiency_weight", 0.0),
            ("fitness_max_steps", 1),
            ("fitness_max_steps", 100),
            ("fitness_loop_window", 1),
            ("fitness_loop_window", 10),
        ],
    )
    def test_valid_overrides(self, field, value):
        config = FitnessConfig(**{field: value}, _env_file=None)
        assert getattr(config, field) == value

    @pytest.mark.parametrize(
        ("field", "invalid_value"),
        [
            ("fitness_gamma", -0.1),
            ("fitness_gamma", 1.1),
            ("fitness_lambda", -0.1),
            ("fitness_loop_penalty_weight", -1.0),
            ("fitness_step_efficiency_weight", -0.5),
            ("fitness_max_steps", 0),
            ("fitness_max_steps", -1),
            ("fitness_loop_window", 0),
            ("fitness_loop_window", -1),
        ],
    )
    def test_invalid_values_rejected(self, field, invalid_value):
        with pytest.raises(ValidationError):
            FitnessConfig(**{field: invalid_value}, _env_file=None)
