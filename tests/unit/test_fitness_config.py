"""Tests for fitness function configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trajectory_aware_gym.config import FitnessModel, settings
from trajectory_aware_gym.fitness.config import FitnessModel as ReExportedFitnessModel

# Base valid kwargs for constructing FitnessModel directly
_VALID_BASE = {
    "gamma": 0.99,
    "lambda": 0.1,
    "loop_penalty_weight": 1.0,
    "step_efficiency_weight": 1.0,
    "max_steps": 50,
    "loop_window": 3,
}


class TestFitnessConfigReExport:
    """fitness.config re-exports FitnessModel from the canonical location."""

    def test_re_export_is_same_class(self):
        assert ReExportedFitnessModel is FitnessModel


class TestFitnessModelDefaults:
    """FitnessModel loads defaults from config/trajectory-aware-gym.yaml via settings."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("gamma", 0.99),
            ("lambda_", 0.1),
            ("loop_penalty_weight", 1.0),
            ("step_efficiency_weight", 1.0),
            ("max_steps", 50),
            ("loop_window", 3),
        ],
    )
    def test_default_values(self, field, expected):
        assert getattr(settings.fitness, field) == expected


class TestFitnessModelValidation:
    """FitnessModel rejects invalid values."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("gamma", 0.0),
            ("gamma", 0.5),
            ("gamma", 1.0),
            ("lambda_", 0.0),
            ("lambda_", 10.0),
            ("loop_penalty_weight", 0.0),
            ("step_efficiency_weight", 0.0),
            ("max_steps", 1),
            ("max_steps", 100),
            ("loop_window", 1),
            ("loop_window", 10),
        ],
    )
    def test_valid_overrides(self, field, value):
        config = settings.fitness.model_copy(update={field: value})
        assert getattr(config, field) == value

    @pytest.mark.parametrize(
        ("field", "invalid_value"),
        [
            ("gamma", -0.1),
            ("gamma", 1.1),
            ("lambda", -0.1),
            ("loop_penalty_weight", -1.0),
            ("step_efficiency_weight", -0.5),
            ("max_steps", 0),
            ("max_steps", -1),
            ("loop_window", 0),
            ("loop_window", -1),
        ],
    )
    def test_invalid_values_rejected(self, field, invalid_value):
        with pytest.raises(ValidationError):
            FitnessModel(**{**_VALID_BASE, field: invalid_value})
