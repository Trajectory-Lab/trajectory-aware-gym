"""Tests for fitness function protocols and result types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trajectory_aware_gym.fitness.types import (
    FitnessBreakdown,
    FitnessResult,
)


class TestFitnessBreakdown:
    """Tests for per-term fitness breakdown model."""

    @pytest.mark.parametrize(
        ("term_name", "raw_value", "weight", "weighted_value"),
        [
            ("discounted_return", 1.5, 1.0, 1.5),
            ("loop_penalty", -0.3, 0.5, -0.15),
            ("step_efficiency", 0.0, 1.0, 0.0),
            ("custom_term", -1.0, 0.0, 0.0),
        ],
    )
    def test_valid_construction(self, term_name, raw_value, weight, weighted_value):
        breakdown = FitnessBreakdown(
            term_name=term_name,
            raw_value=raw_value,
            weight=weight,
            weighted_value=weighted_value,
        )
        assert breakdown.term_name == term_name
        assert breakdown.raw_value == raw_value
        assert breakdown.weight == weight
        assert breakdown.weighted_value == weighted_value

    def test_empty_term_name_rejected(self):
        with pytest.raises(ValidationError):
            FitnessBreakdown(
                term_name="",
                raw_value=0.0,
                weight=1.0,
                weighted_value=0.0,
            )


class TestFitnessResult:
    """Tests for full fitness evaluation result model."""

    def test_valid_construction(self):
        result = FitnessResult(
            score=2.5,
            breakdown=[
                FitnessBreakdown(term_name="main", raw_value=3.0, weight=1.0, weighted_value=3.0),
                FitnessBreakdown(
                    term_name="penalty", raw_value=-0.5, weight=1.0, weighted_value=-0.5
                ),
            ],
            trajectory_length=5,
            metadata={"env": "math12k"},
        )
        assert result.score == 2.5
        assert len(result.breakdown) == 2
        assert result.trajectory_length == 5
        assert result.metadata["env"] == "math12k"

    def test_defaults(self):
        result = FitnessResult(score=0.0, trajectory_length=0)
        assert result.breakdown == []
        assert result.metadata == {}

    def test_negative_score_allowed(self):
        result = FitnessResult(score=-1.5, trajectory_length=3)
        assert result.score == -1.5

    def test_negative_trajectory_length_rejected(self):
        with pytest.raises(ValidationError):
            FitnessResult(score=0.0, trajectory_length=-1)
