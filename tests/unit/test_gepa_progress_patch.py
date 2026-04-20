"""Tests for the GEPA inner-eval progress bar monkey-patch."""

from __future__ import annotations

import importlib

import pytest

from trajectory_aware_gym.optimizers.gepa_progress_patch import (
    enable_gepa_eval_progress,
    reset_gepa_eval_progress_patch,
)


@pytest.fixture(autouse=True)
def _reset_patch():
    reset_gepa_eval_progress_patch()
    yield
    reset_gepa_eval_progress_patch()


def test_enable_gepa_eval_progress_forces_display_progress():
    gepa_utils = importlib.import_module("dspy.teleprompt.gepa.gepa_utils")
    original_evaluate = gepa_utils.Evaluate

    enable_gepa_eval_progress()

    patched = gepa_utils.Evaluate
    assert patched is not original_evaluate
    assert issubclass(patched, original_evaluate)

    instance = patched(devset=[], metric=lambda *_: 0.0)
    assert instance.display_progress is True


def test_enable_gepa_eval_progress_respects_explicit_override():
    enable_gepa_eval_progress()
    gepa_utils = importlib.import_module("dspy.teleprompt.gepa.gepa_utils")
    instance = gepa_utils.Evaluate(devset=[], metric=lambda *_: 0.0, display_progress=False)
    assert instance.display_progress is False


def test_enable_gepa_eval_progress_is_idempotent():
    enable_gepa_eval_progress()
    gepa_utils = importlib.import_module("dspy.teleprompt.gepa.gepa_utils")
    patched_once = gepa_utils.Evaluate

    enable_gepa_eval_progress()

    assert gepa_utils.Evaluate is patched_once


def test_reset_restores_original_evaluate():
    gepa_utils = importlib.import_module("dspy.teleprompt.gepa.gepa_utils")
    original_evaluate = gepa_utils.Evaluate

    enable_gepa_eval_progress()
    assert gepa_utils.Evaluate is not original_evaluate

    reset_gepa_eval_progress_patch()
    assert gepa_utils.Evaluate is original_evaluate
