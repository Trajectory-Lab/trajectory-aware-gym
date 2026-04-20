"""Tests for the GEM env factory, including upstream bug patches."""

from __future__ import annotations

import importlib

import pytest

from trajectory_aware_gym.adapters import gem_env_factory


@pytest.fixture(autouse=True)
def _reset_factory_state():
    gem_env_factory.reset_dataset_cache()
    yield
    gem_env_factory.reset_dataset_cache()


def test_qa_env_step_returns_incorrect_when_answer_not_extractable():
    """Without the patch, QaEnv.step raises UnboundLocalError when the
    model output contains no extractable answer tag. The patched step
    must return ``correct=False`` with reward 0.0 instead.
    """
    gem_env_factory._apply_upstream_patches()

    qa_env = importlib.import_module("gem.envs.qa_env")

    class _StubQaEnv:
        extractor = staticmethod(lambda _action: None)
        answer = "42"

        def check_correct(self, _model_answer, _answer):  # pragma: no cover
            raise AssertionError("check_correct must not be called when extractor returns None")

    obs, reward, terminated, truncated, info = qa_env.QaEnv.step(_StubQaEnv(), "no answer here")

    assert obs == qa_env.TERMINAL_STATE
    assert reward == 0.0
    assert terminated is True
    assert truncated is True
    assert info == {"correct": False}


def test_qa_env_step_reports_correct_when_extractor_matches():
    gem_env_factory._apply_upstream_patches()

    qa_env = importlib.import_module("gem.envs.qa_env")

    class _StubQaEnv:
        extractor = staticmethod(lambda action: action.strip())
        answer = "paris"

        @staticmethod
        def check_correct(model_answer, answer):
            return model_answer.lower() == answer.lower()

    _obs, reward, _terminated, _truncated, info = qa_env.QaEnv.step(_StubQaEnv(), "Paris")

    assert reward == 1.0
    assert info == {"correct": True}


def test_apply_upstream_patches_is_idempotent():
    gem_env_factory._apply_upstream_patches()
    qa_env = importlib.import_module("gem.envs.qa_env")
    patched_step = qa_env.QaEnv.step

    gem_env_factory._apply_upstream_patches()

    assert qa_env.QaEnv.step is patched_step
