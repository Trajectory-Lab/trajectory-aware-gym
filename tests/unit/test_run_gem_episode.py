"""Tests for the configurable smoke helpers in scripts/run_gem_episode.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_gem_episode.py"
SPEC = importlib.util.spec_from_file_location("run_gem_episode", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_gem_episode = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_gem_episode
SPEC.loader.exec_module(run_gem_episode)


def make_args(**overrides):
    """Return CLI-like args for helper testing."""
    base = {
        "environment": None,
        "seed": None,
        "system_prompt": None,
        "smoke": True,
        "experiment_config": None,
        "task_model_id": None,
        "mode": "eval",
        "max_steps": None,
        "temperature": None,
        "max_response_tokens": None,
        "show_log": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_smoke_run_spec_from_experiment_config():
    args = make_args(
        experiment_config=Path("experiments/quick-test/config.yaml"),
        task_model_id="bedrock/test-qwen-profile",
    )

    spec = run_gem_episode.build_smoke_run_spec(args)

    assert spec.environment_id == "math:Orz57K"
    assert spec.experiment_name == "quick-test"
    assert spec.model_id == "bedrock/test-qwen-profile"
    assert spec.seed == 42
    assert spec.episode_count == 1
    assert spec.episode_max_steps == 10
    assert spec.max_response_tokens == 4096
    assert spec.temperature == pytest.approx(0.0)


def test_build_smoke_run_spec_respects_overrides():
    args = make_args(
        experiment_config=Path("experiments/hotpotqa/config.yaml"),
        environment="qa:HotpotQA",
        seed=999,
        task_model_id="bedrock/custom-qwen",
        mode="train",
        max_steps=2,
        temperature=0.3,
        max_response_tokens=1536,
        system_prompt="custom prompt",
    )

    spec = run_gem_episode.build_smoke_run_spec(args)

    assert spec.environment_id == "qa:HotpotQA"
    assert spec.model_id == "bedrock/custom-qwen"
    assert spec.seed == 999
    assert spec.episode_count == 2
    assert spec.episode_max_steps == 10
    assert spec.max_response_tokens == 1536
    assert spec.temperature == pytest.approx(0.3)
    assert spec.system_prompt == "custom prompt"


def test_build_smoke_run_spec_requires_model_without_config():
    args = make_args()

    with pytest.raises(ValueError, match="task-model-id"):
        run_gem_episode.build_smoke_run_spec(args)


def test_build_smoke_messages_include_observation():
    messages = run_gem_episode.build_smoke_messages(
        observation="Solve 2 + 2",
        system_prompt="smoke prompt",
    )

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "smoke prompt"
    assert messages[1]["role"] == "user"
    assert "Solve 2 + 2" in messages[1]["content"]


@pytest.mark.parametrize(
    ("model_id", "expected_api_base"),
    [
        ("ollama_chat/qwen3:1.7b", "http://localhost:11434"),
        ("bedrock/test-qwen-profile", None),
    ],
)
def test_build_completion_kwargs(model_id, expected_api_base):
    kwargs = run_gem_episode._build_completion_kwargs(model_id, temperature=0.0)

    assert kwargs["model"] == model_id
    assert kwargs["temperature"] == pytest.approx(0.0)
    assert kwargs["max_tokens"] == run_gem_episode.DEFAULT_SMOKE_MAX_TOKENS
    if expected_api_base is None:
        assert "api_base" not in kwargs
    else:
        assert kwargs["api_base"] == expected_api_base
