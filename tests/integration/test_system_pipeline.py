"""Integration tests for trajectory + provider system pipeline."""

from __future__ import annotations

import json

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryLogger
from trajectory_aware_gym.config.settings import ProjectPaths


def test_episode_trajectory_persists_to_logs(tmp_path):
    """Episode transitions are persisted into the logs directory."""
    paths = ProjectPaths(root=tmp_path)
    logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy", seed=7)
    logger.set_initial_state("start", {"suffix": "next"})
    logger.add_step(
        action="\\\\boxed{3}",
        observation="higher",
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"suffix": "next"},
    )
    logger.add_step(
        action="\\\\boxed{5}",
        observation="win",
        reward=1.0,
        terminated=True,
        truncated=False,
        info={"suffix": "next"},
    )

    output_file = logger.save(project_paths=paths)

    assert output_file.exists()
    assert output_file.parent == paths.logs

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["total_reward"] == pytest.approx(1.0)
    assert payload["steps"][-1]["terminated"] is True


def test_saved_trajectory_round_trip_validation(tmp_path):
    """Persisted trajectory payload can be revalidated through TrajectoryLog."""
    paths = ProjectPaths(root=tmp_path)
    logger = TrajectoryLogger(environment_id="toy-env")
    logger.set_initial_state("initial")
    logger.add_step(
        action="act",
        observation="obs",
        reward=0.5,
        terminated=True,
        truncated=False,
    )

    output_file = logger.save(project_paths=paths)
    loaded = json.loads(output_file.read_text(encoding="utf-8"))
    reconstructed = TrajectoryLog(**loaded)

    assert reconstructed.environment_id == "toy-env"
    assert reconstructed.total_reward == pytest.approx(0.5)


@pytest.fixture
def fake_lm(monkeypatch):
    """Capture LM constructor payloads without calling external providers."""
    calls: list[dict[str, object]] = []

    class FakeLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.kwargs = kwargs

    import trajectory_aware_gym.config.llm_provider as llm_provider

    monkeypatch.setattr(llm_provider.dspy, "LM", FakeLM)
    return calls


def test_task_model_routing_and_temperature(monkeypatch, fake_lm):
    """Task model factory routes model names and temperature correctly."""
    import trajectory_aware_gym.config.llm_provider as llm_provider

    class FakeGEMConfig:
        gem_temperature_train = 0.9
        gem_temperature_eval = 0.1

    class FakeOllamaConfig:
        ollama_api_base = "http://fake-ollama"
        local_task_model_1_7b = "ollama_chat/qwen3:1.7b"
        local_task_model_4b = "ollama_chat/qwen3:4b"

    class FakeAWSConfig:
        bedrock_llama_1b = "bedrock-llama-1b"
        bedrock_llama_3b = "bedrock-llama-3b"
        bedrock_llama_8b = "bedrock-llama-8b"

    monkeypatch.setattr(llm_provider, "GEMConfig", FakeGEMConfig)
    monkeypatch.setattr(llm_provider, "OllamaConfig", FakeOllamaConfig)
    monkeypatch.setattr(llm_provider, "AWSConfig", FakeAWSConfig)

    llm_provider.get_task_lm("qwen3:1.7b", mode="train")
    llm_provider.get_task_lm("qwen3:4b", mode="eval")
    llm_provider.get_task_lm("llama:1b", mode="train")

    assert fake_lm[0]["model"] == "ollama_chat/qwen3:1.7b"
    assert fake_lm[0]["api_base"] == "http://fake-ollama"
    assert fake_lm[0]["temperature"] == pytest.approx(0.9)

    assert fake_lm[1]["model"] == "ollama_chat/qwen3:4b"
    assert fake_lm[1]["temperature"] == pytest.approx(0.1)

    assert fake_lm[2]["model"] == "bedrock/bedrock-llama-1b"
    assert fake_lm[2]["temperature"] == pytest.approx(0.9)


def test_reflection_model_routing(monkeypatch, fake_lm):
    """Reflection model factory uses GEPA reflection model setting."""
    import trajectory_aware_gym.config.llm_provider as llm_provider

    class FakeGEPAConfig:
        gepa_reflection_model = "anthropic.claude-sonnet-4-5-v2:0"

    monkeypatch.setattr(llm_provider, "GEPAConfig", FakeGEPAConfig)

    llm_provider.get_reflection_lm()

    assert fake_lm[0]["model"] == "bedrock/anthropic.claude-sonnet-4-5-v2:0"
    assert fake_lm[0]["temperature"] == pytest.approx(1.0)
    assert fake_lm[0]["max_tokens"] == 4096
