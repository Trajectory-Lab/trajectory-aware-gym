"""Integration tests for trajectory + provider system pipeline."""

from __future__ import annotations

import types

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import (
    LLMCallMetadata,
    TrajectoryLogger,
    load_trajectory,
)
from trajectory_aware_gym.config.settings import ProjectPaths
from trajectory_aware_gym.metrics import extract_episode_raw_metrics


def test_episode_trajectory_persists_to_logs(tmp_path):
    """Episode transitions are persisted into the logs directory."""
    paths = ProjectPaths(root=tmp_path)
    logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy", seed=7)
    logger.set_initial_state("start", {"suffix": "next"})
    logger.add_step(
        action="\\boxed{3}",
        observation="higher",
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"suffix": "next"},
    )
    logger.add_step(
        action="\\boxed{5}",
        observation="win",
        reward=1.0,
        terminated=True,
        truncated=False,
        info={"suffix": "next"},
    )

    db_path = logger.save(project_paths=paths)

    assert db_path.exists()
    assert db_path.parent == paths.logs

    loaded = load_trajectory(db_path, run_id=logger.last_run_id)
    assert loaded.total_reward == pytest.approx(1.0)
    assert loaded.steps[-1].terminated is True


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

    db_path = logger.save(project_paths=paths)
    loaded = load_trajectory(db_path, run_id=logger.last_run_id)

    assert loaded.environment_id == "toy-env"
    assert loaded.total_reward == pytest.approx(0.5)


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


@pytest.fixture
def mock_settings(monkeypatch):
    """Create a mock settings module with simple objects instead of Properties."""

    # Create simple objects with the needed attributes
    class MockGem:
        def __init__(self):
            self.temperature_train = 0.9
            self.temperature_eval = 0.1

    class MockOllama:
        def __init__(self):
            self.api_base = "http://fake-ollama"
            self.task_model_1_7b = "ollama_chat/qwen3:1.7b"
            self.task_model_4b = "ollama_chat/qwen3:4b"

    class MockAWS:
        def __init__(self):
            self.region = "us-east-1"
            self.bedrock_llama_1b = "us.meta.llama3-2-1b-instruct-v1:0"
            self.bedrock_llama_3b = "us.meta.llama3-2-3b-instruct-v1:0"
            self.bedrock_llama_8b = "us.meta.llama3-1-8b-instruct-v1:0"

    class MockGEPA:
        def __init__(self):
            self.reflection_model = "openai.gpt-oss-120b-1:0"

    # Create a mock settings module
    mock_settings_module = types.ModuleType("mock_settings")
    mock_settings_module.gem = MockGem()
    mock_settings_module.ollama = MockOllama()
    mock_settings_module.aws = MockAWS()
    mock_settings_module.gepa = MockGEPA()

    # Replace the settings module in llm_provider's namespace
    import trajectory_aware_gym.config.llm_provider as llm_provider

    monkeypatch.setattr(llm_provider, "settings", mock_settings_module)


@pytest.mark.parametrize(
    ("model", "mode", "expected_model", "expected_temp", "expected_api_base"),
    [
        pytest.param(
            "qwen3:1.7b",
            "train",
            "ollama_chat/qwen3:1.7b",
            0.9,
            "http://fake-ollama",
            id="ollama-1.7b-train",
        ),
        pytest.param(
            "qwen3:4b",
            "eval",
            "ollama_chat/qwen3:4b",
            0.1,
            "http://fake-ollama",
            id="ollama-4b-eval",
        ),
        pytest.param(
            "llama:1b",
            "train",
            "bedrock/us.meta.llama3-2-1b-instruct-v1:0",
            0.9,
            None,
            id="bedrock-llama-1b-train",
        ),
        pytest.param(
            "llama:3b",
            "eval",
            "bedrock/us.meta.llama3-2-3b-instruct-v1:0",
            0.1,
            None,
            id="bedrock-llama-3b-eval",
        ),
        pytest.param(
            "llama:8b",
            "train",
            "bedrock/us.meta.llama3-1-8b-instruct-v1:0",
            0.9,
            None,
            id="bedrock-llama-8b-train",
        ),
    ],
)
def test_task_model_routing_and_temperature(
    monkeypatch,
    fake_lm,
    mock_settings,
    model,
    mode,
    expected_model,
    expected_temp,
    expected_api_base,
):
    """Task model factory routes model names and temperature correctly."""
    import trajectory_aware_gym.config.llm_provider as llm_provider

    # Settings are already mocked by the fixture
    llm_provider.get_task_lm(model, mode=mode)

    assert fake_lm[0]["model"] == expected_model
    assert fake_lm[0]["temperature"] == pytest.approx(expected_temp)
    assert fake_lm[0]["max_tokens"] == 4096
    if expected_api_base:
        assert fake_lm[0]["api_base"] == expected_api_base
    else:
        assert "api_base" not in fake_lm[0]


def test_task_model_unrecognised_name_returns_none(monkeypatch, fake_lm, mock_settings):
    """Unrecognised model name falls through match/case and returns None."""
    import trajectory_aware_gym.config.llm_provider as llm_provider

    result = llm_provider.get_task_lm("nonexistent-model", mode="train")

    assert result is None
    assert len(fake_lm) == 0


def test_reflection_model_routing(monkeypatch, fake_lm, mock_settings):
    """Reflection model factory uses GEPA reflection model setting."""
    import trajectory_aware_gym.config.llm_provider as llm_provider

    llm_provider.get_reflection_lm()

    assert fake_lm[0]["model"] == "bedrock/openai.gpt-oss-120b-1:0"
    assert fake_lm[0]["temperature"] == pytest.approx(1.0)
    assert fake_lm[0]["max_tokens"] == 4096
    assert fake_lm[0]["aws_region_name"] == "us-east-1"


def test_reflection_model_routing_accepts_explicit_model(fake_lm, mock_settings):
    """Reflection model factory can be overridden by experiment-scoped config."""
    import trajectory_aware_gym.config.llm_provider as llm_provider

    llm_provider.get_reflection_lm(
        "openai.gpt-oss-120b-1:0",
        temperature=0.7,
        max_tokens=2048,
    )

    assert fake_lm[0]["model"] == "bedrock/openai.gpt-oss-120b-1:0"
    assert fake_lm[0]["temperature"] == pytest.approx(0.7)
    assert fake_lm[0]["max_tokens"] == 2048
    assert fake_lm[0]["aws_region_name"] == "us-east-1"


def test_episode_trajectory_produces_valid_raw_metrics(tmp_path):
    """Raw metrics are extractable from a persisted trajectory with LLM call metadata."""
    paths = ProjectPaths(root=tmp_path)
    logger = TrajectoryLogger(environment_id="math:Orz57K", seed=42)
    logger.set_system_prompt("Solve carefully.")
    logger.set_initial_state("Solve 2+2", {"source": "integration"})
    logger.add_step(
        action="\\boxed{4}",
        observation="Correct",
        reward=1.0,
        terminated=True,
        truncated=False,
        info={"correct": True},
        llm_calls=[
            LLMCallMetadata(
                model_id="ollama_chat/qwen3:1.7b",
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
                cost_usd=0.005,
            )
        ],
    )

    db_path = logger.save(project_paths=paths)
    loaded = load_trajectory(db_path, run_id=logger.last_run_id)
    metrics = extract_episode_raw_metrics(loaded)

    assert metrics.environment_id == "math:Orz57K"
    assert metrics.step_count == 1
    assert metrics.success is True
    assert metrics.total_reward == 1.0
    assert metrics.total_tokens == 60
    assert metrics.prompt_tokens == 50
    assert metrics.completion_tokens == 10
    assert metrics.llm_cost_usd == pytest.approx(0.005)
    assert metrics.token_data_coverage == 1.0
    assert metrics.cost_data_coverage == 1.0


def test_multi_step_episode_metrics_aggregate_across_steps(tmp_path):
    """Multi-step trajectory metrics aggregate token/cost data and track coverage."""
    paths = ProjectPaths(root=tmp_path)
    logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy", seed=7)
    logger.set_initial_state("start")
    logger.add_step(
        action="\\boxed{3}",
        observation="higher",
        reward=0.0,
        terminated=False,
        truncated=False,
        llm_calls=[
            LLMCallMetadata(
                model_id="test-model",
                prompt_tokens=30,
                completion_tokens=5,
                total_tokens=35,
                cost_usd=0.001,
            )
        ],
    )
    logger.add_step(
        action="\\boxed{5}",
        observation="win",
        reward=1.0,
        terminated=True,
        truncated=False,
        llm_calls=[
            LLMCallMetadata(
                model_id="test-model",
                prompt_tokens=40,
                completion_tokens=8,
                total_tokens=48,
                cost_usd=0.002,
            )
        ],
    )

    db_path = logger.save(project_paths=paths)
    loaded = load_trajectory(db_path, run_id=logger.last_run_id)
    metrics = extract_episode_raw_metrics(loaded)

    assert metrics.step_count == 2
    assert metrics.total_tokens == 83
    assert metrics.prompt_tokens == 70
    assert metrics.llm_cost_usd == pytest.approx(0.003)
    assert metrics.tokens_per_step == pytest.approx(83 / 2)
    assert metrics.cost_per_step_usd == pytest.approx(0.003 / 2)
    assert metrics.repeat_action_rate == 0.0
    assert metrics.token_data_coverage == 1.0


def test_partial_instrumentation_coverage_tracked(tmp_path):
    """Coverage fractions reflect steps with and without LLM metadata."""
    paths = ProjectPaths(root=tmp_path)
    logger = TrajectoryLogger(environment_id="math:Orz57K", seed=1)
    logger.set_initial_state("problem")
    logger.add_step(
        action="attempt1",
        observation="wrong",
        reward=0.0,
        terminated=False,
        truncated=False,
        llm_calls=[
            LLMCallMetadata(
                model_id="m",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cost_usd=0.001,
            )
        ],
    )
    logger.add_step(
        action="attempt2",
        observation="done",
        reward=0.0,
        terminated=False,
        truncated=True,
    )

    db_path = logger.save(project_paths=paths)
    loaded = load_trajectory(db_path, run_id=logger.last_run_id)
    metrics = extract_episode_raw_metrics(loaded)

    assert metrics.step_count == 2
    assert metrics.token_data_coverage == pytest.approx(0.5)
    assert metrics.cost_data_coverage == pytest.approx(0.5)
    assert metrics.total_tokens == 15
    assert metrics.llm_cost_usd == pytest.approx(0.001)
