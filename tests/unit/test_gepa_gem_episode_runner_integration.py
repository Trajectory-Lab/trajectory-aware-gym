"""Integration-style unit tests for GEPA with the concrete GEM episode runner."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from trajectory_aware_gym.adapters.gem_episode_runner import GEMEpisodeRunner
from trajectory_aware_gym.optimizers import GEPAOptimizer, build_trajectory_evaluator


class _FakeGemModule:
    def __init__(self, env):
        self._env = env

    def make(self, environment_id: str):
        assert environment_id == "math:Orz57K"
        return self._env


def test_gepa_optimizer_improves_prompt_with_gem_episode_runner(monkeypatch):
    class FakeEnv:
        def reset(self, **kwargs):
            return "Solve 2 + 2", {}

        def step(self, action: str):
            reward = 1.0 if action == "\\boxed{4}" else 0.0
            return "done", reward, True, False, {"correct": reward > 0}

        def close(self):
            return None

    def fake_import_module(name: str):
        if name == "gem":
            return _FakeGemModule(FakeEnv())
        if name == "gem.envs":
            return SimpleNamespace()
        raise AssertionError(f"unexpected import: {name}")

    async def fake_completion(*, messages, **kwargs):
        system_prompt = messages[0]["content"]
        action = "\\boxed{4}" if "better" in system_prompt else "\\boxed{0}"
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        message = SimpleNamespace(content=action, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.importlib.import_module",
        fake_import_module,
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.acompletion",
        AsyncMock(side_effect=fake_completion),
    )
    monkeypatch.setattr(
        "trajectory_aware_gym.adapters.gem_episode_runner.completion_cost",
        lambda *, completion_response: 0.0,
    )

    runner = GEMEpisodeRunner(
        environment_id="math:Orz57K",
        model_id="ollama_chat/qwen3:1.7b",
        temperature=0.0,
        max_steps=1,
    )
    optimizer = GEPAOptimizer(
        evaluator=build_trajectory_evaluator(runner),
        mutator=lambda prompt, iteration, candidate_index: f"{prompt} better",
        population_size=3,
        iterations=2,
        elite_count=1,
        random_seed=0,
    )

    result = optimizer.optimize("baseline")

    assert result.best_prompt.endswith("better")
    assert result.best_fitness > 0.0
