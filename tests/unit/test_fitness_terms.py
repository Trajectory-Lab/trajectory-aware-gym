"""Tests for individual fitness term implementations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import (
    LLMCallMetadata,
    TrajectoryLog,
    TrajectoryStep,
)
from trajectory_aware_gym.config import FitnessModel, settings
from trajectory_aware_gym.fitness.terms import (
    ActionStabilityTerm,
    CallEfficiencyBonusTerm,
    DiscountedReturnTerm,
    LoopDetectionPenaltyTerm,
    NormalizedProgressTerm,
    StepEfficiencyBonusTerm,
)


def _cfg(**overrides: object) -> FitnessModel:
    """Build a FitnessModel from YAML defaults with specific overrides."""
    return settings.fitness.model_copy(update=overrides)


def _stub_llm_call() -> LLMCallMetadata:
    return LLMCallMetadata(model_id="stub", prompt_tokens=0, completion_tokens=0, total_tokens=0)


@pytest.fixture
def make_trajectory():
    """Factory for building TrajectoryLog objects with controlled properties.

    By default each step is given a single stub LLM call so that call-based
    efficiency terms have something to measure. Pass ``calls_per_step`` to
    override this, or ``llm_calls_per_step``/``tool_calls_per_step`` to
    control each separately.
    """

    def _make(
        rewards: list[float],
        actions: list[str] | None = None,
        environment_id: str = "test-env",
        calls_per_step: int | None = None,
        llm_calls_per_step: int | None = None,
        tool_calls_per_step: int | None = None,
    ) -> TrajectoryLog:
        now = datetime.now(UTC)
        n = len(rewards)
        if actions is None:
            actions = [f"action_{i}" for i in range(n)]

        if llm_calls_per_step is None:
            llm_calls_per_step = calls_per_step if calls_per_step is not None else 1
        if tool_calls_per_step is None:
            tool_calls_per_step = 0

        steps = [
            TrajectoryStep(
                step_index=i + 1,
                action=actions[i],
                observation=f"obs_{i}",
                reward=rewards[i],
                terminated=(i == n - 1),
                truncated=False,
                llm_calls=[_stub_llm_call() for _ in range(llm_calls_per_step)],
                tool_calls=[],  # tool_calls require ToolCall instances; add in specific tests
            )
            for i in range(n)
        ]
        _ = tool_calls_per_step  # reserved for future use
        return TrajectoryLog(
            environment_id=environment_id,
            seed=42,
            started_at=now,
            finished_at=now + timedelta(seconds=max(n, 1)),
            initial_observation="start",
            steps=steps,
            total_reward=sum(rewards),
            num_steps=n,
        )

    return _make


class TestDefaultConfigFallback:
    """Terms use settings.fitness when no config is passed."""

    def test_discounted_return_default(self, make_trajectory):
        term = DiscountedReturnTerm()
        trajectory = make_trajectory(rewards=[0.0, 1.0])
        assert term.compute(trajectory) > 0

    def test_loop_detection_default(self, make_trajectory):
        term = LoopDetectionPenaltyTerm()
        trajectory = make_trajectory(rewards=[0.0, 1.0], actions=["a", "a"])
        assert term.compute(trajectory) < 0

    def test_step_efficiency_default(self, make_trajectory):
        term = StepEfficiencyBonusTerm()
        trajectory = make_trajectory(rewards=[0.0, 1.0])
        assert term.compute(trajectory) > 0

    def test_call_efficiency_default(self, make_trajectory):
        term = CallEfficiencyBonusTerm()
        trajectory = make_trajectory(rewards=[0.0, 1.0])
        assert term.compute(trajectory) > 0


class TestDiscountedReturnTerm:
    """Tests for the reverse-time discounted return (Eq. 3.1)."""

    def test_empty_trajectory(self, make_trajectory):
        trajectory = make_trajectory(rewards=[])
        config = _cfg(gamma=0.99, lambda_=0.1)
        term = DiscountedReturnTerm(config)
        assert term.compute(trajectory) == 0.0

    def test_name(self):
        term = DiscountedReturnTerm(settings.fitness)
        assert term.name == "discounted_return"

    @pytest.mark.parametrize(
        ("rewards", "gamma", "lam", "expected"),
        [
            # Single successful step: raw=1.1, max=1.1 → 1.0
            ([1.0], 0.99, 0.1, 1.0),
            # Single failed step: raw=0.0 → 0.0
            ([0.0], 0.99, 0.1, 0.0),
            # Two steps, success: raw=1.6, max=1.7 → 0.9412
            ([0.0, 1.0], 0.5, 0.1, 0.9411764705882354),
            # γ=0: only final step counts: raw=1.0, max=1.0 → 1.0
            ([0.0, 1.0], 0.0, 0.0, 1.0),
            # γ=1: uniform weighting: raw=2.0, max=2.0 → 1.0
            ([0.0, 1.0], 1.0, 0.0, 1.0),
            # Negative final reward (failure): raw=-0.05, max=2.19 → -0.0228
            ([0.5, -1.0], 0.99, 0.1, -0.02283105022831049),
            # Three steps success, γ=0.5, λ=0: raw=1.75, max=1.75 → 1.0
            ([0.0, 0.0, 1.0], 0.5, 0.0, 1.0),
            # λ=0 disables auxiliary term: raw=1.5, max=1.5 → 1.0
            ([0.0, 1.0], 0.5, 0.0, 1.0),
            # Only auxiliary term (γ=0): raw=1.5, max=2.0 → 0.75
            ([0.0, 1.0], 0.0, 0.5, 0.75),
        ],
    )
    def test_computation(self, make_trajectory, rewards, gamma, lam, expected):
        config = _cfg(gamma=gamma, lambda_=lam)
        term = DiscountedReturnTerm(config)
        trajectory = make_trajectory(rewards=rewards)
        result = term.compute(trajectory)
        assert result == pytest.approx(expected, abs=1e-9)


class TestLoopDetectionPenaltyTerm:
    """Tests for the loop detection penalty."""

    def test_empty_trajectory(self, make_trajectory):
        trajectory = make_trajectory(rewards=[], actions=[])
        term = LoopDetectionPenaltyTerm(settings.fitness)
        assert term.compute(trajectory) == 0.0

    def test_single_step(self, make_trajectory):
        trajectory = make_trajectory(rewards=[1.0], actions=["a"])
        term = LoopDetectionPenaltyTerm(settings.fitness)
        assert term.compute(trajectory) == 0.0

    def test_name(self):
        term = LoopDetectionPenaltyTerm(settings.fitness)
        assert term.name == "loop_detection_penalty"

    @pytest.mark.parametrize(
        ("actions", "window", "expected"),
        [
            # No loops
            (["a", "b", "c"], 3, 0.0),
            # All identical: step 1 matches [a], step 2 matches [a,a] -> 2/2 = -1.0
            (["a", "a", "a"], 3, -1.0),
            # Partial: step 1(b) no match in [a], step 2(a) match in [a,b] -> 1/2 = -0.5
            (["a", "b", "a"], 3, -0.5),
            # Outside window with window=2:
            # i=1: "b" not in ["a"] -> no loop
            # i=2: "c" not in ["a","b"][:window=2] -> ["a","b"] -> no loop
            # i=3: "a" not in ["b","c"] (window=2, lookback from 1 to 3) -> no loop
            (["a", "b", "c", "a"], 2, 0.0),
            # All identical with window=1:
            # i=1: "a" in ["a"] -> loop, i=2: "a" in ["a"] -> loop -> 2/2 = -1.0
            (["a", "a", "a"], 1, -1.0),
            # Two unique alternating: a,b,a,b with window=3
            # i=1: "b" not in ["a"] -> no
            # i=2: "a" in ["a","b"] -> yes
            # i=3: "b" in ["a","b","a"] -> yes -> 2/3
            (["a", "b", "a", "b"], 3, pytest.approx(-2 / 3)),
        ],
    )
    def test_computation(self, make_trajectory, actions, window, expected):
        rewards = [0.0] * (len(actions) - 1) + [1.0] if actions else []
        config = _cfg(loop_window=window)
        term = LoopDetectionPenaltyTerm(config)
        trajectory = make_trajectory(rewards=rewards, actions=actions)
        result = term.compute(trajectory)
        assert result == pytest.approx(expected, abs=1e-9)


class TestStepEfficiencyBonusTerm:
    """Tests for the env-step efficiency bonus (fewer turns = higher score)."""

    def test_empty_trajectory(self, make_trajectory):
        trajectory = make_trajectory(rewards=[])
        term = StepEfficiencyBonusTerm(settings.fitness)
        assert term.compute(trajectory) == 0.0

    def test_name(self):
        term = StepEfficiencyBonusTerm(settings.fitness)
        assert term.name == "step_efficiency_bonus"

    @pytest.mark.parametrize(
        ("num_steps", "max_steps", "final_reward", "expected"),
        [
            # Failed trajectory: no bonus
            (5, 50, 0.0, 0.0),
            # Negative final reward: no bonus
            (5, 50, -1.0, 0.0),
            # Quick success: 1.0 - 5/50 = 0.9
            (5, 50, 1.0, 0.9),
            # Max steps success: 1.0 - 50/50 = 0.0
            (50, 50, 1.0, 0.0),
            # Over max steps: clamped to 0.0
            (60, 50, 1.0, 0.0),
            # Single step success: 1.0 - 1/50 = 0.98
            (1, 50, 1.0, 0.98),
            # Half steps: 1.0 - 25/50 = 0.5
            (25, 50, 1.0, 0.5),
        ],
    )
    def test_computation(self, make_trajectory, num_steps, max_steps, final_reward, expected):
        rewards = [0.0] * (num_steps - 1) + [final_reward] if num_steps > 0 else []
        config = _cfg(max_steps=max_steps)
        term = StepEfficiencyBonusTerm(config)
        trajectory = make_trajectory(rewards=rewards)
        result = term.compute(trajectory)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_independent_of_call_count(self, make_trajectory):
        """Step efficiency looks only at env-step count, not LLM/tool call volume."""
        config = _cfg(max_steps=50, call_budget_per_step=8)
        term = StepEfficiencyBonusTerm(config)
        cheap = make_trajectory(rewards=[0.0, 1.0], calls_per_step=1)
        pricey = make_trajectory(rewards=[0.0, 1.0], calls_per_step=8)
        # Both have 2 env-steps so they score the same on this term.
        assert term.compute(cheap) == pytest.approx(term.compute(pricey), abs=1e-9)


class TestCallEfficiencyBonusTerm:
    """Tests for the LLM + tool call efficiency bonus."""

    def test_empty_trajectory(self, make_trajectory):
        trajectory = make_trajectory(rewards=[])
        term = CallEfficiencyBonusTerm(settings.fitness)
        assert term.compute(trajectory) == 0.0

    def test_name(self):
        term = CallEfficiencyBonusTerm(settings.fitness)
        assert term.name == "call_efficiency_bonus"

    def test_no_bonus_when_no_calls_logged(self, make_trajectory):
        """If a trajectory records no LLM/tool calls, there is no signal."""
        trajectory = make_trajectory(rewards=[0.0, 1.0], calls_per_step=0)
        term = CallEfficiencyBonusTerm(settings.fitness)
        assert term.compute(trajectory) == 0.0

    @pytest.mark.parametrize(
        ("num_steps", "calls_per_step", "max_steps", "call_budget", "final_reward", "expected"),
        [
            # Failed trajectory: no bonus regardless of call count
            (5, 1, 50, 8, 0.0, 0.0),
            # Negative final reward: no bonus
            (5, 1, 50, 8, -1.0, 0.0),
            # Cheap success: 5 calls / (50*8=400) -> 1 - 5/400 = 0.9875
            (5, 1, 50, 8, 1.0, 1.0 - 5 / 400),
            # Many cheap steps: 50 calls / 400 -> 1 - 50/400 = 0.875
            (50, 1, 50, 8, 1.0, 1.0 - 50 / 400),
            # Expensive steps dominate even at low env-step count:
            # 2 env-steps * 10 calls = 20 / (5*4=20) -> 0.0 (at budget)
            (2, 10, 5, 4, 1.0, 0.0),
            # Over call budget: 3 env-steps * 10 calls = 30 / (5*4=20) clamp to 0
            (3, 10, 5, 4, 1.0, 0.0),
            # Single cheap step, tight budget: 1/(1*8)=0.125 -> 0.875
            (1, 1, 1, 8, 1.0, 0.875),
            # Shorter trajectory beats longer one at equal calls-per-step:
            # 10 calls / (50*8=400) = 0.975
            (10, 1, 50, 8, 1.0, 1.0 - 10 / 400),
        ],
    )
    def test_computation(
        self,
        make_trajectory,
        num_steps,
        calls_per_step,
        max_steps,
        call_budget,
        final_reward,
        expected,
    ):
        rewards = [0.0] * (num_steps - 1) + [final_reward] if num_steps > 0 else []
        config = _cfg(max_steps=max_steps, call_budget_per_step=call_budget)
        term = CallEfficiencyBonusTerm(config)
        trajectory = make_trajectory(rewards=rewards, calls_per_step=calls_per_step)
        result = term.compute(trajectory)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_tool_calls_count_toward_cost(self, make_trajectory):
        """Tool calls are summed alongside LLM calls in the denominator."""
        from trajectory_aware_gym.adapters.trajectory_logger import ToolCall

        trajectory = make_trajectory(rewards=[0.0, 1.0], calls_per_step=1)
        # Add a tool call to step 0; total calls now = (1+1) + (1+0) = 3
        trajectory.steps[0].tool_calls.append(
            ToolCall(tool_name="exec", tool_input="x", tool_output="y", success=True)
        )

        config = _cfg(max_steps=10, call_budget_per_step=4)
        term = CallEfficiencyBonusTerm(config)
        # 3 total calls / (10 * 4 = 40) -> 1 - 3/40 = 0.925
        assert term.compute(trajectory) == pytest.approx(1.0 - 3 / 40, abs=1e-9)


class TestNormalizedProgressTerm:
    """Tests for the reward-trend progress term (from objective-profile design)."""

    def test_empty_trajectory(self, make_trajectory):
        trajectory = make_trajectory(rewards=[])
        term = NormalizedProgressTerm()
        assert term.compute(trajectory) == 0.0

    def test_name(self):
        term = NormalizedProgressTerm()
        assert term.name == "normalized_progress"

    @pytest.mark.parametrize(
        ("rewards", "expected"),
        [
            # Single positive step -> 0.5
            ([1.0], 0.5),
            # Single zero step -> 0.0
            ([0.0], 0.0),
            # Single negative step -> 0.0
            ([-1.0], 0.0),
            # Strictly increasing: 2/2 = 1.0
            ([0.0, 0.5, 1.0], 1.0),
            # Strictly decreasing: 0/2 = 0.0
            ([1.0, 0.5, 0.0], 0.0),
            # Flat (non-decreasing): 2/2 = 1.0
            ([0.5, 0.5, 0.5], 1.0),
            # Mixed: [0.0, 0.5, 0.3, 0.8] -> up, down, up -> 2/3
            ([0.0, 0.5, 0.3, 0.8], pytest.approx(2 / 3)),
            # Two steps, increase: 1/1 = 1.0
            ([0.0, 1.0], 1.0),
            # Two steps, decrease: 0/1 = 0.0
            ([1.0, 0.0], 0.0),
        ],
    )
    def test_computation(self, make_trajectory, rewards, expected):
        term = NormalizedProgressTerm()
        trajectory = make_trajectory(rewards=rewards)
        assert term.compute(trajectory) == pytest.approx(expected, abs=1e-9)

    def test_conforms_to_fitness_term_protocol(self):
        from trajectory_aware_gym.fitness.types import FitnessTerm

        assert isinstance(NormalizedProgressTerm(), FitnessTerm)


class TestActionStabilityTerm:
    """Tests for the action stability term (from objective-profile design)."""

    def test_empty_trajectory(self, make_trajectory):
        trajectory = make_trajectory(rewards=[], actions=[])
        term = ActionStabilityTerm()
        assert term.compute(trajectory) == 1.0

    def test_single_step(self, make_trajectory):
        trajectory = make_trajectory(rewards=[1.0], actions=["a"])
        term = ActionStabilityTerm()
        assert term.compute(trajectory) == 1.0

    def test_name(self):
        term = ActionStabilityTerm()
        assert term.name == "action_stability"

    @pytest.mark.parametrize(
        ("actions", "expected"),
        [
            # All unique: no repetition, no oscillation -> 1.0
            (["a", "b", "c", "d"], 1.0),
            # All identical: repeat_ratio=1.0, oscillation_ratio=0.0
            # penalty = 0.7*1.0 + 0.3*0.0 = 0.7 -> stability = 0.3
            (["a", "a", "a", "a"], 0.3),
            # Pure oscillation: a,b,a,b
            # repeat_ratio = 0/3 = 0.0
            # oscillation: i=2 a==a!=b yes, i=3 b==b!=a yes -> 2/2 = 1.0
            # penalty = 0.7*0 + 0.3*1.0 = 0.3 -> stability = 0.7
            (["a", "b", "a", "b"], 0.7),
            # Mixed: a,a,b,a
            # repeat_ratio: 1 (a==a at i=1) / 3 = 1/3
            # oscillation: i=2 b!=a (no), i=3 a==a!=b (yes) -> 1/2
            # penalty = 0.7*(1/3) + 0.3*(1/2) = 0.2333 + 0.15 = 0.3833
            # stability = 1 - 0.3833 = 0.6167
            (["a", "a", "b", "a"], pytest.approx(1.0 - (0.7 * (1 / 3) + 0.3 * (1 / 2)))),
            # Two unique steps: no repetition -> 1.0
            (["a", "b"], 1.0),
            # Two identical steps: repeat_ratio=1.0, no oscillation possible
            # penalty = 0.7*1.0 = 0.7 -> stability = 0.3
            (["a", "a"], 0.3),
        ],
    )
    def test_computation(self, make_trajectory, actions, expected):
        rewards = [0.0] * (len(actions) - 1) + [1.0] if actions else []
        term = ActionStabilityTerm()
        trajectory = make_trajectory(rewards=rewards, actions=actions)
        assert term.compute(trajectory) == pytest.approx(expected, abs=1e-9)

    def test_conforms_to_fitness_term_protocol(self):
        from trajectory_aware_gym.fitness.types import FitnessTerm

        assert isinstance(ActionStabilityTerm(), FitnessTerm)
