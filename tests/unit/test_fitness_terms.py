"""Tests for individual fitness term implementations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog, TrajectoryStep
from trajectory_aware_gym.config import FitnessConfig
from trajectory_aware_gym.fitness.terms import (
    ActionStabilityTerm,
    DiscountedReturnTerm,
    LoopDetectionPenaltyTerm,
    NormalizedProgressTerm,
    StepEfficiencyBonusTerm,
)


@pytest.fixture
def make_trajectory():
    """Factory for building TrajectoryLog objects with controlled properties."""

    def _make(
        rewards: list[float],
        actions: list[str] | None = None,
        environment_id: str = "test-env",
    ) -> TrajectoryLog:
        now = datetime.now(UTC)
        n = len(rewards)
        if actions is None:
            actions = [f"action_{i}" for i in range(n)]
        steps = [
            TrajectoryStep(
                step_index=i + 1,
                action=actions[i],
                observation=f"obs_{i}",
                reward=rewards[i],
                terminated=(i == n - 1),
                truncated=False,
            )
            for i in range(n)
        ]
        return TrajectoryLog(
            environment_id=environment_id,
            seed=42,
            started_at=now,
            finished_at=now + timedelta(seconds=max(n, 1)),
            initial_observation="start",
            steps=steps,
            total_reward=sum(rewards),
        )

    return _make


class TestDiscountedReturnTerm:
    """Tests for the reverse-time discounted return (Eq. 3.1)."""

    def test_empty_trajectory(self, make_trajectory):
        trajectory = make_trajectory(rewards=[])
        config = FitnessConfig(fitness_gamma=0.99, fitness_lambda=0.1, _env_file=None)
        term = DiscountedReturnTerm(config)
        assert term.compute(trajectory) == 0.0

    def test_name(self):
        term = DiscountedReturnTerm(FitnessConfig(_env_file=None))
        assert term.name == "discounted_return"

    @pytest.mark.parametrize(
        ("rewards", "gamma", "lam", "expected"),
        [
            # Single successful step: γ^0 * 1.0 + 0.1 * 1.0 = 1.1
            ([1.0], 0.99, 0.1, 1.1),
            # Single failed step: 0.0 + 0.1 * 0.0 = 0.0
            ([0.0], 0.99, 0.1, 0.0),
            # Two steps, success: (γ^1 + γ^0) * 1.0 + λ * (0.0 + 1.0)
            # = (0.5 + 1.0) + 0.1 * 1.0 = 1.6
            ([0.0, 1.0], 0.5, 0.1, 1.6),
            # γ=0: only final step counts: 0^1 + 0^0 = 0 + 1 = 1.0
            ([0.0, 1.0], 0.0, 0.0, 1.0),
            # γ=1: uniform weighting: 1^1 + 1^0 = 1 + 1 = 2.0
            ([0.0, 1.0], 1.0, 0.0, 2.0),
            # Negative final reward (failure): main=0, aux=λ*(0.5 + -1.0)=-0.05
            ([0.5, -1.0], 0.99, 0.1, -0.05),
            # Three steps success, γ=0.5, λ=0:
            # 0.5^2 + 0.5^1 + 0.5^0 = 0.25 + 0.5 + 1.0 = 1.75
            ([0.0, 0.0, 1.0], 0.5, 0.0, 1.75),
            # λ=0 disables auxiliary term
            ([0.0, 1.0], 0.5, 0.0, 1.5),
            # Only auxiliary term (γ=0, two steps, success):
            # main = 0^1 * 1 + 0^0 * 1 = 0 + 1 = 1.0, aux = 0.5 * (0.0+1.0) = 0.5
            ([0.0, 1.0], 0.0, 0.5, 1.5),
        ],
    )
    def test_computation(self, make_trajectory, rewards, gamma, lam, expected):
        config = FitnessConfig(fitness_gamma=gamma, fitness_lambda=lam, _env_file=None)
        term = DiscountedReturnTerm(config)
        trajectory = make_trajectory(rewards=rewards)
        result = term.compute(trajectory)
        assert result == pytest.approx(expected, abs=1e-9)


class TestLoopDetectionPenaltyTerm:
    """Tests for the loop detection penalty."""

    def test_empty_trajectory(self, make_trajectory):
        trajectory = make_trajectory(rewards=[], actions=[])
        term = LoopDetectionPenaltyTerm(FitnessConfig(_env_file=None))
        assert term.compute(trajectory) == 0.0

    def test_single_step(self, make_trajectory):
        trajectory = make_trajectory(rewards=[1.0], actions=["a"])
        term = LoopDetectionPenaltyTerm(FitnessConfig(_env_file=None))
        assert term.compute(trajectory) == 0.0

    def test_name(self):
        term = LoopDetectionPenaltyTerm(FitnessConfig(_env_file=None))
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
        config = FitnessConfig(fitness_loop_window=window, _env_file=None)
        term = LoopDetectionPenaltyTerm(config)
        trajectory = make_trajectory(rewards=rewards, actions=actions)
        result = term.compute(trajectory)
        assert result == pytest.approx(expected, abs=1e-9)


class TestStepEfficiencyBonusTerm:
    """Tests for the step efficiency bonus."""

    def test_empty_trajectory(self, make_trajectory):
        trajectory = make_trajectory(rewards=[])
        term = StepEfficiencyBonusTerm(FitnessConfig(_env_file=None))
        assert term.compute(trajectory) == 0.0

    def test_name(self):
        term = StepEfficiencyBonusTerm(FitnessConfig(_env_file=None))
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
        config = FitnessConfig(fitness_max_steps=max_steps, _env_file=None)
        term = StepEfficiencyBonusTerm(config)
        trajectory = make_trajectory(rewards=rewards)
        result = term.compute(trajectory)
        assert result == pytest.approx(expected, abs=1e-9)


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
