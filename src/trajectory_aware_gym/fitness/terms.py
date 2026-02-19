"""Individual fitness term implementations."""

from __future__ import annotations

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog
from trajectory_aware_gym.fitness.config import FitnessConfig


class DiscountedReturnTerm:
    """Reverse-time discounted return (Equation 3.1).

    F_main(τ) = Σ_{t=0}^{T-1} γ^(T-1-t) · w_t · R_T  +  λ · Σ_{t=0}^{T-1} r_t

    γ^(T-1-t) gives full weight to the final step and discounts earlier steps.
    R_T is a binary indicator (1 if final reward > 0, else 0).
    r_t are the actual per-step rewards from the environment.
    """

    def __init__(self, config: FitnessConfig | None = None) -> None:
        self._config = config or FitnessConfig()

    @property
    def name(self) -> str:
        return "discounted_return"

    def compute(self, trajectory: TrajectoryLog) -> float:
        steps = trajectory.steps
        if not steps:
            return 0.0

        total_steps = len(steps)
        gamma = self._config.fitness_gamma
        lam = self._config.fitness_lambda

        is_success = 1.0 if steps[-1].reward > 0 else 0.0

        main_term = sum(
            gamma ** (total_steps - 1 - t) * 1.0 * is_success for t in range(total_steps)
        )

        auxiliary_term = sum(step.reward for step in steps)

        return main_term + lam * auxiliary_term


class LoopDetectionPenaltyTerm:
    """Penalty for repeated identical actions within a sliding window.

    Detects when an action at step i matches any action in the
    preceding `window` steps. Returns a negative value proportional
    to the fraction of steps exhibiting loops.
    """

    def __init__(self, config: FitnessConfig | None = None) -> None:
        self._config = config or FitnessConfig()

    @property
    def name(self) -> str:
        return "loop_detection_penalty"

    def compute(self, trajectory: TrajectoryLog) -> float:
        steps = trajectory.steps
        if len(steps) < 2:
            return 0.0

        window = self._config.fitness_loop_window
        loop_count = 0

        for i in range(1, len(steps)):
            lookback_start = max(0, i - window)
            window_actions = [steps[j].action for j in range(lookback_start, i)]
            if steps[i].action in window_actions:
                loop_count += 1

        max_possible_loops = len(steps) - 1
        loop_ratio = loop_count / max_possible_loops

        return -loop_ratio


class StepEfficiencyBonusTerm:
    """Bonus for shorter successful trajectories.

    efficiency = 1.0 - (actual_steps / max_steps), clamped to [0, 1].
    Only awards a bonus when the trajectory ends with a positive final reward.
    """

    def __init__(self, config: FitnessConfig | None = None) -> None:
        self._config = config or FitnessConfig()

    @property
    def name(self) -> str:
        return "step_efficiency_bonus"

    def compute(self, trajectory: TrajectoryLog) -> float:
        steps = trajectory.steps
        if not steps:
            return 0.0

        if steps[-1].reward <= 0:
            return 0.0

        max_steps = self._config.fitness_max_steps
        actual_steps = len(steps)
        efficiency = 1.0 - (actual_steps / max_steps)

        return max(0.0, efficiency)
