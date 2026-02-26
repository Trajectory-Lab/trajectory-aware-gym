"""Individual fitness term implementations."""

from __future__ import annotations

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog
from trajectory_aware_gym.config.core import FitnessModel


class DiscountedReturnTerm:
    """Reverse-time discounted return (Equation 3.1).

    F_main(τ) = Σ_{t=0}^{T-1} γ^(T-1-t) · w_t · R_T  +  λ · Σ_{t=0}^{T-1} r_t

    γ^(T-1-t) gives full weight to the final step and discounts earlier steps.
    R_T is a binary indicator (1 if final reward > 0, else 0).
    r_t are the actual per-step rewards from the environment.
    """

    def __init__(self, config: FitnessModel | None = None) -> None:
        if config is None:
            from trajectory_aware_gym.config import settings

            config = settings.fitness
        self._config = config

    @property
    def name(self) -> str:
        return "discounted_return"

    def compute(self, trajectory: TrajectoryLog) -> float:
        steps = trajectory.steps
        if not steps:
            return 0.0

        total_steps = len(steps)
        gamma = self._config.gamma
        lam = self._config.lambda_

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

    def __init__(self, config: FitnessModel | None = None) -> None:
        if config is None:
            from trajectory_aware_gym.config import settings

            config = settings.fitness
        self._config = config

    @property
    def name(self) -> str:
        return "loop_detection_penalty"

    def compute(self, trajectory: TrajectoryLog) -> float:
        steps = trajectory.steps
        if len(steps) < 2:
            return 0.0

        window = self._config.loop_window
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

    def __init__(self, config: FitnessModel | None = None) -> None:
        if config is None:
            from trajectory_aware_gym.config import settings

            config = settings.fitness
        self._config = config

    @property
    def name(self) -> str:
        return "step_efficiency_bonus"

    def compute(self, trajectory: TrajectoryLog) -> float:
        steps = trajectory.steps
        if not steps:
            return 0.0

        if steps[-1].reward <= 0:
            return 0.0

        max_steps = self._config.max_steps
        actual_steps = len(steps)
        efficiency = 1.0 - (actual_steps / max_steps)

        return max(0.0, efficiency)


# ---------------------------------------------------------------------------
# Experimental terms adapted from the objective-profile design (PR #118).
# These are disabled by default and not included in CompositeFitness.
# Enable by adding to the composite's term list with a non-zero weight
# once validated in ablation experiments.
# See also: docs/fitness_objective_profile.md
# ---------------------------------------------------------------------------


class NormalizedProgressTerm:
    """Reward-trend quality measured as the fraction of non-decreasing steps.

    Captures whether per-turn rewards are generally improving over the
    trajectory. Returns a value in [0, 1] where 1.0 means every
    consecutive reward was >= the previous one.

    For single-step trajectories, returns 0.5 if the reward is positive,
    0.0 otherwise. Empty trajectories return 0.0.

    Adapted from PR #118's ``compute_progress_component``.
    """

    @property
    def name(self) -> str:
        return "normalized_progress"

    def compute(self, trajectory: TrajectoryLog) -> float:
        steps = trajectory.steps
        if not steps:
            return 0.0

        rewards = [step.reward for step in steps]
        if len(rewards) < 2:
            return 0.5 if rewards[0] > 0 else 0.0

        non_decreasing = sum(
            1 for prev, curr in zip(rewards, rewards[1:], strict=False) if curr >= prev
        )
        return non_decreasing / (len(rewards) - 1)


class ActionStabilityTerm:
    """Penalty for repetitive and oscillating action patterns.

    Combines two signals:
    - Consecutive repetition: fraction of steps where action[i] == action[i-1]
    - Oscillation: fraction of steps where action[i] == action[i-2] != action[i-1]

    Returns a value in [0, 1] where 1.0 means perfectly stable (no
    repetition or oscillation) and 0.0 means maximally unstable.

    Adapted from PR #118's ``compute_stability_component``.
    """

    REPETITION_WEIGHT = 0.7
    OSCILLATION_WEIGHT = 0.3

    @property
    def name(self) -> str:
        return "action_stability"

    def compute(self, trajectory: TrajectoryLog) -> float:
        actions = [step.action for step in trajectory.steps]
        if len(actions) < 2:
            return 1.0

        repeated = sum(1 for prev, curr in zip(actions, actions[1:], strict=False) if prev == curr)
        repeat_ratio = repeated / (len(actions) - 1)

        oscillations = 0
        for i in range(2, len(actions)):
            if actions[i] == actions[i - 2] and actions[i] != actions[i - 1]:
                oscillations += 1
        oscillation_ratio = oscillations / max(1, len(actions) - 2)

        penalty = (
            self.REPETITION_WEIGHT * repeat_ratio + self.OSCILLATION_WEIGHT * oscillation_ratio
        )
        return max(0.0, min(1.0, 1.0 - penalty))
