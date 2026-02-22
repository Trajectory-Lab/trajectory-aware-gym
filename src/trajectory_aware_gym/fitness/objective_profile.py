"""Profile-based trajectory-aware fitness scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class FitnessWeights:
    """Objective weights for profile-based trajectory fitness."""

    outcome: float = 0.40
    progress: float = 0.25
    efficiency: float = 0.20
    stability: float = 0.15


@dataclass(frozen=True)
class ObjectiveProfile:
    """Normalized objective components in [0, 1]."""

    outcome: float
    progress: float
    efficiency: float
    stability: float


@dataclass(frozen=True)
class TrajectoryFitnessResult:
    """Scalar + structured trajectory fitness output."""

    score: float
    profile: ObjectiveProfile
    weights: FitnessWeights
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


def compute_outcome_component(trajectory: TrajectoryLog) -> float:
    """Compute normalized outcome signal from terminal and reward evidence."""
    has_steps = bool(trajectory.steps)
    terminated = has_steps and trajectory.steps[-1].terminated
    truncated = has_steps and trajectory.steps[-1].truncated
    success_bonus = 0.5 if terminated and not truncated and trajectory.total_reward > 0 else 0.0
    reward_signal = _clip01(trajectory.total_reward)
    return _clip01(success_bonus + (0.5 * reward_signal))


def compute_progress_component(trajectory: TrajectoryLog) -> float:
    """Compute normalized progress quality from reward deltas over time."""
    rewards = [step.reward for step in trajectory.steps]
    if len(rewards) < 2:
        return _clip01(0.5 * (1.0 if rewards and rewards[0] > 0 else 0.0))

    positive_moves = sum(1 for left, right in zip(rewards, rewards[1:], strict=False) if right >= left)
    return _clip01(positive_moves / (len(rewards) - 1))


def compute_efficiency_component(trajectory: TrajectoryLog, *, max_steps: int) -> float:
    """Compute step efficiency where shorter successful trajectories score higher."""
    if max_steps <= 0:
        return 0.0
    steps = len(trajectory.steps)
    return _clip01(1.0 - (steps / max_steps))


def compute_stability_component(trajectory: TrajectoryLog) -> float:
    """Compute stability by penalizing repetition and oscillation in actions."""
    actions = [step.action for step in trajectory.steps]
    if len(actions) < 2:
        return 1.0

    repeated = sum(1 for left, right in zip(actions, actions[1:], strict=False) if left == right)
    repeated_ratio = repeated / (len(actions) - 1)

    oscillations = 0
    for index in range(2, len(actions)):
        if actions[index] == actions[index - 2] and actions[index] != actions[index - 1]:
            oscillations += 1
    oscillation_ratio = oscillations / max(1, len(actions) - 2)

    penalty = (0.7 * repeated_ratio) + (0.3 * oscillation_ratio)
    return _clip01(1.0 - penalty)


def score_trajectory_profile(
    trajectory: TrajectoryLog,
    *,
    weights: FitnessWeights | None = None,
    max_steps: int = 50,
) -> TrajectoryFitnessResult:
    """Score trajectory with normalized objective profile and diagnostics."""
    objective_weights = weights or FitnessWeights()

    outcome = compute_outcome_component(trajectory)
    progress = compute_progress_component(trajectory)
    efficiency = compute_efficiency_component(trajectory, max_steps=max_steps)
    stability = compute_stability_component(trajectory)

    profile = ObjectiveProfile(
        outcome=outcome,
        progress=progress,
        efficiency=efficiency,
        stability=stability,
    )

    score = (
        objective_weights.outcome * profile.outcome
        + objective_weights.progress * profile.progress
        + objective_weights.efficiency * profile.efficiency
        + objective_weights.stability * profile.stability
    )

    diagnostics: list[str] = []
    if trajectory.steps and trajectory.steps[-1].truncated:
        diagnostics.append("trajectory_truncated")
    if profile.progress < 0.4:
        diagnostics.append("low_progress")
    if profile.stability < 0.5:
        diagnostics.append("unstable_action_pattern")
    if profile.outcome < 0.5:
        diagnostics.append("weak_outcome_signal")

    return TrajectoryFitnessResult(
        score=score,
        profile=profile,
        weights=objective_weights,
        diagnostics=tuple(diagnostics),
    )
