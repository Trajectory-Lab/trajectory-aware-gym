"""Trajectory-aware fitness scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog


@dataclass(frozen=True)
class TrajectoryFitnessConfig:
    """Configuration for trajectory-aware fitness scoring."""

    discount_factor: float = 0.95
    success_weight: float = 1.0
    auxiliary_weight: float = 1.0
    loop_penalty_weight: float = 0.25
    efficiency_bonus_weight: float = 0.25
    max_steps: int = 50


@dataclass(frozen=True)
class TrajectoryFitnessBreakdown:
    """Detailed decomposition of trajectory fitness components."""

    final_fitness: float
    discounted_success_component: float
    auxiliary_reward_component: float
    loop_penalty_component: float
    step_efficiency_component: float
    success_indicator: float
    repeated_action_count: int


def _compute_discounted_success_component(
    *,
    num_steps: int,
    success_indicator: float,
    discount_factor: float,
    success_weight: float,
) -> float:
    if num_steps == 0 or success_indicator == 0:
        return 0.0

    return sum(
        (discount_factor ** (num_steps - step_index)) * success_weight * success_indicator
        for step_index in range(1, num_steps + 1)
    )


def _count_repeated_actions(trajectory: TrajectoryLog) -> int:
    action_counts: dict[str, int] = {}
    for step in trajectory.steps:
        action_counts[step.action] = action_counts.get(step.action, 0) + 1

    return sum(max(0, count - 1) for count in action_counts.values())


def score_trajectory(
    trajectory: TrajectoryLog,
    config: TrajectoryFitnessConfig | None = None,
) -> TrajectoryFitnessBreakdown:
    """Compute trajectory-aware fitness from per-step transitions and outcome signals."""
    settings = config or TrajectoryFitnessConfig()
    num_steps = len(trajectory.steps)

    success_indicator = float(
        bool(trajectory.steps)
        and trajectory.steps[-1].terminated
        and not trajectory.steps[-1].truncated
        and trajectory.total_reward > 0
    )

    discounted_success_component = _compute_discounted_success_component(
        num_steps=num_steps,
        success_indicator=success_indicator,
        discount_factor=settings.discount_factor,
        success_weight=settings.success_weight,
    )

    auxiliary_reward_component = settings.auxiliary_weight * trajectory.total_reward
    repeated_action_count = _count_repeated_actions(trajectory)
    loop_penalty_component = settings.loop_penalty_weight * repeated_action_count

    step_efficiency_component = 0.0
    if settings.max_steps > 0 and success_indicator > 0:
        efficiency_ratio = max(0.0, 1 - (num_steps / settings.max_steps))
        step_efficiency_component = settings.efficiency_bonus_weight * efficiency_ratio

    final_fitness = (
        discounted_success_component
        + auxiliary_reward_component
        + step_efficiency_component
        - loop_penalty_component
    )

    return TrajectoryFitnessBreakdown(
        final_fitness=final_fitness,
        discounted_success_component=discounted_success_component,
        auxiliary_reward_component=auxiliary_reward_component,
        loop_penalty_component=loop_penalty_component,
        step_efficiency_component=step_efficiency_component,
        success_indicator=success_indicator,
        repeated_action_count=repeated_action_count,
    )
