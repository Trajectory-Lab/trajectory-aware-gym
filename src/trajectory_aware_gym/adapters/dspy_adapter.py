"""Bridge between trajectory-aware fitness and DSPy metric interface."""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore[import-untyped]

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog
from trajectory_aware_gym.fitness.composite import CompositeFitness
from trajectory_aware_gym.fitness.config import FitnessConfig
from trajectory_aware_gym.fitness.types import FitnessResult


class TrajectoryFitnessMetric:
    """DSPy-compatible metric wrapping trajectory-aware fitness.

    Expects ``prediction`` to carry a ``trajectory`` attribute containing
    a TrajectoryLog instance (set by the GEM-DSPy adapter during episode
    execution).

    When ``return_feedback=True``, returns ``dspy.Prediction(score=...,
    feedback=...)`` for GEPA reflection. Otherwise returns a plain float
    for ``dspy.Evaluate``.
    """

    def __init__(
        self,
        config: FitnessConfig | None = None,
        fitness: CompositeFitness | None = None,
        *,
        return_feedback: bool = True,
    ) -> None:
        self._config = config or FitnessConfig()
        self._fitness = fitness or CompositeFitness(self._config)
        self._return_feedback = return_feedback

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any = None,
        **kwargs: Any,
    ) -> float | dspy.Prediction:
        trajectory: TrajectoryLog | None = getattr(prediction, "trajectory", None)

        if trajectory is None:
            if self._return_feedback:
                return dspy.Prediction(
                    score=0.0,
                    feedback="No trajectory data available for fitness evaluation.",
                )
            return 0.0

        result = self._fitness.evaluate(trajectory)

        if self._return_feedback:
            feedback = self._format_feedback(result)
            return dspy.Prediction(score=result.score, feedback=feedback)

        return result.score

    def _format_feedback(self, result: FitnessResult) -> str:
        lines = [f"Fitness score: {result.score:.4f}"]
        lines.append(f"Trajectory length: {result.trajectory_length} steps")
        for item in result.breakdown:
            lines.append(
                f"  {item.term_name}: {item.raw_value:.4f} "
                f"(weight={item.weight:.2f}, contribution={item.weighted_value:.4f})"
            )
        return "\n".join(lines)
