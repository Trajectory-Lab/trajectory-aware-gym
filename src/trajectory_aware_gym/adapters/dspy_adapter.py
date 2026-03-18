"""Bridge between trajectory-aware fitness and DSPy metric interface."""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore[import-untyped]
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback  # type: ignore[import-untyped]

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog
from trajectory_aware_gym.config.core import FitnessModel
from trajectory_aware_gym.fitness.composite import CompositeFitness
from trajectory_aware_gym.fitness.types import FitnessResult


class TrajectoryFitnessMetric:
    """DSPy-compatible metric wrapping trajectory-aware fitness.

    Expects ``prediction`` to carry a ``trajectory`` attribute containing
    a TrajectoryLog instance (set by the GEM-DSPy adapter during episode
    execution).

    Supports both DSPy Evaluate (3-arg) and GEPA (5-arg) call conventions::

        # dspy.Evaluate style
        metric(example, prediction, trace)

        # dspy.GEPA style
        metric(gold, pred, trace, pred_name, pred_trace)

    When ``return_feedback=True``, returns ``dspy.Prediction(score=...,
    feedback=...)`` (satisfying GEPA's ``ScoreWithFeedback`` protocol).
    Otherwise returns a plain float for ``dspy.Evaluate``.
    """

    def __init__(
        self,
        config: FitnessModel | None = None,
        fitness: CompositeFitness | None = None,
        *,
        return_feedback: bool = True,
    ) -> None:
        if config is None:
            from trajectory_aware_gym.config import settings

            config = settings.fitness
        self._config = config
        self._fitness = fitness or CompositeFitness(self._config)
        self._return_feedback = return_feedback

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> float | ScoreWithFeedback:
        trajectory: TrajectoryLog | None = getattr(prediction, "trajectory", None)

        if trajectory is None:
            if self._return_feedback:
                return ScoreWithFeedback(
                    score=0.0,
                    feedback="No trajectory data available for fitness evaluation.",
                )
            return 0.0

        result = self._fitness.evaluate(trajectory)
        bounded_score = self._bound_score(result.score)

        if self._return_feedback:
            feedback = self._format_feedback(result, bounded_score)
            return ScoreWithFeedback(score=bounded_score, feedback=feedback)

        return bounded_score

    def _format_feedback(self, result: FitnessResult, bounded_score: float) -> str:
        lines = [
            f"Fitness score: {bounded_score:.4f}",
            f"Raw composite score: {result.score:.4f}",
        ]
        lines.append(f"Trajectory length: {result.trajectory_length} steps")
        for item in result.breakdown:
            lines.append(
                f"  {item.term_name}: {item.raw_value:.4f} "
                f"(weight={item.weight:.2f}, contribution={item.weighted_value:.4f})"
            )
        return "\n".join(lines)

    @staticmethod
    def _bound_score(score: float) -> float:
        return max(0.0, min(1.0, score))
