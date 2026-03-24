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

        # Normalization range: [0, max] so that failed trajectories (raw ≤ 0)
        # always map to 0.  Only successful episodes earn positive fitness.
        #   DiscountedReturnTerm:      [0, 1]  weight = 1.0 (always)
        #   LoopDetectionPenaltyTerm:  [-1, 0] weight = loop_penalty_weight
        #   StepEfficiencyBonusTerm:   [0, 1]  weight = step_efficiency_weight
        eff_w = self._config.step_efficiency_weight
        self._score_max = 1.0 + eff_w  # best case: 1 + 0 + eff_w*1

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
        score = self._normalize(result.score)

        if self._return_feedback:
            feedback = self._format_feedback(result, score)
            return ScoreWithFeedback(score=score, feedback=feedback)

        return score

    def _normalize(self, raw_score: float) -> float:
        """Clamp negative raw scores to 0 then rescale to [0, 1]."""
        if self._score_max <= 0:
            return 0.0
        return max(0.0, raw_score) / self._score_max

    def _format_feedback(self, result: FitnessResult, normalized_score: float) -> str:
        lines = [
            f"Fitness score: {normalized_score:.4f} (raw composite: {result.score:.4f})",
        ]
        lines.append(f"Trajectory length: {result.trajectory_length} steps")
        for item in result.breakdown:
            lines.append(
                f"  {item.term_name}: {item.raw_value:.4f} "
                f"(weight={item.weight:.2f}, contribution={item.weighted_value:.4f})"
            )
        return "\n".join(lines)
