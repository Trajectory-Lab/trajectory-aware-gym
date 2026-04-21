"""Bridge between trajectory-aware fitness and DSPy metric interface."""

from __future__ import annotations

import threading
from typing import Any

import dspy  # type: ignore[import-untyped]
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback  # type: ignore[import-untyped]

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog
from trajectory_aware_gym.config.core import FitnessModel
from trajectory_aware_gym.fitness.composite import CompositeFitness
from trajectory_aware_gym.fitness.types import FitnessResult


def _normalize_instructions(value: str | None) -> str:
    """Return the canonical form ``dspy.Signature`` would store.

    ``Signature.with_instructions`` strips trailing whitespace from the stored
    value, so normalising to ``rstrip`` on both record and query keeps the
    comparison robust against YAML ``|`` block-scalars (trailing ``\\n``) and
    stray trailing spaces in seed prompts.
    """
    if value is None:
        return ""
    return value.rstrip()


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
        #   CallEfficiencyBonusTerm:   [0, 1]  weight = call_efficiency_weight
        step_w = self._config.step_efficiency_weight
        call_w = self._config.call_efficiency_weight
        self._score_max = 1.0 + step_w + call_w  # best case: 1 + 0 + step_w*1 + call_w*1

        # Sidecar outcome log: each call appends (seed, instructions, status).
        # The runner reads this after GEPA finishes to count how many of the
        # baseline vs best-program validation episodes were actually scorable.
        self._outcomes: list[dict[str, Any]] = []
        self._outcomes_lock = threading.Lock()

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> float | ScoreWithFeedback:
        trajectory: TrajectoryLog | None = getattr(prediction, "trajectory", None)
        forward_status = str(getattr(prediction, "status", "ok"))
        instructions = getattr(prediction, "instructions", None)

        if trajectory is None:
            outcome_status = forward_status if forward_status != "ok" else "no_trajectory"
            self._record_outcome(example, instructions, outcome_status)
            if self._return_feedback:
                return ScoreWithFeedback(
                    score=0.0,
                    feedback="No trajectory data available for fitness evaluation.",
                )
            return 0.0

        result = self._fitness.evaluate(trajectory)
        score = self._normalize(result.score)
        self._record_outcome(example, instructions, forward_status)

        if self._return_feedback:
            feedback = self._format_feedback(result, score)
            return ScoreWithFeedback(score=score, feedback=feedback)

        return score

    def _record_outcome(
        self,
        example: dspy.Example,
        instructions: str | None,
        status: str,
    ) -> None:
        seed = getattr(example, "seed", None)
        normalized = _normalize_instructions(instructions)
        with self._outcomes_lock:
            self._outcomes.append(
                {
                    "seed": seed,
                    "instructions": normalized,
                    "status": status,
                }
            )

    def reset_outcomes(self) -> None:
        """Clear the outcome sidecar. Call between replications that reuse the metric."""
        with self._outcomes_lock:
            self._outcomes.clear()

    def scorable_count(
        self,
        instructions: str,
        *,
        seeds: frozenset[int] | set[int] | None = None,
    ) -> int:
        """Count unique seeds for which this candidate produced a scorable episode.

        "Scorable" = the forward call completed without an infra exception and
        a trajectory was attached. We deduplicate by seed because GEPA may call
        the metric multiple times for the same (candidate, seed) pair; the most
        recent status wins so transient failures on earlier attempts do not
        mask later successes.

        Instructions strings are normalized (``rstrip``) on both sides because
        ``dspy.Signature.with_instructions`` strips trailing whitespace from
        the stored signature — without normalization, a YAML ``|`` block-scalar
        seed prompt (trailing ``\\n``) would never match the stored value.
        """
        target = _normalize_instructions(instructions)
        latest: dict[int, str] = {}
        with self._outcomes_lock:
            rows = list(self._outcomes)
        for row in rows:
            if row["instructions"] != target:
                continue
            seed = row["seed"]
            if not isinstance(seed, int):
                continue
            if seeds is not None and seed not in seeds:
                continue
            latest[seed] = row["status"]
        return sum(1 for status in latest.values() if status == "ok")

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
