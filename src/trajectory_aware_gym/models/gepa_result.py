"""Structured result model for GEPA optimization runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def accuracy_from_subscores(subscores: dict) -> float:
    """Derive accuracy from GEPA per-example fitness scores.

    After normalization, score > 0 ⟺ the episode succeeded (total_reward > 0),
    so accuracy can be read directly from the scores GEPA already computed.
    """
    if not subscores:
        return 0.0
    return sum(1 for s in subscores.values() if s > 0) / len(subscores)


def _normalize_validation_subscores(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, float] = {}
    for key, score in value.items():
        if isinstance(score, int | float) and not isinstance(score, bool):
            out[str(key)] = float(score)
    return out


class GEPARunResult(BaseModel, frozen=True):
    """Structured summary of a GEPA optimization run."""

    baseline_fitness: float
    final_fitness: float
    baseline_accuracy: float
    final_accuracy: float
    best_program_index: int
    optimized_instructions: str

    @classmethod
    def from_module(cls, module: Any, seed_prompt: str) -> GEPARunResult | None:
        """Extract results from an optimized GEMSolverModule.

        Returns None if the module lacks GEPA detailed_results.
        """
        detailed = getattr(module, "detailed_results", None)
        if detailed is None:
            return None

        best_idx = detailed.best_idx
        return cls(
            baseline_fitness=detailed.val_aggregate_scores[0],
            final_fitness=detailed.val_aggregate_scores[best_idx],
            baseline_accuracy=accuracy_from_subscores(detailed.val_subscores[0]),
            final_accuracy=accuracy_from_subscores(detailed.val_subscores[best_idx]),
            best_program_index=best_idx,
            optimized_instructions=getattr(module, "instructions", seed_prompt),
        )


def build_validation_audit_from_detailed(
    optimized_module: Any,
    *,
    episodes: int,
) -> dict[str, Any] | None:
    """Extract a validation-audit payload from an optimized module's `detailed_results`.

    Returns `None` when the module has no GEPA detail or when the detail is
    malformed in a way that makes baseline vs. best comparison unsound.
    """
    detailed = getattr(optimized_module, "detailed_results", None)
    if detailed is None:
        return None

    val_scores = getattr(detailed, "val_aggregate_scores", None)
    val_subscores = getattr(detailed, "val_subscores", None)
    best_idx = getattr(detailed, "best_idx", None)
    discovery = getattr(detailed, "discovery_eval_counts", None)
    if not isinstance(val_scores, list) or not isinstance(val_subscores, list):
        return None
    if not val_scores or not val_subscores or not isinstance(best_idx, int):
        return None
    if best_idx < 0 or best_idx >= len(val_scores) or best_idx >= len(val_subscores):
        return None

    baseline_subscores = _normalize_validation_subscores(val_subscores[0])
    optimized_subscores = _normalize_validation_subscores(val_subscores[best_idx])
    baseline_accuracy = (
        accuracy_from_subscores(baseline_subscores) if baseline_subscores is not None else None
    )
    optimized_accuracy = (
        accuracy_from_subscores(optimized_subscores) if optimized_subscores is not None else None
    )

    def discovery_count(index: int) -> int | None:
        if isinstance(discovery, list) and index < len(discovery):
            value = discovery[index]
            if isinstance(value, int | float) and not isinstance(value, bool):
                return int(value)
        return None

    return {
        "source": "gepa_detailed_results",
        "episodes": episodes,
        "candidate_count": len(val_scores),
        "best_program_index": best_idx,
        "baseline": {
            "program_index": 0,
            "val_aggregate_score": float(val_scores[0]),
            "accuracy": baseline_accuracy,
            "correct": round(baseline_accuracy * episodes)
            if isinstance(baseline_accuracy, int | float)
            else None,
            "discovery_eval_count": discovery_count(0),
            "subscores": baseline_subscores,
        },
        "optimized": {
            "program_index": best_idx,
            "val_aggregate_score": float(val_scores[best_idx]),
            "accuracy": optimized_accuracy,
            "correct": round(optimized_accuracy * episodes)
            if isinstance(optimized_accuracy, int | float)
            else None,
            "discovery_eval_count": discovery_count(best_idx),
            "subscores": optimized_subscores,
        },
    }


def build_validation_audit_from_result(
    result: GEPARunResult | None,
    *,
    episodes: int,
    source: str,
) -> dict[str, Any]:
    """Fallback audit payload when only a `GEPARunResult` summary is available.

    Used on resume from a saved `gepa_done` checkpoint where the full GEPA
    detailed_results are not reconstructable; marks `details_available=False`
    so downstream consumers know subscore-level evidence is missing.
    """
    if result is None:
        return {
            "source": source,
            "episodes": episodes,
            "available": False,
            "details_available": False,
            "baseline": None,
            "optimized": None,
        }

    return {
        "source": source,
        "episodes": episodes,
        "available": True,
        "details_available": False,
        "best_program_index": result.best_program_index,
        "baseline": {
            "program_index": 0,
            "accuracy": result.baseline_accuracy,
            "correct": round(result.baseline_accuracy * episodes),
        },
        "optimized": {
            "program_index": result.best_program_index,
            "accuracy": result.final_accuracy,
            "correct": round(result.final_accuracy * episodes),
        },
    }
