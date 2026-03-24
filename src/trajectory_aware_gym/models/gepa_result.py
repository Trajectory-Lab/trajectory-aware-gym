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
