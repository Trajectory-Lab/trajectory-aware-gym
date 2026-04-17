"""Unified run report format for cross-provider comparison.

Produces identically structured JSON summaries regardless of whether the
experiment ran on Bedrock (actual pricing) or Ollama (normalized proxy
pricing), so paper figures and tables can consume them uniformly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from trajectory_aware_gym.metrics.cost_normalization import compute_normalized_cost
from trajectory_aware_gym.storage.trajectory_db import load_experiment_run, query_trajectories


class RunReport(BaseModel):
    """Unified per-replication experiment summary."""

    experiment_run_id: str
    config_name: str
    operator: str
    provider: str
    task_model_id: str
    environment_id: str
    seed: int | None

    # Performance
    baseline_eval: dict[str, Any] | None = None
    eval_summary: dict[str, Any] | None = None

    # Cost — actual (Bedrock) or unavailable (Ollama)
    total_tokens: int | None = None
    total_tokens_known: int | None = None
    task_model_cost_usd: float | None = None
    task_model_cost_known_usd: float | None = None
    task_model_token_data_coverage: float | None = None
    task_model_cost_data_coverage: float | None = None
    reflection_cost_usd: float | None = None
    total_cost_usd: float | None = None
    total_cost_known_usd: float | None = None
    cost_type: str | None = None

    # Cost — normalized (Ollama only, for paper comparison)
    normalized_cost_usd: float | None = None
    normalization_reference: str | None = None

    # Timing
    wall_clock_seconds: float | None = None
    mean_llm_latency_ms: float | None = None

    # Metadata
    git_commit: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    logging_summary: dict[str, Any] | None = None


def build_run_report(
    *,
    experiment_run_id: str,
    db_path: Path,
    cost_summary: dict[str, Any],
    baseline_eval_summary: dict[str, Any] | None = None,
    eval_summary: dict[str, Any] | None = None,
    wall_clock_seconds: float | None = None,
    reference_prices: dict[str, dict[str, float]] | None = None,
    logging_summary: dict[str, Any] | None = None,
) -> RunReport:
    """Assemble a RunReport from DB record + caller-provided summaries.

    The caller passes ``cost_summary``, ``eval_summary``, etc. because
    these are already computed in the runner and do not need to be
    re-derived from raw episode data.
    """
    run = load_experiment_run(db_path, experiment_run_id)

    task_model_cost = cost_summary.get("task_model_cost")
    task_model_cost_known = cost_summary.get("task_model_cost_known", task_model_cost)
    reflection_cost = cost_summary.get("reflection_cost")
    total_cost = cost_summary.get("total_cost")
    total_cost_known = cost_summary.get("total_cost_known", total_cost)
    total_tokens = cost_summary.get("total_tokens")
    total_tokens_known = cost_summary.get("total_tokens_known", total_tokens)
    task_tokens = cost_summary.get("task_model_tokens")
    task_token_coverage = cost_summary.get("task_model_token_data_coverage")
    task_cost_coverage = cost_summary.get("task_model_cost_data_coverage")

    # Determine cost type and compute normalized cost for Ollama models.
    cost_type = cost_summary.get("cost_type")
    if not isinstance(cost_type, str):
        cost_type = "actual" if run.provider != "ollama" else "unavailable"

    normalized_cost_usd: float | None = None
    normalization_reference: str | None = None
    if run.provider == "ollama" and reference_prices and isinstance(task_tokens, int):
        # Use task_model_tokens split for normalization (prompt/completion
        # split isn't available at the summary level, so we approximate
        # with 70/30 prompt/completion split — consistent with typical
        # agentic workloads).
        prompt_approx = int(task_tokens * 0.7)
        completion_approx = task_tokens - prompt_approx
        maybe_cost = compute_normalized_cost(
            run.task_model_id,
            prompt_approx,
            completion_approx,
            reference_prices,
        )
        if maybe_cost is not None:
            normalized_cost_usd = maybe_cost
            ref = reference_prices.get(run.task_model_id, {})
            normalization_reference = (
                f"{run.task_model_id} @ ${ref.get('input_per_1m_tokens', '?')}/1M"
            )

    # Mean LLM latency from episode trajectories (best-effort).
    mean_latency: float | None = None
    try:
        trajectories = query_trajectories(db_path, experiment_run_id=experiment_run_id)
        latencies: list[float] = []
        for traj in trajectories:
            for step in traj.steps:
                for llm_call in step.llm_calls:
                    if llm_call.latency_ms is not None:
                        latencies.append(llm_call.latency_ms)
        if latencies:
            mean_latency = sum(latencies) / len(latencies)
    except Exception:  # noqa: BLE001
        pass

    return RunReport(
        experiment_run_id=experiment_run_id,
        config_name=run.config_name,
        operator=run.operator,
        provider=run.provider,
        task_model_id=run.task_model_id,
        environment_id=run.environment_id,
        seed=run.replication_seed,
        baseline_eval=baseline_eval_summary,
        eval_summary=eval_summary,
        total_tokens=total_tokens,
        total_tokens_known=total_tokens_known,
        task_model_cost_usd=task_model_cost,
        task_model_cost_known_usd=task_model_cost_known,
        task_model_token_data_coverage=task_token_coverage,
        task_model_cost_data_coverage=task_cost_coverage,
        reflection_cost_usd=reflection_cost,
        total_cost_usd=total_cost,
        total_cost_known_usd=total_cost_known,
        cost_type=cost_type,
        normalized_cost_usd=normalized_cost_usd,
        normalization_reference=normalization_reference,
        wall_clock_seconds=wall_clock_seconds,
        mean_llm_latency_ms=mean_latency,
        git_commit=run.git_commit,
        started_at=run.started_at.isoformat() if run.started_at else None,
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        logging_summary=(
            logging_summary
            if logging_summary is not None
            else run.logging_summary.model_dump(mode="json")
            if run.logging_summary is not None
            else None
        ),
    )
