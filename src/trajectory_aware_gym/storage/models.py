"""Pydantic models for experiment-level storage records."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class LoggingEvent(BaseModel):
    """One observability event emitted while logging or persisting a run."""

    stage: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    episode_run_id: str | None = None
    step_index: int | None = Field(default=None, ge=1)
    field: str | None = None
    value_repr: str | None = None
    message: str = Field(min_length=1)


class EpisodeLoggingSummary(BaseModel):
    """Per-episode logging health used to build run-level summaries."""

    status: Literal["complete", "partial", "failed"] = "complete"
    persistence_requested: bool = True
    trajectory_persisted: bool = False
    metrics_available: bool = False
    numeric_anomaly_count: int = Field(default=0, ge=0)
    events: list[LoggingEvent] = Field(default_factory=list)


class LoggingSummary(BaseModel):
    """Run-level summary of logging health and degradation."""

    status: Literal["complete", "partial", "failed"] = "complete"
    trajectory_persisted_episodes: int = Field(default=0, ge=0)
    trajectory_failed_episodes: int = Field(default=0, ge=0)
    metrics_unavailable_episodes: int = Field(default=0, ge=0)
    numeric_anomaly_count: int = Field(default=0, ge=0)
    events: list[LoggingEvent] = Field(default_factory=list)
    events_truncated: bool = False

    @classmethod
    def from_episode_summaries(
        cls,
        episode_summaries: list[EpisodeLoggingSummary],
        *,
        extra_events: list[LoggingEvent] | None = None,
        max_events: int = 100,
    ) -> LoggingSummary:
        """Aggregate per-episode summaries into one run-level summary."""

        persisted = sum(
            1
            for summary in episode_summaries
            if summary.persistence_requested and summary.trajectory_persisted
        )
        failed = sum(
            1
            for summary in episode_summaries
            if summary.persistence_requested and not summary.trajectory_persisted
        )
        metrics_unavailable = sum(
            1 for summary in episode_summaries if not summary.metrics_available
        )
        numeric_anomalies = sum(summary.numeric_anomaly_count for summary in episode_summaries)

        collected_events: list[LoggingEvent] = []
        events_truncated = False
        for event in [event for summary in episode_summaries for event in summary.events] + list(
            extra_events or []
        ):
            if len(collected_events) < max_events:
                collected_events.append(event)
            else:
                events_truncated = True
                break

        requested_episode_count = sum(
            1 for summary in episode_summaries if summary.persistence_requested
        )
        if (
            requested_episode_count > 0
            and failed == requested_episode_count
            and metrics_unavailable == len(episode_summaries)
        ):
            status: Literal["complete", "partial", "failed"] = "failed"
        elif failed or metrics_unavailable or numeric_anomalies or collected_events:
            status = "partial"
        else:
            status = "complete"

        return cls(
            status=status,
            trajectory_persisted_episodes=persisted,
            trajectory_failed_episodes=failed,
            metrics_unavailable_episodes=metrics_unavailable,
            numeric_anomaly_count=numeric_anomalies,
            events=collected_events,
            events_truncated=events_truncated,
        )


class ExperimentRunRecord(BaseModel):
    """Experiment-level metadata stored in the ``experiment_runs`` table.

    One record per (config × model × seed) replication.  Links to
    zero-or-more ``episodes`` rows via ``experiment_run_id`` FK.
    """

    experiment_run_id: str = Field(min_length=1)
    config_name: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    config_yaml: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    git_commit: str | None = None
    git_branch: str | None = None
    provider: str = Field(min_length=1)
    task_model_id: str = Field(min_length=1)
    reflection_model_id: str | None = None
    environment_id: str = Field(min_length=1)
    gepa_budget_mode: str | None = None
    replication_seed: int | None = None
    seed_prompt: str | None = None
    optimized_prompt: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal["running", "gepa_done", "completed", "failed"] = "running"
    hostname: str | None = None
    result_summary: dict[str, Any] | None = None
    cost_summary: dict[str, Any] | None = None
    error_summary: str | None = None
    logging_summary: LoggingSummary | None = None
    schema_version: str = Field(min_length=1)
