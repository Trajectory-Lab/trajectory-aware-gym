"""Pydantic models for experiment-level storage records."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

type LoggingStatus = Literal["complete", "partial", "failed"]


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

    status: LoggingStatus = "complete"
    persistence_requested: bool = True
    trajectory_persisted: bool = False
    metrics_available: bool = False
    numeric_anomaly_count: int = Field(default=0, ge=0)
    events: list[LoggingEvent] = Field(default_factory=list)


class LoggingSummary(BaseModel):
    """Run-level summary of logging health and degradation."""

    status: LoggingStatus = "complete"
    trajectory_persisted_episodes: int = Field(default=0, ge=0)
    trajectory_failed_episodes: int = Field(default=0, ge=0)
    metrics_unavailable_episodes: int = Field(default=0, ge=0)
    numeric_anomaly_count: int = Field(default=0, ge=0)
    events: list[LoggingEvent] = Field(default_factory=list)
    events_truncated: bool = False


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
