"""Helpers for aggregating logging health across experiment phases."""

from __future__ import annotations

from trajectory_aware_gym.storage.models import (
    EpisodeLoggingSummary,
    LoggingEvent,
    LoggingStatus,
    LoggingSummary,
)


def aggregate_logging_summary(
    episode_summaries: list[EpisodeLoggingSummary],
    *,
    extra_events: list[LoggingEvent] | None = None,
    max_events: int = 100,
) -> LoggingSummary:
    """Aggregate per-episode logging summaries into one run-level summary."""

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
    metrics_unavailable = sum(1 for summary in episode_summaries if not summary.metrics_available)
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
        status: LoggingStatus = "failed"
    elif failed or metrics_unavailable or numeric_anomalies or collected_events:
        status = "partial"
    else:
        status = "complete"

    return LoggingSummary(
        status=status,
        trajectory_persisted_episodes=persisted,
        trajectory_failed_episodes=failed,
        metrics_unavailable_episodes=metrics_unavailable,
        numeric_anomaly_count=numeric_anomalies,
        events=collected_events,
        events_truncated=events_truncated,
    )
