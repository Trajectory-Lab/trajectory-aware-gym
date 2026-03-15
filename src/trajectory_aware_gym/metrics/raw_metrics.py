"""Raw metric extraction for trajectory logs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog


def _get_ci(mapping: Mapping[str, Any], key: str) -> Any:
    """Get a mapping value using case-insensitive key lookup."""
    for candidate_key, candidate_value in mapping.items():
        if candidate_key.lower() == key.lower():
            return candidate_value
    return None


def _extract_numeric(
    mapping: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]
) -> float | None:
    """Extract the first numeric value from a list of candidate key paths."""
    for path in paths:
        current: Any = mapping
        matched = True
        for key in path:
            if not isinstance(current, Mapping):
                matched = False
                break
            current = _get_ci(current, key)
            if current is None:
                matched = False
                break
        if matched and isinstance(current, int | float) and not isinstance(current, bool):
            return float(current)
    return None


def _extract_int(mapping: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]) -> int | None:
    """Extract the first integer value from a list of candidate key paths."""
    value = _extract_numeric(mapping, paths)
    if value is None:
        return None
    return int(value)


def _percentile(values: list[float], percentile: float) -> float:
    """Return percentile using nearest-rank on a pre-sorted list."""
    if not values:
        raise ValueError("values must not be empty")
    rank = max(1, math.ceil((percentile / 100.0) * len(values)))
    return values[rank - 1]


def _repeat_action_rate(actions: list[str]) -> float:
    """Compute proportion of consecutive repeated actions."""
    if len(actions) < 2:
        return 0.0
    repeats = sum(1 for left, right in zip(actions, actions[1:], strict=False) if left == right)
    return repeats / (len(actions) - 1)


class EpisodeRawMetrics(BaseModel):
    """Per-episode raw metrics for downstream normalization and aggregation."""

    run_id: str
    environment_id: str
    seed: int | None = None

    started_at: datetime
    finished_at: datetime
    episode_latency_seconds: float = Field(ge=0.0)

    step_count: int = Field(ge=0)
    terminated: bool
    truncated: bool
    success: bool

    total_reward: float
    reward_per_step: float
    steps_per_second: float
    reward_per_second: float
    repeat_action_rate: float = Field(ge=0.0, le=1.0)

    llm_cost_usd: float | None = Field(default=None, ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    mean_llm_latency_seconds: float | None = Field(default=None, ge=0.0)
    p95_llm_latency_seconds: float | None = Field(default=None, ge=0.0)

    cost_per_step_usd: float | None = Field(default=None, ge=0.0)
    cost_per_success_usd: float | None = Field(default=None, ge=0.0)
    tokens_per_step: float | None = Field(default=None, ge=0.0)

    cost_data_coverage: float = Field(ge=0.0, le=1.0)
    token_data_coverage: float = Field(ge=0.0, le=1.0)
    llm_latency_data_coverage: float = Field(ge=0.0, le=1.0)


def extract_episode_raw_metrics(trajectory: TrajectoryLog) -> EpisodeRawMetrics:
    """Extract pragmatic raw metrics from one trajectory log."""
    step_count = len(trajectory.steps)
    episode_latency = max(0.0, (trajectory.finished_at - trajectory.started_at).total_seconds())
    actions = [step.action for step in trajectory.steps]

    terminated = trajectory.steps[-1].terminated if trajectory.steps else False
    truncated = trajectory.steps[-1].truncated if trajectory.steps else False
    success = terminated and trajectory.total_reward > 0

    cost_values: list[float] = []
    prompt_token_values: list[int] = []
    completion_token_values: list[int] = []
    total_token_values: list[int] = []
    llm_latency_values: list[float] = []

    cost_seen_steps = 0
    token_seen_steps = 0
    latency_seen_steps = 0

    cost_paths = (
        ("cost_usd",),
        ("llm_cost_usd",),
        ("cost",),
        ("metrics", "cost_usd"),
        ("usage", "cost_usd"),
        ("usage", "cost"),
    )
    prompt_token_paths = (
        ("prompt_tokens",),
        ("input_tokens",),
        ("usage", "prompt_tokens"),
        ("usage", "input_tokens"),
        ("token_usage", "prompt_tokens"),
        ("metrics", "prompt_tokens"),
    )
    completion_token_paths = (
        ("completion_tokens",),
        ("output_tokens",),
        ("usage", "completion_tokens"),
        ("usage", "output_tokens"),
        ("token_usage", "completion_tokens"),
        ("metrics", "completion_tokens"),
    )
    total_token_paths = (
        ("total_tokens",),
        ("usage", "total_tokens"),
        ("token_usage", "total_tokens"),
        ("metrics", "total_tokens"),
    )
    latency_paths = (
        ("latency_seconds",),
        ("llm_latency_seconds",),
        ("duration_seconds",),
        ("elapsed_seconds",),
        ("metrics", "latency_seconds"),
        ("metrics", "llm_latency_seconds"),
    )

    for step in trajectory.steps:
        info: Mapping[str, Any] = step.info if isinstance(step.info, Mapping) else {}

        step_cost = _extract_numeric(info, cost_paths)
        if step_cost is not None:
            cost_values.append(step_cost)
            cost_seen_steps += 1

        step_prompt_tokens = _extract_int(info, prompt_token_paths)
        step_completion_tokens = _extract_int(info, completion_token_paths)
        step_total_tokens = _extract_int(info, total_token_paths)

        if step_total_tokens is None and (
            step_prompt_tokens is not None or step_completion_tokens is not None
        ):
            step_total_tokens = (step_prompt_tokens or 0) + (step_completion_tokens or 0)

        if step_prompt_tokens is not None:
            prompt_token_values.append(step_prompt_tokens)
        if step_completion_tokens is not None:
            completion_token_values.append(step_completion_tokens)
        if step_total_tokens is not None:
            total_token_values.append(step_total_tokens)

        if (
            step_prompt_tokens is not None
            or step_completion_tokens is not None
            or step_total_tokens is not None
        ):
            token_seen_steps += 1

        step_llm_latency = _extract_numeric(info, latency_paths)
        if step_llm_latency is not None:
            llm_latency_values.append(step_llm_latency)
            latency_seen_steps += 1

    llm_cost_usd = sum(cost_values) if cost_values else None
    prompt_tokens = sum(prompt_token_values) if prompt_token_values else None
    completion_tokens = sum(completion_token_values) if completion_token_values else None
    total_tokens = sum(total_token_values) if total_token_values else None

    if llm_latency_values:
        sorted_latencies = sorted(llm_latency_values)
        mean_llm_latency_seconds = sum(sorted_latencies) / len(sorted_latencies)
        p95_llm_latency_seconds = _percentile(sorted_latencies, percentile=95.0)
    else:
        mean_llm_latency_seconds = None
        p95_llm_latency_seconds = None

    reward_per_step = trajectory.total_reward / step_count if step_count else 0.0
    steps_per_second = step_count / episode_latency if episode_latency > 0 else 0.0
    reward_per_second = trajectory.total_reward / episode_latency if episode_latency > 0 else 0.0

    cost_per_step_usd = (
        llm_cost_usd / step_count if llm_cost_usd is not None and step_count else None
    )
    cost_per_success_usd = llm_cost_usd if llm_cost_usd is not None and success else None
    tokens_per_step = total_tokens / step_count if total_tokens is not None and step_count else None

    denominator = step_count if step_count else 1

    return EpisodeRawMetrics(
        run_id=trajectory.run_id,
        environment_id=trajectory.environment_id,
        seed=trajectory.seed,
        started_at=trajectory.started_at,
        finished_at=trajectory.finished_at,
        episode_latency_seconds=episode_latency,
        step_count=step_count,
        terminated=terminated,
        truncated=truncated,
        success=success,
        total_reward=trajectory.total_reward,
        reward_per_step=reward_per_step,
        steps_per_second=steps_per_second,
        reward_per_second=reward_per_second,
        repeat_action_rate=_repeat_action_rate(actions),
        llm_cost_usd=llm_cost_usd,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        mean_llm_latency_seconds=mean_llm_latency_seconds,
        p95_llm_latency_seconds=p95_llm_latency_seconds,
        cost_per_step_usd=cost_per_step_usd,
        cost_per_success_usd=cost_per_success_usd,
        tokens_per_step=tokens_per_step,
        cost_data_coverage=cost_seen_steps / denominator,
        token_data_coverage=token_seen_steps / denominator,
        llm_latency_data_coverage=latency_seen_steps / denominator,
    )


def load_trajectory_log(file_path: Path) -> TrajectoryLog:
    """Load one trajectory JSON file as a validated `TrajectoryLog`."""
    return TrajectoryLog.model_validate_json(file_path.read_text(encoding="utf-8"))


def collect_raw_metrics(log_paths: Iterable[Path]) -> list[EpisodeRawMetrics]:
    """Load trajectory logs and return extracted per-episode metrics."""
    metrics: list[EpisodeRawMetrics] = []
    for path in sorted(log_paths):
        trajectory = load_trajectory_log(path)
        metrics.append(extract_episode_raw_metrics(trajectory))
    return metrics
