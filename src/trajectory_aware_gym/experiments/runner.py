"""Production experiment runner with replication and cost tracking."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import socket
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import dspy  # type: ignore[import-untyped]
import yaml
from litellm import completion_cost  # type: ignore[import-untyped]

from trajectory_aware_gym.adapters.dspy_adapter import TrajectoryFitnessMetric
from trajectory_aware_gym.adapters.gem_episode_runner import GEMEpisodeResult, GEMEpisodeRunner
from trajectory_aware_gym.adapters.gem_solver_module import GEMSolverModule
from trajectory_aware_gym.adapters.trajectory_logger import SCHEMA_VERSION, TrajectoryLog
from trajectory_aware_gym.config import settings
from trajectory_aware_gym.config.core import Settings
from trajectory_aware_gym.config.llm_provider import get_reflection_lm
from trajectory_aware_gym.metrics import EpisodeRawMetrics
from trajectory_aware_gym.metrics.logging_summary import aggregate_logging_summary
from trajectory_aware_gym.metrics.run_report import build_run_report
from trajectory_aware_gym.models.experiment import (
    ExperimentConfig,
    TaskModelConfig,
)
from trajectory_aware_gym.models.gepa_result import GEPARunResult, accuracy_from_subscores
from trajectory_aware_gym.optimizers.gepa_progress_patch import enable_gepa_eval_progress
from trajectory_aware_gym.storage import (
    ExperimentRunRecord,
    LoggingEvent,
    LoggingStatus,
    LoggingSummary,
    episode_exists,
    generate_experiment_run_id,
    get_git_info,
    get_operator,
    save_experiment_run,
    save_trajectory,
    update_experiment_run,
)

DEFAULT_RESULTS_ROOT = Path("results")
_COST_DECIMAL_PLACES = 6
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DB_PATH = _PROJECT_ROOT / "logs" / "trajectories.db"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunExperimentArgs:
    """CLI-resolved arguments for a production experiment run.

    Note: the seed prompt comes from ``ExperimentConfig.seed_prompt`` in the
    YAML. ``seed_prompt_override`` below is a CLI escape hatch for ad-hoc
    experiments; leave it as ``None`` to use the per-experiment prompt.
    """

    config_path: Path
    max_metric_calls: int | None = None
    budget_mode: Literal["light", "medium", "heavy"] | None = None
    seed_prompt_override: str | None = None
    models: tuple[str, ...] | None = None
    seeds: tuple[int, ...] | None = None
    fresh: bool = False
    purge: bool = False
    resume: str | None = None
    results_root: Path = DEFAULT_RESULTS_ROOT
    halt_on_budget_exceeded: bool = False
    fail_fast: bool = True


def resolve_gepa_budget_kwargs(
    config: ExperimentConfig,
    max_metric_calls_override: int | None = None,
    budget_mode_override: Literal["light", "medium", "heavy"] | None = None,
) -> dict[str, Any]:
    """Build ``dspy.GEPA`` budget kwargs from config, with optional overrides.

    Returns exactly one of ``{"auto": mode}`` or ``{"max_metric_calls": n}``.
    dspy.GEPA enforces that exactly one budget source is provided.

    Resolution order:
    1. ``max_metric_calls_override`` → ``{"max_metric_calls": n}``
    2. ``budget_mode_override`` → ``{"auto": override}``
    3. config's ``gepa_budget.mode`` → ``{"auto": mode}``
    """
    if max_metric_calls_override is not None and budget_mode_override is not None:
        raise ValueError(
            "max_metric_calls_override and budget_mode_override are mutually exclusive"
        )
    if max_metric_calls_override is not None:
        if max_metric_calls_override < 1:
            raise ValueError("max_metric_calls override must be >= 1")
        return {"max_metric_calls": max_metric_calls_override}
    if budget_mode_override is not None:
        return {"auto": budget_mode_override}
    return {"auto": config.gepa_budget.mode}


def select_task_models(
    config: ExperimentConfig, selected_names: tuple[str, ...] | None
) -> list[TaskModelConfig]:
    """Filter configured task models by optional name allow-list."""
    if not selected_names:
        return list(config.task_models)

    selected = set(selected_names)
    models = [model for model in config.task_models if model.name in selected]
    if not models:
        raise ValueError(f"No configured task_models matched requested models: {sorted(selected)}")
    return models


def select_replication_seeds(
    config: ExperimentConfig, selected_seeds: tuple[int, ...] | None
) -> tuple[int, ...]:
    """Filter configured replication seeds by optional subset."""
    configured = set(config.seeds.replication_seeds)
    if not selected_seeds:
        return config.seeds.replication_seeds

    requested = set(selected_seeds)
    unknown = sorted(requested - configured)
    if unknown:
        raise ValueError(f"Requested seeds not in config.replication_seeds: {unknown}")

    return tuple(seed for seed in config.seeds.replication_seeds if seed in requested)


def _safe_segment(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_iso(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _config_hash(config: ExperimentConfig | dict[str, Any]) -> str:
    snapshot = (
        config.model_dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(config, ExperimentConfig)
        else config
    )
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            handle.write(content)
            tmp_path = Path(handle.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _write_jsonl(path: Path, rows: list[EpisodeRawMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[EpisodeRawMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        field_names = list(EpisodeRawMetrics.model_fields.keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=field_names).writeheader()
        return

    field_names = list(rows[0].model_dump(mode="json").keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _load_raw_metrics_jsonl(path: Path) -> list[EpisodeRawMetrics]:
    if not path.exists():
        return []
    rows: list[EpisodeRawMetrics] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    rows.append(EpisodeRawMetrics.model_validate(parsed))
    except Exception:  # noqa: BLE001
        logger.warning("Failed to load raw metrics JSONL from %s", path, exc_info=True)
        return []
    return rows


def _load_raw_metrics_csv(path: Path) -> list[EpisodeRawMetrics]:
    if not path.exists():
        return []
    rows: list[EpisodeRawMetrics] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw_row in csv.DictReader(handle):
                parsed_row = {
                    key: None if value == "" else yaml.safe_load(value)
                    for key, value in raw_row.items()
                    if key is not None
                }
                rows.append(EpisodeRawMetrics.model_validate(parsed_row))
    except Exception:  # noqa: BLE001
        logger.warning("Failed to load raw metrics CSV from %s", path, exc_info=True)
        return []
    return rows


def _load_training_metrics_rows(replication_dir: Path) -> list[EpisodeRawMetrics]:
    rows = _load_raw_metrics_jsonl(replication_dir / "training_metrics.jsonl")
    if rows:
        return rows
    return _load_raw_metrics_csv(replication_dir / "training_metrics.csv")


def _raw_metrics_summary(rows: list[EpisodeRawMetrics]) -> dict[str, float | int | None]:
    return _raw_metrics_summary_with_omissions(rows, metrics_unavailable_episodes=0)


def _raw_metrics_summary_for_results(
    results: list[GEMEpisodeResult],
) -> dict[str, float | int | None]:
    rows = [result.raw_metrics for result in results if result.raw_metrics is not None]
    return _raw_metrics_summary_with_omissions(
        rows,
        metrics_unavailable_episodes=len(results) - len(rows),
    )


def _raw_metrics_summary_with_omissions(
    rows: list[EpisodeRawMetrics],
    *,
    metrics_unavailable_episodes: int,
) -> dict[str, float | int | None]:
    if not rows:
        return {
            "episodes": 0,
            "metrics_unavailable_episodes": metrics_unavailable_episodes,
            "successes": 0,
            "mean_latency_seconds": None,
            "known_total_cost_usd": None,
            "total_cost_usd": None,
            "known_total_tokens": None,
            "total_tokens": None,
            "mean_cost_data_coverage": None,
            "mean_token_data_coverage": None,  # nosec B105 — not a password
            "mean_llm_latency_data_coverage": None,
        }

    episodes = len(rows)
    successes = sum(1 for row in rows if row.success)
    latencies = [row.episode_latency_seconds for row in rows]
    costs = [row.llm_cost_usd for row in rows if row.llm_cost_usd is not None]
    tokens = [row.total_tokens for row in rows if row.total_tokens is not None]
    mean_cost_coverage = sum(r.cost_data_coverage for r in rows) / episodes
    mean_token_coverage = sum(r.token_data_coverage for r in rows) / episodes
    complete_cost = (
        metrics_unavailable_episodes == 0
        and all(r.cost_data_coverage == 1.0 for r in rows)
        and all(r.llm_cost_usd is not None for r in rows)
    )
    complete_tokens = (
        metrics_unavailable_episodes == 0
        and all(r.token_data_coverage == 1.0 for r in rows)
        and all(r.total_tokens is not None for r in rows)
    )

    return {
        "episodes": episodes,
        "metrics_unavailable_episodes": metrics_unavailable_episodes,
        "successes": successes,
        "mean_latency_seconds": round(sum(latencies) / episodes, _COST_DECIMAL_PLACES),
        "known_total_cost_usd": round(sum(costs), _COST_DECIMAL_PLACES),
        "total_cost_usd": round(sum(costs), _COST_DECIMAL_PLACES) if complete_cost else None,
        "known_total_tokens": int(sum(tokens)),
        "total_tokens": int(sum(tokens)) if complete_tokens else None,
        "mean_cost_data_coverage": round(mean_cost_coverage, _COST_DECIMAL_PLACES),
        "mean_token_data_coverage": round(mean_token_coverage, _COST_DECIMAL_PLACES),
        "mean_llm_latency_data_coverage": round(
            sum(r.llm_latency_data_coverage for r in rows) / episodes, _COST_DECIMAL_PLACES
        ),
    }


def _extract_reflection_usage(reflection_lm: Any | None) -> tuple[int, float]:
    if reflection_lm is None:
        return (0, 0.0)

    history = getattr(reflection_lm, "history", [])
    total_tokens = 0
    total_cost = 0.0

    for entry in history:
        if not isinstance(entry, dict):
            continue

        usage = entry.get("usage")
        if isinstance(usage, dict):
            usage_total = usage.get("total_tokens")
            if isinstance(usage_total, int):
                total_tokens += usage_total

        response = entry.get("response")
        if response is not None:
            try:
                maybe_cost = completion_cost(completion_response=response)
            except Exception:  # noqa: BLE001  # LiteLLM raises bare Exception for unmapped models
                maybe_cost = None
            if isinstance(maybe_cost, int | float):
                total_cost += float(maybe_cost)

    return (total_tokens, total_cost)


def _result_success(result: GEMEpisodeResult) -> bool | None:
    if result.raw_metrics is not None:
        return result.raw_metrics.success
    if result.trajectory is not None and result.trajectory.episode_outcome is not None:
        return result.trajectory.episode_outcome == "success"
    return None


def _results_logging_summary(results: list[GEMEpisodeResult]) -> LoggingSummary:
    return aggregate_logging_summary([result.logging_summary for result in results])


def _persist_training_trajectories(
    results: list[GEMEpisodeResult],
    *,
    experiment_run_id: str,
    db_path: Path = _DB_PATH,
) -> list[GEMEpisodeResult]:
    persisted_results: list[GEMEpisodeResult] = []
    for result in results:
        trajectory = result.trajectory
        if not isinstance(trajectory, TrajectoryLog):
            persisted_results.append(result)
            continue

        logging_summary = result.logging_summary.model_copy(deep=True)
        logging_summary.persistence_requested = True
        log_path = result.log_path

        try:
            if not episode_exists(db_path, trajectory.run_id):
                save_trajectory(
                    db_path,
                    trajectory,
                    experiment_run_id=experiment_run_id,
                )
            log_path = db_path
            logging_summary.trajectory_persisted = True
            if logging_summary.status == "failed":
                logging_summary.status = "partial"
        except Exception as exc:  # noqa: BLE001
            if logging_summary.status == "complete":
                logging_summary.status = "partial"
            logging_summary.events.append(
                LoggingEvent(
                    stage="save",
                    kind="persistence_failed",
                    episode_run_id=trajectory.run_id,
                    message=(f"Failed to persist a GEPA training trajectory to SQLite: {exc!r}"),
                )
            )

        persisted_results.append(
            replace(
                result,
                log_path=log_path,
                logging_summary=logging_summary,
            )
        )

    return persisted_results


def _merge_logging_summaries(
    summaries: list[LoggingSummary],
    *,
    max_events: int = 100,
) -> LoggingSummary:
    if not summaries:
        return LoggingSummary()

    events: list[LoggingEvent] = []
    events_truncated = any(summary.events_truncated for summary in summaries)
    for summary in summaries:
        for event in summary.events:
            if len(events) < max_events:
                events.append(event)
            else:
                events_truncated = True
                break

    if summaries and all(summary.status == "failed" for summary in summaries):
        status: LoggingStatus = "failed"
    elif any(summary.status != "complete" for summary in summaries) or events:
        status = "partial"
    else:
        status = "complete"

    return LoggingSummary(
        status=status,
        trajectory_persisted_episodes=sum(
            summary.trajectory_persisted_episodes for summary in summaries
        ),
        trajectory_failed_episodes=sum(summary.trajectory_failed_episodes for summary in summaries),
        metrics_unavailable_episodes=sum(
            summary.metrics_unavailable_episodes for summary in summaries
        ),
        numeric_anomaly_count=sum(summary.numeric_anomaly_count for summary in summaries),
        events=events,
        events_truncated=events_truncated,
    )


def _summarize_task_usage_rows(
    rows: list[EpisodeRawMetrics],
    *,
    metrics_unavailable_episodes: int,
) -> dict[str, float | int | bool | None]:
    episodes = len(rows) + metrics_unavailable_episodes
    if episodes == 0:
        return {
            "episodes": 0,
            "total_tokens": 0,
            "known_total_tokens": 0,
            "token_data_coverage": 1.0,  # nosec B105 - coverage metric, not a credential
            "total_cost_usd": 0.0,
            "known_cost_usd": 0.0,
            "cost_data_coverage": 1.0,
            "has_missing_cost_data": False,
            "metrics_unavailable_episodes": 0,
        }

    known_total_tokens = int(sum(row.total_tokens for row in rows if row.total_tokens is not None))
    known_cost_usd = float(sum(row.llm_cost_usd for row in rows if row.llm_cost_usd is not None))
    token_coverage = sum((row.token_data_coverage for row in rows), 0.0) / episodes
    cost_coverage = sum((row.cost_data_coverage for row in rows), 0.0) / episodes
    complete_tokens = (
        metrics_unavailable_episodes == 0
        and all(row.token_data_coverage == 1.0 for row in rows)
        and all(row.total_tokens is not None for row in rows)
    )
    complete_cost = (
        metrics_unavailable_episodes == 0
        and all(row.cost_data_coverage == 1.0 for row in rows)
        and all(row.llm_cost_usd is not None for row in rows)
    )

    return {
        "episodes": episodes,
        "total_tokens": known_total_tokens if complete_tokens else None,
        "known_total_tokens": known_total_tokens,
        "token_data_coverage": round(token_coverage, _COST_DECIMAL_PLACES),
        "total_cost_usd": round(known_cost_usd, _COST_DECIMAL_PLACES) if complete_cost else None,
        "known_cost_usd": round(known_cost_usd, _COST_DECIMAL_PLACES),
        "cost_data_coverage": round(cost_coverage, _COST_DECIMAL_PLACES),
        "has_missing_cost_data": not complete_cost,
        "metrics_unavailable_episodes": metrics_unavailable_episodes,
    }


def _merge_task_usage_summaries(
    summaries: list[dict[str, float | int | bool | None]],
) -> dict[str, float | int | bool | None]:
    def as_int(summary: dict[str, float | int | bool | None], key: str) -> int | None:
        value = summary.get(key)
        return int(value) if isinstance(value, int | float) else None

    def as_float(summary: dict[str, float | int | bool | None], key: str) -> float | None:
        value = summary.get(key)
        return float(value) if isinstance(value, int | float) else None

    total_episodes = sum(
        episodes for summary in summaries if (episodes := as_int(summary, "episodes")) is not None
    )
    if total_episodes == 0:
        return _summarize_task_usage_rows([], metrics_unavailable_episodes=0)

    known_total_tokens = sum(
        tokens
        for summary in summaries
        if (tokens := as_int(summary, "known_total_tokens")) is not None
    )
    known_cost_usd = sum(
        cost for summary in summaries if (cost := as_float(summary, "known_cost_usd")) is not None
    )
    metrics_unavailable = sum(
        unavailable
        for summary in summaries
        if (unavailable := as_int(summary, "metrics_unavailable_episodes")) is not None
    )
    token_coverage = (
        sum(
            coverage * episodes
            for summary in summaries
            if (coverage := as_float(summary, "token_data_coverage")) is not None
            and (episodes := as_int(summary, "episodes")) is not None
        )
        / total_episodes
    )
    cost_coverage = (
        sum(
            coverage * episodes
            for summary in summaries
            if (coverage := as_float(summary, "cost_data_coverage")) is not None
            and (episodes := as_int(summary, "episodes")) is not None
        )
        / total_episodes
    )
    complete_tokens = metrics_unavailable == 0 and all(
        summary.get("total_tokens") is not None
        for summary in summaries
        if (episodes := as_int(summary, "episodes")) is not None and episodes > 0
    )
    complete_cost = metrics_unavailable == 0 and all(
        summary.get("total_cost_usd") is not None
        for summary in summaries
        if (episodes := as_int(summary, "episodes")) is not None and episodes > 0
    )

    return {
        "episodes": total_episodes,
        "total_tokens": known_total_tokens if complete_tokens else None,
        "known_total_tokens": known_total_tokens,
        "token_data_coverage": round(token_coverage, _COST_DECIMAL_PLACES),
        "total_cost_usd": round(known_cost_usd, _COST_DECIMAL_PLACES) if complete_cost else None,
        "known_cost_usd": round(known_cost_usd, _COST_DECIMAL_PLACES),
        "cost_data_coverage": round(cost_coverage, _COST_DECIMAL_PLACES),
        "has_missing_cost_data": not complete_cost,
        "metrics_unavailable_episodes": metrics_unavailable,
    }


def _summarize_task_usage(results: list[GEMEpisodeResult]) -> dict[str, float | int | bool | None]:
    rows = [result.raw_metrics for result in results if result.raw_metrics is not None]
    metrics_unavailable = len(results) - len(rows)
    return _summarize_task_usage_rows(rows, metrics_unavailable_episodes=metrics_unavailable)


def _load_gepa_result(metadata: dict[str, Any]) -> GEPARunResult | None:
    payload = metadata.get("result")
    if not isinstance(payload, dict):
        return None
    try:
        return GEPARunResult.model_validate(payload)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to parse saved GEPA result from metadata", exc_info=True)
        return None


def _extract_task_usage(results: list[GEMEpisodeResult]) -> dict[str, float | int | bool | None]:
    """Backward-compatible alias for task-usage aggregation."""
    return _summarize_task_usage(results)


def _extract_fitness_history(optimized_module: Any) -> list[dict[str, Any]]:
    detailed = getattr(optimized_module, "detailed_results", None)
    if detailed is None:
        return []

    aggregate = getattr(detailed, "val_aggregate_scores", None)
    subscores = getattr(detailed, "val_subscores", None)
    discovery = getattr(detailed, "discovery_eval_counts", None)

    if not isinstance(aggregate, list):
        return []

    rows: list[dict[str, Any]] = []
    for index, score in enumerate(aggregate):
        if not isinstance(score, int | float):
            continue
        row: dict[str, Any] = {
            "index": index,
            "val_aggregate_score": float(score),
        }
        if isinstance(subscores, list) and index < len(subscores):
            row["accuracy"] = accuracy_from_subscores(subscores[index])
        if isinstance(discovery, list) and index < len(discovery):
            row["metric_calls"] = int(discovery[index])
        rows.append(row)
    return rows


def _extract_pareto_frontier(optimized_module: Any) -> list[dict[str, Any]]:
    detailed = getattr(optimized_module, "detailed_results", None)
    if detailed is None:
        return []

    frontier = getattr(detailed, "pareto_frontier", None)
    if isinstance(frontier, list):
        out: list[dict[str, Any]] = []
        for item in frontier:
            if isinstance(item, dict):
                out.append(item)
        return out
    return []


def _build_validation_summary(*, accuracy: float, episodes: int) -> dict[str, Any]:
    return {
        "episodes": episodes,
        "correct": round(accuracy * episodes),
        "accuracy": accuracy,
    }


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_effective_config_snapshot(
    config: ExperimentConfig,
    *,
    seed_prompt: str,
    budget_mode: Literal["light", "medium", "heavy"],
    seed_prompt_override: str | None,
    budget_mode_override: Literal["light", "medium", "heavy"] | None,
    max_metric_calls_override: int | None,
) -> dict[str, Any]:
    """Capture the effective runtime config, including CLI overrides."""

    snapshot = config.model_dump(mode="json", by_alias=True, exclude_none=True)
    snapshot["seed_prompt"] = seed_prompt
    snapshot["gepa_budget"]["mode"] = budget_mode

    runtime_overrides: dict[str, Any] = {}
    if seed_prompt_override is not None:
        runtime_overrides["seed_prompt"] = seed_prompt_override
    if budget_mode_override is not None:
        runtime_overrides["budget_mode"] = budget_mode_override
    if max_metric_calls_override is not None:
        runtime_overrides["max_metric_calls"] = max_metric_calls_override
    if runtime_overrides:
        snapshot["runtime_overrides"] = runtime_overrides

    return snapshot


def _replication_status(replication_dir: Path) -> str | None:
    """Return the status from run_metadata.json, or None if missing/corrupt."""
    payload = _load_json_dict(replication_dir / "run_metadata.json")
    if payload is None:
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def _is_replication_completed(replication_dir: Path) -> bool:
    return _replication_status(replication_dir) == "completed"


def _build_task_lm(config: ExperimentConfig, task_model: TaskModelConfig) -> Any:
    kwargs: dict[str, Any] = {
        "model": task_model.model_id,
        "temperature": config.eval_protocol.temperature_train,
        "max_tokens": config.eval_protocol.max_response_tokens,
        "top_p": config.eval_protocol.top_p,
    }
    # top_k <= 0 means disabled per GEM paper convention (Table 3).
    top_k = config.eval_protocol.top_k
    if top_k > 0:
        model_id = task_model.model_id
        if task_model.provider == "bedrock" and "anthropic" not in model_id:
            kwargs["additional_model_request_fields"] = {"top_k": top_k}
        else:
            kwargs["top_k"] = top_k
    if task_model.provider == "ollama":
        kwargs["api_base"] = settings.ollama.api_base
    if task_model.provider in ("bedrock", "sagemaker"):
        aws_region = getattr(settings.aws, "region", None)
        if aws_region is not None:
            kwargs["aws_region_name"] = aws_region
    return dspy.LM(**kwargs)


def _build_examples(config: ExperimentConfig, start_seed: int, count: int) -> list[dspy.Example]:
    """Build dspy.Example list by resetting GEM env with sequential seeds."""
    from trajectory_aware_gym.adapters.gem_env_factory import make_env

    env = make_env(config.environment.gem_env_id)

    examples: list[dspy.Example] = []
    for index in range(count):
        seed = start_seed + index
        observation, _ = env.reset(seed=seed)
        examples.append(
            dspy.Example(problem=str(observation), seed=seed).with_inputs("problem", "seed")
        )

    if hasattr(env, "close"):
        env.close()
    return examples


def _build_trainset(config: ExperimentConfig) -> list[dspy.Example]:
    return _build_examples(config, config.seeds.data_seed, config.environment.train_size)


def _build_valset(config: ExperimentConfig) -> list[dspy.Example]:
    """Build a held-out validation slice disjoint from train and eval.

    Seed range: [data_seed + train_size, data_seed + train_size + val_size).
    GEPA uses val scores for Pareto selection but does not reflect on val
    contents, mirroring the train/val/test split from the GEPA paper.
    """
    start_seed = config.seeds.data_seed + config.environment.train_size
    return _build_examples(config, start_seed, config.environment.effective_val_size)


def _budget_alert_fraction() -> float:
    raw_threshold = settings.cost_tracking.alert_threshold
    if raw_threshold > 1.0:
        return raw_threshold / 100.0
    return max(0.0, raw_threshold)


def _model_replication_dir(
    args: RunExperimentArgs,
    config: ExperimentConfig,
    task_model: TaskModelConfig,
    replication_seed: int,
    run_timestamp: str,
) -> Path:
    return (
        args.results_root
        / _safe_segment(config.name)
        / run_timestamp
        / _safe_segment(task_model.name)
        / f"replication_{replication_seed}"
    )


def _find_resumable_run(results_root: Path, config_name: str) -> str | None:
    """Find the most recent incomplete run's timestamp directory, if any.

    A run is incomplete if its ``run_summary.json`` has no ``finished_at`` field.
    Returns the directory name (timestamp string) or None.
    """
    config_dir = results_root / _safe_segment(config_name)
    if not config_dir.exists():
        return None

    candidates: list[str] = []
    for entry in config_dir.iterdir():
        if not entry.is_dir():
            continue
        summary_path = entry / "run_summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("finished_at") is None:
            candidates.append(entry.name)

    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def _eval_examples(config: ExperimentConfig) -> list[dspy.Example]:
    start_seed = (
        config.seeds.data_seed
        + config.environment.train_size
        + config.environment.effective_val_size
    )
    return _build_examples(config, start_seed, config.environment.effective_eval_size)


@dataclass(frozen=True)
class _EvalTask:
    """One eval episode to run."""

    episode_index: int
    seed: int
    expected_observation: str | None
    instructions: str


def _run_eval_task(
    task: _EvalTask,
    config: ExperimentConfig,
    task_model_id: str,
    experiment_run_id: str | None = None,
) -> GEMEpisodeResult:
    """Run a single eval episode. Thread-safe: creates its own runner/env."""
    runner = GEMEpisodeRunner(
        environment_id=config.environment.gem_env_id,
        model_id=task_model_id,
        temperature=config.eval_protocol.temperature_eval,
        max_steps=config.environment.max_steps,
        max_response_tokens=config.eval_protocol.max_response_tokens,
        top_p=config.eval_protocol.top_p,
        top_k=config.eval_protocol.top_k,
        seed=config.seeds.data_seed + config.environment.train_size,
        experiment_name=config.name,
        tools=config.environment.active_tool_names,
        experiment_run_id=experiment_run_id,
    )
    return runner.run_episode(
        task.instructions,
        episode_index=task.episode_index,
        seed_override=task.seed,
        expected_observation=task.expected_observation,
        persist=True,
    )


def _run_heldout_eval(
    *,
    config: ExperimentConfig,
    task_model_id: str,
    instructions: str,
    experiment_run_id: str | None = None,
) -> tuple[list[GEMEpisodeResult], dict[str, Any]]:
    eval_examples = _eval_examples(config)
    rollouts = config.eval_protocol.rollouts_per_task

    tasks: list[_EvalTask] = []
    for example_index, example in enumerate(eval_examples):
        for rollout_index in range(rollouts):
            tasks.append(
                _EvalTask(
                    episode_index=example_index * rollouts + rollout_index,
                    seed=int(example.seed) + rollout_index,
                    expected_observation=str(example.problem) if rollout_index == 0 else None,
                    instructions=instructions,
                )
            )

    import math
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_tasks = len(tasks)
    max_workers = config.eval_protocol.max_eval_workers
    per_episode_timeout = config.eval_protocol.eval_episode_timeout_seconds
    # Total timeout: enough waves to finish all tasks, plus margin.
    waves = math.ceil(total_tasks / max_workers)
    total_timeout = per_episode_timeout * waves * 1.5
    logger.info(
        "Starting held-out eval: %d episodes, %d workers, %ds/episode timeout",
        total_tasks,
        max_workers,
        per_episode_timeout,
    )
    results: list[GEMEpisodeResult | None] = [None] * total_tasks
    done_count = 0
    timed_out_episodes: list[int] = []
    eval_start = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_idx = {
            executor.submit(_run_eval_task, task, config, task_model_id, experiment_run_id): idx
            for idx, task in enumerate(tasks)
        }
        try:
            for future in as_completed(future_to_idx, timeout=total_timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception("Eval episode %d failed", idx)
                done_count += 1
                elapsed = time.monotonic() - eval_start
                per_ep = elapsed / done_count
                remaining = per_ep * (total_tasks - done_count)
                mins_left = remaining / 60
                pct = done_count / total_tasks
                bar_len = 30
                filled = int(bar_len * pct)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"\rEval: {bar} {done_count}/{total_tasks} ({pct:.0%}) ~{mins_left:.1f}m left",
                    end="",
                    flush=True,
                )
        except TimeoutError:
            timed_out_episodes = [idx for future, idx in future_to_idx.items() if not future.done()]
            for future in future_to_idx:
                future.cancel()
            logger.warning(
                "Eval timed out after %.0fs with %d episodes still running: %s",
                time.monotonic() - eval_start,
                len(timed_out_episodes),
                timed_out_episodes[:20],
            )
        print()  # newline after progress bar
    finally:
        # Don't block on stuck threads (e.g. math_verify/sympy deadlocks).
        executor.shutdown(wait=False, cancel_futures=True)

    completed = [r for r in results if r is not None]
    failed = total_tasks - len(completed)
    if timed_out_episodes:
        logger.warning(
            "Eval had %d timed-out episodes (counted as failures): %s",
            len(timed_out_episodes),
            timed_out_episodes[:20],
        )
    if failed > 0:
        logger.warning(
            "Eval had %d/%d failed episodes — accuracy is computed over %d completed only",
            failed,
            total_tasks,
            len(completed),
        )
    scorable = [result for result in completed if _result_success(result) is not None]
    metrics_unavailable = len(completed) - len(scorable)
    successes = sum(1 for result in scorable if _result_success(result))
    correct = successes
    total = len(scorable)
    summary = {
        "episodes_attempted": total_tasks,
        "episodes": total,
        "failed": failed,
        "timed_out": len(timed_out_episodes),
        "metrics_unavailable": metrics_unavailable,
        "successes": successes,
        "success_rate": (successes / total) if total else 0.0,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "temperature_eval": config.eval_protocol.temperature_eval,
    }
    return (completed, summary)


def run_experiment(args: RunExperimentArgs) -> dict[str, Any]:
    """Run a production GEPA experiment across configured models and replications."""
    config = ExperimentConfig.from_yaml(args.config_path)

    seed_prompt = args.seed_prompt_override or config.seed_prompt
    if args.seed_prompt_override is not None:
        logger.info("Using CLI --seed-prompt override (config.seed_prompt ignored for this run).")

    models = select_task_models(config, args.models)
    replication_seeds = select_replication_seeds(config, args.seeds)
    gepa_budget_kwargs = resolve_gepa_budget_kwargs(
        config,
        max_metric_calls_override=args.max_metric_calls,
        budget_mode_override=args.budget_mode,
    )
    # For reporting only — auto mode resolves to None here since DSPy
    # computes the true budget internally from trainset/valset sizes.
    budget_override = gepa_budget_kwargs.get("max_metric_calls")
    effective_budget_mode = gepa_budget_kwargs.get("auto", config.gepa_budget.mode)
    effective_config_snapshot = _build_effective_config_snapshot(
        config,
        seed_prompt=seed_prompt,
        budget_mode=effective_budget_mode,
        seed_prompt_override=args.seed_prompt_override,
        budget_mode_override=args.budget_mode,
        max_metric_calls_override=args.max_metric_calls,
    )

    if args.purge:
        purge_target = args.results_root / _safe_segment(config.name)
        if purge_target.exists():
            logger.info("Purging all results for config: %s", purge_target)
            shutil.rmtree(purge_target)

    trainset = _build_trainset(config)
    valset = _build_valset(config)

    global_started = _utc_now()

    # Decide whether to resume an incomplete run or start fresh.
    if args.resume:
        run_timestamp = args.resume
        logger.info("Resuming specified run: %s", run_timestamp)
    elif args.fresh:
        run_timestamp = global_started.strftime("%Y%m%dT%H%M%SZ")
    else:
        resumable = _find_resumable_run(args.results_root, config.name)
        if resumable:
            run_timestamp = resumable
            logger.info("Resuming incomplete run: %s", run_timestamp)
        else:
            run_timestamp = global_started.strftime("%Y%m%dT%H%M%SZ")

    experiment_summary_path = (
        args.results_root / _safe_segment(config.name) / run_timestamp / "run_summary.json"
    )
    run_dir_id = f"{_safe_segment(config.name)}-{run_timestamp}"
    config_hash = _config_hash(effective_config_snapshot)
    operator = get_operator()
    git_commit, git_branch = get_git_info()

    existing_run_summary = _load_json_dict(experiment_summary_path)
    if existing_run_summary is not None:
        existing_hash = existing_run_summary.get("config_hash")
        if isinstance(existing_hash, str) and existing_hash != config_hash:
            raise ValueError(
                "Cannot resume run with different effective config. "
                f"{existing_hash=} {config_hash=}"
            )
        preserved_started_at = existing_run_summary.get("started_at")
        if isinstance(preserved_started_at, str):
            maybe_started = _parse_iso(preserved_started_at)
            if maybe_started is not None:
                global_started = maybe_started

    run_summary: dict[str, Any] = existing_run_summary or {}
    run_summary.update(
        {
            "run_id": run_dir_id,
            "config": config.name,
            "config_hash": config_hash,
            "git_commit": git_commit,
            "started_at": _format_iso(global_started),
            "finished_at": None,
            "gepa_budget_mode": effective_budget_mode,
            "max_metric_calls_override": budget_override,
        }
    )
    if not isinstance(run_summary.get("models"), dict):
        run_summary["models"] = {}

    # Write run_summary.json at start so _find_resumable_run can detect it.
    _write_json(experiment_summary_path, run_summary)

    def _finalize_run_summary() -> dict[str, Any]:
        finished_at = _utc_now()
        run_summary["finished_at"] = _format_iso(finished_at)
        run_summary["elapsed_seconds"] = round((finished_at - global_started).total_seconds(), 3)
        _write_json(experiment_summary_path, run_summary)
        return run_summary

    override_patch = config.fitness_override.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    with Settings.override_fitness(override_patch):
        metric = TrajectoryFitnessMetric(return_feedback=True)

        for task_model in models:
            model_id = task_model.model_id
            logger.info("Starting model: %s (%s)", task_model.name, model_id)
            model_entries = run_summary["models"].setdefault(task_model.name, {})

            for replication_seed in replication_seeds:
                replication_dir = _model_replication_dir(
                    args, config, task_model, replication_seed, run_timestamp
                )
                replication_dir.mkdir(parents=True, exist_ok=True)

                if _is_replication_completed(replication_dir):
                    logger.info(
                        "Skipping completed replication model=%s seed=%s",
                        task_model.name,
                        replication_seed,
                    )
                    model_entries.setdefault(str(replication_seed), {"status": "skipped"})
                    continue

                prior_metadata = _load_json_dict(replication_dir / "run_metadata.json") or {}
                prior_hash = prior_metadata.get("config_hash")
                if isinstance(prior_hash, str) and prior_hash != config_hash:
                    raise ValueError(
                        "Cannot resume replication with different effective config. "
                        f"{prior_hash=} {config_hash=}"
                    )

                started_at = _utc_now()
                persisted_started_at = prior_metadata.get("started_at")
                if isinstance(persisted_started_at, str):
                    maybe_started = _parse_iso(persisted_started_at)
                    if maybe_started is not None:
                        started_at = maybe_started

                prior_status = prior_metadata.get("status")
                metadata = dict(prior_metadata)
                metadata.pop("error", None)
                metadata.update(
                    {
                        "run_id": run_dir_id,
                        "status": "gepa_done" if prior_status == "gepa_done" else "running",
                        "config_hash": config_hash,
                        "git_commit": git_commit,
                        "started_at": _format_iso(started_at),
                        "finished_at": None,
                        "seed": replication_seed,
                        "model_name": task_model.name,
                        "model_id": model_id,
                        "gepa_budget_mode": effective_budget_mode,
                        "max_metric_calls_override": budget_override,
                    }
                )

                config_snapshot_yaml = yaml.dump(
                    effective_config_snapshot,
                    default_flow_style=False,
                    sort_keys=False,
                )

                # --- Experiment run DB record ---
                persisted_exp_run_id = prior_metadata.get("experiment_run_id")
                exp_run_id = (
                    persisted_exp_run_id
                    if isinstance(persisted_exp_run_id, str) and persisted_exp_run_id.strip()
                    else generate_experiment_run_id(
                        config_name=config.name,
                        provider=task_model.provider,
                        model_id=model_id,
                        operator=operator,
                        seed=replication_seed,
                        timestamp=started_at,
                    )
                )
                metadata["experiment_run_id"] = exp_run_id
                _write_json(replication_dir / "run_metadata.json", metadata)
                (replication_dir / "config_snapshot.yaml").write_text(
                    config_snapshot_yaml,
                    encoding="utf-8",
                )

                exp_run_record = ExperimentRunRecord(
                    experiment_run_id=exp_run_id,
                    config_name=config.name,
                    config_hash=config_hash,
                    config_yaml=config_snapshot_yaml,
                    operator=operator,
                    git_commit=git_commit,
                    git_branch=git_branch,
                    provider=task_model.provider,
                    task_model_id=model_id,
                    reflection_model_id=config.reflection_model.model_id,
                    environment_id=config.environment.gem_env_id,
                    gepa_budget_mode=effective_budget_mode,
                    replication_seed=replication_seed,
                    seed_prompt=seed_prompt,
                    started_at=started_at,
                    status="running",
                    hostname=socket.gethostname(),
                    schema_version=SCHEMA_VERSION,
                )
                try:
                    save_experiment_run(_DB_PATH, exp_run_record)
                except ValueError:
                    logger.debug("Experiment run record already exists, reusing %s", exp_run_id)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Failed to save experiment run record %s (continuing)",
                        exp_run_id,
                        exc_info=True,
                    )

                task_lm = _build_task_lm(config, task_model)
                dspy.configure(lm=task_lm)

                reflection_lm = get_reflection_lm(
                    config.reflection_model.model_id,
                    temperature=config.reflection_model.temperature,
                    max_tokens=config.reflection_model.max_tokens,
                )

                runner = GEMEpisodeRunner(
                    environment_id=config.environment.gem_env_id,
                    model_id=model_id,
                    temperature=config.eval_protocol.temperature_train,
                    max_steps=config.environment.max_steps,
                    max_response_tokens=config.eval_protocol.max_response_tokens,
                    top_p=config.eval_protocol.top_p,
                    top_k=config.eval_protocol.top_k,
                    seed=config.seeds.data_seed,
                    experiment_name=config.name,
                    tools=config.environment.active_tool_names,
                    experiment_run_id=exp_run_id,
                )
                # GEPA reuses one module for both trainset rollouts and
                # valset Pareto scoring. Mark the valset seeds so the module
                # forces greedy (eval temp) decoding on those calls — matches
                # GEM Table 3: train=1.0, evaluation=0.0.
                val_seed_set = frozenset(
                    int(seed)
                    for seed in (getattr(ex, "seed", None) for ex in valset)
                    if seed is not None
                )
                module = GEMSolverModule(
                    runner,
                    default_instructions=seed_prompt,
                    val_seeds=val_seed_set,
                    val_temperature=config.eval_protocol.temperature_eval,
                )

                gepa_log_dir = replication_dir / "gepa_logs"
                gepa_log_dir.mkdir(parents=True, exist_ok=True)

                enable_gepa_eval_progress()
                optimizer = dspy.GEPA(
                    metric=metric,
                    **gepa_budget_kwargs,
                    reflection_minibatch_size=config.gepa_budget.tasks_per_minibatch,
                    num_threads=settings.gepa.num_threads,
                    log_dir=str(gepa_log_dir),
                    track_stats=True,
                    seed=replication_seed,
                    reflection_lm=reflection_lm,
                )

                training_results: list[GEMEpisodeResult] = []
                training_rows: list[EpisodeRawMetrics] = []
                training_usage = _summarize_task_usage_rows([], metrics_unavailable_episodes=0)
                training_logging_summary = LoggingSummary()
                reflection_tokens: int | None = 0
                reflection_cost: float | None = 0.0
                baseline_eval_results: list[GEMEpisodeResult] = []
                eval_results: list[GEMEpisodeResult] = []
                try:
                    # --- Phase 1: GEPA optimization (skip if already done) ---
                    prior_status = _replication_status(replication_dir)
                    prompt_path = replication_dir / "optimized_prompt.txt"
                    resumed_from_gepa_done = prior_status == "gepa_done" and prompt_path.exists()

                    if resumed_from_gepa_done:
                        optimized_prompt = prompt_path.read_text(encoding="utf-8")
                        logger.info(
                            "Resuming from saved GEPA result model=%s seed=%s",
                            task_model.name,
                            replication_seed,
                        )
                        result = _load_gepa_result(prior_metadata)
                        training_rows = _load_training_metrics_rows(replication_dir)
                        training_usage = _summarize_task_usage_rows(
                            training_rows,
                            metrics_unavailable_episodes=0,
                        )

                        resume_phase_events: list[LoggingEvent] = []
                        saved_gepa_phase = prior_metadata.get("gepa_phase_summary")
                        if isinstance(saved_gepa_phase, dict):
                            saved_training_usage = saved_gepa_phase.get("training_usage")
                            if isinstance(saved_training_usage, dict):
                                training_usage.update(saved_training_usage)
                                training_usage.setdefault("episodes", len(training_rows))

                            saved_training_logging = saved_gepa_phase.get("logging_summary")
                            if isinstance(saved_training_logging, dict):
                                try:
                                    training_logging_summary = LoggingSummary.model_validate(
                                        saved_training_logging
                                    )
                                except Exception:  # noqa: BLE001
                                    resume_phase_events.append(
                                        LoggingEvent(
                                            stage="resume",
                                            kind="gepa_logging_summary_invalid",
                                            message=(
                                                "Saved GEPA logging summary could not be parsed "
                                                "during resume; later summaries may be incomplete."
                                            ),
                                        )
                                    )
                            else:
                                resume_phase_events.append(
                                    LoggingEvent(
                                        stage="resume",
                                        kind="gepa_logging_summary_missing",
                                        message=(
                                            "Saved GEPA logging summary missing during resume; "
                                            "later summaries may be incomplete."
                                        ),
                                    )
                                )

                            saved_reflection_tokens = saved_gepa_phase.get("reflection_tokens")
                            saved_reflection_cost = saved_gepa_phase.get("reflection_cost")
                            reflection_tokens = (
                                int(saved_reflection_tokens)
                                if isinstance(saved_reflection_tokens, int | float)
                                else None
                            )
                            reflection_cost = (
                                float(saved_reflection_cost)
                                if isinstance(saved_reflection_cost, int | float)
                                else None
                            )
                            if reflection_tokens is None or reflection_cost is None:
                                resume_phase_events.append(
                                    LoggingEvent(
                                        stage="resume",
                                        kind="gepa_reflection_usage_missing",
                                        message=(
                                            "Saved GEPA reflection usage missing during resume; "
                                            "final cost totals will remain partial."
                                        ),
                                    )
                                )
                        else:
                            training_logging_summary = LoggingSummary(
                                status="partial",
                                events=[
                                    LoggingEvent(
                                        stage="resume",
                                        kind="gepa_phase_summary_missing",
                                        message=(
                                            "Resumed from gepa_done without a saved GEPA phase "
                                            "summary; final logging and reflection totals may be "
                                            "incomplete."
                                        ),
                                    )
                                ],
                            )
                            reflection_tokens = None
                            reflection_cost = None

                        if resume_phase_events:
                            training_logging_summary = _merge_logging_summaries(
                                [
                                    training_logging_summary,
                                    LoggingSummary(status="partial", events=resume_phase_events),
                                ]
                            )
                    else:
                        compile_start = time.monotonic()
                        optimized_module = optimizer.compile(
                            student=module,
                            trainset=trainset,
                            valset=valset,
                        )
                        compile_elapsed = time.monotonic() - compile_start
                        logger.info(
                            "GEPA compile complete model=%s seed=%s elapsed=%.2fs",
                            task_model.name,
                            replication_seed,
                            compile_elapsed,
                        )

                        result = GEPARunResult.from_module(optimized_module, seed_prompt)
                        optimized_prompt = (
                            result.optimized_instructions
                            if result is not None
                            else getattr(optimized_module, "instructions", seed_prompt)
                        )

                        # Save GEPA artifacts immediately.
                        prompt_path.write_text(optimized_prompt, encoding="utf-8")
                        fitness_history = _extract_fitness_history(optimized_module)
                        _write_json(
                            replication_dir / "fitness_history.json",
                            {"history": fitness_history},
                        )
                        _write_json(
                            replication_dir / "pareto_frontier.json",
                            {"pareto_frontier": _extract_pareto_frontier(optimized_module)},
                        )
                        training_results = _persist_training_trajectories(
                            list(runner.episode_history),
                            experiment_run_id=exp_run_id,
                        )
                        training_rows = [
                            episode.raw_metrics
                            for episode in training_results
                            if episode.raw_metrics is not None
                        ]
                        training_usage = _summarize_task_usage(training_results)
                        training_logging_summary = _results_logging_summary(training_results)
                        reflection_tokens, reflection_cost = _extract_reflection_usage(
                            reflection_lm
                        )
                        _write_csv(
                            replication_dir / "training_metrics.csv",
                            training_rows,
                        )
                        _write_jsonl(
                            replication_dir / "training_metrics.jsonl",
                            training_rows,
                        )
                        _write_json(
                            replication_dir / "training_metrics_summary.json",
                            _raw_metrics_summary_for_results(training_results),
                        )

                        metadata["status"] = "gepa_done"
                        if result is not None:
                            metadata["result"] = result.model_dump(mode="json")
                        metadata["gepa_phase_summary"] = {
                            "training_usage": training_usage,
                            "logging_summary": training_logging_summary.model_dump(mode="json"),
                            "reflection_tokens": reflection_tokens,
                            "reflection_cost": (
                                round(reflection_cost, _COST_DECIMAL_PLACES)
                                if isinstance(reflection_cost, int | float)
                                else None
                            ),
                        }
                        _write_json(replication_dir / "run_metadata.json", metadata)
                        try:
                            update_experiment_run(
                                _DB_PATH,
                                exp_run_id,
                                status="gepa_done",
                                optimized_prompt=optimized_prompt,
                            )
                        except Exception:  # noqa: BLE001
                            logger.warning(
                                "update_experiment_run(gepa_done) failed",
                                exc_info=True,
                            )
                        logger.info(
                            "Saved GEPA artifacts model=%s seed=%s",
                            task_model.name,
                            replication_seed,
                        )

                    training_metrics_unavailable = training_usage.get(
                        "metrics_unavailable_episodes", 0
                    )
                    _write_json(
                        replication_dir / "training_metrics_summary.json",
                        _raw_metrics_summary_with_omissions(
                            training_rows,
                            metrics_unavailable_episodes=(
                                int(training_metrics_unavailable)
                                if isinstance(training_metrics_unavailable, int | float)
                                else 0
                            ),
                        ),
                    )

                    # --- Phase 2: Held-out evaluation ---
                    logger.info("Running baseline eval with seed prompt...")
                    baseline_eval_results, baseline_eval_summary = _run_heldout_eval(
                        config=config,
                        task_model_id=model_id,
                        instructions=seed_prompt,
                        experiment_run_id=exp_run_id,
                    )

                    logger.info("Running optimized eval...")
                    eval_results, eval_summary = _run_heldout_eval(
                        config=config,
                        task_model_id=model_id,
                        instructions=optimized_prompt,
                        experiment_run_id=exp_run_id,
                    )

                    heldout_results = baseline_eval_results + eval_results
                    heldout_logging_summary = _results_logging_summary(heldout_results)
                    logging_summary = _merge_logging_summaries(
                        [training_logging_summary, heldout_logging_summary]
                    )
                    val_total = config.environment.effective_val_size
                    baseline_validation_summary: dict[str, Any] | None = None
                    optimized_validation_summary: dict[str, Any] | None = None
                    if result is not None:
                        baseline_validation_summary = _build_validation_summary(
                            accuracy=result.baseline_accuracy,
                            episodes=val_total,
                        )
                        optimized_validation_summary = _build_validation_summary(
                            accuracy=result.final_accuracy,
                            episodes=val_total,
                        )
                    heldout_rows = [
                        result.raw_metrics
                        for result in heldout_results
                        if result.raw_metrics is not None
                    ]
                    _write_csv(replication_dir / "raw_metrics.csv", heldout_rows)
                    _write_jsonl(replication_dir / "raw_metrics.jsonl", heldout_rows)
                    _write_json(
                        replication_dir / "raw_metrics_summary.json",
                        {
                            "scope": "heldout_eval",
                            "baseline_eval": _raw_metrics_summary_for_results(
                                baseline_eval_results
                            ),
                            "optimized_eval": _raw_metrics_summary_for_results(eval_results),
                            "heldout_total": _raw_metrics_summary_for_results(heldout_results),
                        },
                    )

                    baseline_usage = _summarize_task_usage(baseline_eval_results)
                    eval_usage = _summarize_task_usage(eval_results)
                    task_usage = _merge_task_usage_summaries(
                        [training_usage, baseline_usage, eval_usage]
                    )

                    task_tokens = task_usage["total_tokens"]
                    task_tokens_known_value = task_usage["known_total_tokens"]
                    task_tokens_known = (
                        int(task_tokens_known_value)
                        if isinstance(task_tokens_known_value, int | float)
                        else 0
                    )
                    task_cost = task_usage["total_cost_usd"]
                    task_cost_known_value = task_usage["known_cost_usd"]
                    task_cost_known = (
                        float(task_cost_known_value)
                        if isinstance(task_cost_known_value, int | float)
                        else 0.0
                    )
                    total_tokens = (
                        int(task_tokens) + reflection_tokens
                        if isinstance(task_tokens, int) and isinstance(reflection_tokens, int)
                        else None
                    )
                    total_tokens_known = task_tokens_known + (
                        reflection_tokens if isinstance(reflection_tokens, int) else 0
                    )
                    total_cost_known = task_cost_known + (
                        reflection_cost if isinstance(reflection_cost, int | float) else 0.0
                    )
                    total_cost = (
                        round(total_cost_known, _COST_DECIMAL_PLACES)
                        if isinstance(task_cost, int | float)
                        and isinstance(reflection_cost, int | float)
                        else None
                    )
                    cost_type = (
                        "actual"
                        if total_cost is not None
                        else "partial"
                        if total_cost_known > 0
                        else "unavailable"
                    )

                    cost_summary = {
                        "task_model_tokens": task_tokens,
                        "task_model_tokens_known": task_tokens_known,
                        "task_model_token_data_coverage": task_usage["token_data_coverage"],
                        "task_model_metrics_unavailable_episodes": task_usage[
                            "metrics_unavailable_episodes"
                        ],
                        "task_model_cost": task_cost,
                        "task_model_cost_known": round(task_cost_known, _COST_DECIMAL_PLACES),
                        "task_model_cost_data_coverage": task_usage["cost_data_coverage"],
                        "reflection_tokens": reflection_tokens,
                        "reflection_cost": (
                            round(reflection_cost, _COST_DECIMAL_PLACES)
                            if isinstance(reflection_cost, int | float)
                            else None
                        ),
                        "total_tokens": total_tokens,
                        "total_tokens_known": total_tokens_known,
                        "total_cost": total_cost,
                        "total_cost_known": round(total_cost_known, _COST_DECIMAL_PLACES),
                        "cost_type": cost_type,
                        "effective_budget_usd": config.cost_budget.effective_budget_usd,
                        "training_task_model_tokens": training_usage["total_tokens"],
                        "training_task_model_tokens_known": training_usage["known_total_tokens"],
                        "training_task_model_token_data_coverage": training_usage[
                            "token_data_coverage"
                        ],
                        "training_metrics_unavailable_episodes": training_usage[
                            "metrics_unavailable_episodes"
                        ],
                        "training_task_model_cost": training_usage["total_cost_usd"],
                        "training_task_model_cost_known": training_usage["known_cost_usd"],
                        "training_task_model_cost_data_coverage": training_usage[
                            "cost_data_coverage"
                        ],
                        "baseline_eval_task_model_tokens": baseline_usage["total_tokens"],
                        "baseline_eval_task_model_tokens_known": baseline_usage[
                            "known_total_tokens"
                        ],
                        "baseline_eval_task_model_token_data_coverage": baseline_usage[
                            "token_data_coverage"
                        ],
                        "baseline_eval_metrics_unavailable_episodes": baseline_usage[
                            "metrics_unavailable_episodes"
                        ],
                        "baseline_eval_task_model_cost": baseline_usage["total_cost_usd"],
                        "baseline_eval_task_model_cost_known": baseline_usage["known_cost_usd"],
                        "baseline_eval_task_model_cost_data_coverage": baseline_usage[
                            "cost_data_coverage"
                        ],
                        "eval_task_model_tokens": eval_usage["total_tokens"],
                        "eval_task_model_tokens_known": eval_usage["known_total_tokens"],
                        "eval_task_model_token_data_coverage": eval_usage["token_data_coverage"],
                        "eval_metrics_unavailable_episodes": eval_usage[
                            "metrics_unavailable_episodes"
                        ],
                        "eval_task_model_cost": eval_usage["total_cost_usd"],
                        "eval_task_model_cost_known": eval_usage["known_cost_usd"],
                        "eval_task_model_cost_data_coverage": eval_usage["cost_data_coverage"],
                    }
                    _write_json(replication_dir / "cost_summary.json", cost_summary)

                    alert_level = config.cost_budget.effective_budget_usd * _budget_alert_fraction()
                    if isinstance(total_cost, int | float) and total_cost >= alert_level:
                        logger.warning(
                            "Cost alert model=%s seed=%s total_cost=%.6f budget=%.6f",
                            task_model.name,
                            replication_seed,
                            total_cost,
                            config.cost_budget.effective_budget_usd,
                        )
                    elif cost_type != "actual" and total_cost_known >= alert_level:
                        logger.warning(
                            "Known cost reached alert level but actual cost is incomplete "
                            "model=%s seed=%s known_total_cost=%.6f budget=%.6f cost_type=%s",
                            task_model.name,
                            replication_seed,
                            total_cost_known,
                            config.cost_budget.effective_budget_usd,
                            cost_type,
                        )

                    if (
                        isinstance(total_cost, int | float)
                        and total_cost > config.cost_budget.effective_budget_usd
                    ):
                        logger.warning(
                            "Cost budget exceeded model=%s seed=%s total_cost=%.6f budget=%.6f",
                            task_model.name,
                            replication_seed,
                            total_cost,
                            config.cost_budget.effective_budget_usd,
                        )
                        if args.halt_on_budget_exceeded:
                            raise RuntimeError(
                                "Cost budget exceeded and halt_on_budget_exceeded=True"
                            )

                    finished = _utc_now()
                    metadata["logging_summary"] = logging_summary.model_dump(mode="json")
                    metadata.update(
                        {
                            "status": "completed",
                            "finished_at": _format_iso(finished),
                            "elapsed_seconds": round((finished - started_at).total_seconds(), 3),
                            "result": result.model_dump(mode="json")
                            if result is not None
                            else None,
                            "baseline_validation": baseline_validation_summary,
                            "optimized_validation": optimized_validation_summary,
                            "baseline_eval": baseline_eval_summary,
                            "eval": eval_summary,
                        }
                    )
                    _write_json(replication_dir / "run_metadata.json", metadata)

                    # Update experiment_run DB record to "completed".
                    try:
                        update_experiment_run(
                            _DB_PATH,
                            exp_run_id,
                            status="completed",
                            finished_at=_format_iso(finished),
                            result_summary=eval_summary,
                            cost_summary=cost_summary,
                            logging_summary=logging_summary,
                        )
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "update_experiment_run(completed) failed",
                            exc_info=True,
                        )

                    # Generate unified run report.
                    try:
                        run_report = build_run_report(
                            experiment_run_id=exp_run_id,
                            db_path=_DB_PATH,
                            cost_summary=cost_summary,
                            baseline_validation_summary=baseline_validation_summary,
                            optimized_validation_summary=optimized_validation_summary,
                            baseline_eval_summary=baseline_eval_summary,
                            eval_summary=eval_summary,
                            wall_clock_seconds=round((finished - started_at).total_seconds(), 3),
                            reference_prices=settings.cost_normalization.reference_prices,
                            prompt_token_ratio=settings.cost_normalization.prompt_token_ratio,
                            logging_summary=logging_summary.model_dump(mode="json"),
                        )
                        _write_json(
                            replication_dir / "run_report.json",
                            json.loads(run_report.model_dump_json()),
                        )
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "Failed to generate run report for %s",
                            exp_run_id,
                            exc_info=True,
                        )

                    model_entries[str(replication_seed)] = {
                        "status": "completed",
                        "experiment_run_id": exp_run_id,
                        "cost_summary": cost_summary,
                        "baseline_validation": baseline_validation_summary,
                        "optimized_validation": optimized_validation_summary,
                        "baseline_eval": baseline_eval_summary,
                        "eval": eval_summary,
                        "logging_summary": logging_summary.model_dump(mode="json"),
                    }

                    train_bench = config.environment.gem_env_id.split(":")[-1]
                    eval_bench = (
                        config.environment.dataset.eval_split
                        if config.environment.dataset
                        else train_bench
                    )
                    if (
                        baseline_validation_summary is not None
                        and optimized_validation_summary is not None
                    ):
                        validation_lines = (
                            f"  Validation: Baseline accuracy:  "
                            f"{baseline_validation_summary['accuracy']:.1%} "
                            f"({baseline_validation_summary['correct']}/{val_total}) "
                            f"({train_bench})\n"
                            f"  Validation: Optimized accuracy: "
                            f"{optimized_validation_summary['accuracy']:.1%} "
                            f"({optimized_validation_summary['correct']}/{val_total}) "
                            f"({train_bench})\n"
                        )
                    else:
                        validation_lines = (
                            f"  Validation: Baseline accuracy:  N/A ({train_bench})\n"
                            f"  Validation: Optimized accuracy: N/A ({train_bench})\n"
                        )
                    bl_eval_acc = baseline_eval_summary["accuracy"]
                    bl_eval_correct = baseline_eval_summary["correct"]
                    bl_eval_total = baseline_eval_summary["episodes"]
                    eval_acc = eval_summary["accuracy"]
                    eval_correct = eval_summary["correct"]
                    eval_total = eval_summary["episodes"]
                    total_cost_display = (
                        f"${total_cost:.4f}"
                        if isinstance(total_cost, int | float)
                        else f"unknown (known ${total_cost_known:.4f}, {cost_type})"
                    )
                    total_tokens_display = (
                        f"{total_tokens:,}"
                        if isinstance(total_tokens, int)
                        else f"unknown (known {total_tokens_known:,})"
                    )
                    print(
                        f"\n{'=' * 60}\n"
                        f"  Replication complete: {task_model.name} seed={replication_seed}\n"
                        f"{validation_lines}"
                        f"  Test/eval: Baseline accuracy:  {bl_eval_acc:.1%} "
                        f"({bl_eval_correct}/{bl_eval_total}) ({eval_bench})\n"
                        f"  Test/eval: Optimized accuracy: {eval_acc:.1%} "
                        f"({eval_correct}/{eval_total}) ({eval_bench})\n"
                        f"  Total cost:         {total_cost_display}\n"
                        f"  Total tokens:       {total_tokens_display}\n"
                        f"  Elapsed:            {(finished - started_at).total_seconds():.1f}s\n"
                        f"{'=' * 60}",
                        flush=True,
                    )
                except Exception as exc:
                    finished = _utc_now()
                    partial_logging_summary = _merge_logging_summaries(
                        [
                            training_logging_summary,
                            _results_logging_summary(baseline_eval_results + eval_results),
                        ]
                    )
                    metadata.update(
                        {
                            "status": "failed",
                            "finished_at": _format_iso(finished),
                            "elapsed_seconds": round((finished - started_at).total_seconds(), 3),
                            "error": repr(exc),
                            "logging_summary": partial_logging_summary.model_dump(mode="json"),
                        }
                    )
                    _write_json(replication_dir / "run_metadata.json", metadata)
                    try:
                        update_experiment_run(
                            _DB_PATH,
                            exp_run_id,
                            status="failed",
                            finished_at=_format_iso(finished),
                            error_summary=repr(exc),
                            logging_summary=partial_logging_summary,
                        )
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "update_experiment_run(failed) failed",
                            exc_info=True,
                        )
                    model_entries[str(replication_seed)] = {
                        "status": "failed",
                        "experiment_run_id": exp_run_id,
                        "error": repr(exc),
                        "logging_summary": partial_logging_summary.model_dump(mode="json"),
                    }
                    logger.exception(
                        "Replication failed model=%s seed=%s (fail_fast=%s)",
                        task_model.name,
                        replication_seed,
                        args.fail_fast,
                    )
                    if args.fail_fast:
                        # Persist run_summary before bubbling so postmortem
                        # tools always see finished_at/elapsed_seconds.
                        _finalize_run_summary()
                        raise

    return _finalize_run_summary()
