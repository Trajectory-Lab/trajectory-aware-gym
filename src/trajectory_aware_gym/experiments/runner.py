"""Production experiment runner with replication and cost tracking."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import dspy  # type: ignore[import-untyped]
import yaml
from litellm import completion_cost  # type: ignore[import-untyped]

from trajectory_aware_gym.adapters.dspy_adapter import TrajectoryFitnessMetric
from trajectory_aware_gym.adapters.gem_episode_runner import GEMEpisodeResult, GEMEpisodeRunner
from trajectory_aware_gym.adapters.gem_solver_module import GEMSolverModule
from trajectory_aware_gym.config import settings
from trajectory_aware_gym.config.core import FitnessModel
from trajectory_aware_gym.config.llm_provider import (
    TaskModelName,
    get_reflection_lm,
    get_task_lm,
    get_task_model_id,
)
from trajectory_aware_gym.metrics import EpisodeRawMetrics
from trajectory_aware_gym.models.experiment import (
    ExperimentConfig,
    FitnessOverride,
    TaskModelConfig,
)
from trajectory_aware_gym.models.gepa_result import GEPARunResult

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_SEED_PROMPT = (
    "You are a math problem solver. "
    "Solve the problem step by step, then give your final answer "
    "inside \\boxed{}.  For example: \\boxed{42}"
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunExperimentArgs:
    """CLI-resolved arguments for a production experiment run."""

    config_path: Path
    max_metric_calls: int | None = None
    seed_prompt: str = DEFAULT_SEED_PROMPT
    models: tuple[str, ...] | None = None
    seeds: tuple[int, ...] | None = None
    fresh: bool = False
    results_root: Path = DEFAULT_RESULTS_ROOT
    halt_on_budget_exceeded: bool = False


def derive_max_metric_calls(config: ExperimentConfig, override: int | None = None) -> int:
    """Derive GEPA metric call budget from config, with optional override."""
    if override is not None:
        if override < 1:
            raise ValueError("max_metric_calls override must be >= 1")
        return override

    budget = config.gepa_budget
    upper_bound = budget.iterations * budget.population_size * budget.tasks_per_minibatch
    return max(1, upper_bound)


def select_task_models(config: ExperimentConfig, selected_names: tuple[str, ...] | None) -> list[TaskModelConfig]:
    """Filter configured task models by optional name allow-list."""
    if not selected_names:
        return list(config.task_models)

    selected = set(selected_names)
    models = [model for model in config.task_models if model.name in selected]
    if not models:
        raise ValueError(f"No configured task_models matched requested models: {sorted(selected)}")
    return models


def select_replication_seeds(config: ExperimentConfig, selected_seeds: tuple[int, ...] | None) -> tuple[int, ...]:
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


def _iso(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json", by_alias=True, exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _raw_metrics_summary(rows: list[EpisodeRawMetrics]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "episodes": 0,
            "successes": 0,
            "mean_latency_seconds": None,
            "total_cost": None,
            "total_tokens": None,
            "mean_cost_data_coverage": None,
            "mean_token_data_coverage": None,
            "mean_llm_latency_data_coverage": None,
        }

    episodes = len(rows)
    successes = sum(1 for row in rows if row.success)
    latencies = [row.episode_latency_seconds for row in rows]
    costs = [row.llm_cost_usd for row in rows if row.llm_cost_usd is not None]
    tokens = [row.total_tokens for row in rows if row.total_tokens is not None]

    return {
        "episodes": episodes,
        "successes": successes,
        "mean_latency_seconds": round(sum(latencies) / episodes, 6),
        "total_cost": round(sum(costs), 6) if costs else None,
        "total_tokens": int(sum(tokens)) if tokens else None,
        "mean_cost_data_coverage": round(sum(r.cost_data_coverage for r in rows) / episodes, 6),
        "mean_token_data_coverage": round(sum(r.token_data_coverage for r in rows) / episodes, 6),
        "mean_llm_latency_data_coverage": round(
            sum(r.llm_latency_data_coverage for r in rows) / episodes,
            6,
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
            except (KeyError, TypeError, ValueError):
                maybe_cost = None
            if isinstance(maybe_cost, int | float):
                total_cost += float(maybe_cost)

    return (total_tokens, total_cost)


def _extract_task_usage(results: list[GEMEpisodeResult]) -> tuple[int, float]:
    tokens = sum(result.trajectory.total_tokens for result in results)
    cost = sum(result.trajectory.total_cost_usd for result in results)
    return (tokens, cost)


def _extract_fitness_history(optimized_module: Any) -> list[dict[str, Any]]:
    detailed = getattr(optimized_module, "detailed_results", None)
    if detailed is None:
        return []

    history = getattr(detailed, "history", None)
    if isinstance(history, list):
        return [item for item in history if isinstance(item, dict)]

    aggregate = getattr(detailed, "val_aggregate_scores", None)
    if isinstance(aggregate, list):
        rows: list[dict[str, Any]] = []
        for index, score in enumerate(aggregate):
            if isinstance(score, int | float):
                rows.append({"index": index, "val_aggregate_score": float(score)})
        return rows
    return []


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


def _completed(replication_dir: Path) -> bool:
    metadata_path = replication_dir / "run_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


def _resolve_task_model_id(task_model: TaskModelConfig) -> str:
    if task_model.provider in ("bedrock", "sagemaker"):
        model_id = get_task_model_id(cast(TaskModelName, task_model.name))
        if model_id is None:
            raise ValueError(f"Unable to resolve model id for task model: {task_model.name}")
        return model_id
    return task_model.model_id


def _build_task_lm(config: ExperimentConfig, task_model: TaskModelConfig) -> Any:
    if task_model.provider in ("bedrock", "sagemaker"):
        task_lm = get_task_lm(cast(TaskModelName, task_model.name), mode="train")
        if task_lm is None:
            raise ValueError(f"Unable to build task LM for {task_model.name}")
        return task_lm

    kwargs: dict[str, Any] = {
        "model": task_model.model_id,
        "temperature": config.eval_protocol.temperature_train,
        "max_tokens": config.eval_protocol.max_response_tokens,
    }
    if task_model.provider == "ollama":
        kwargs["api_base"] = settings.ollama.api_base
    return dspy.LM(**kwargs)


def _build_trainset(config: ExperimentConfig) -> list[dspy.Example]:
    import importlib

    gem = importlib.import_module("gem")
    importlib.import_module("gem.envs")
    env = gem.make(config.environment.gem_env_id)

    examples: list[dspy.Example] = []
    for index in range(config.environment.train_size):
        seed = config.seeds.data_seed + index
        observation, _ = env.reset(seed=seed)
        examples.append(
            dspy.Example(problem=str(observation), seed=seed).with_inputs("problem", "seed")
        )

    if hasattr(env, "close"):
        env.close()
    return examples


@contextmanager
def _fitness_override_context(override: FitnessOverride) -> Any:
    base = settings.fitness.model_dump(mode="json", by_alias=True)
    patch = override.model_dump(mode="json", by_alias=True, exclude_none=True)
    if not patch:
        yield
        return

    merged = {**base, **patch}
    type(settings)._fitness = FitnessModel(**merged)
    try:
        yield
    finally:
        type(settings)._fitness = FitnessModel(**base)


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
) -> Path:
    return (
        args.results_root
        / _safe_segment(config.name)
        / _safe_segment(task_model.name)
        / f"replication_{replication_seed}"
    )


def _eval_examples(config: ExperimentConfig) -> list[dspy.Example]:
    import importlib

    gem = importlib.import_module("gem")
    importlib.import_module("gem.envs")
    env = gem.make(config.environment.gem_env_id)

    examples: list[dspy.Example] = []
    start_seed = config.seeds.data_seed + config.environment.train_size
    eval_size = config.environment.effective_val_size
    for index in range(eval_size):
        seed = start_seed + index
        observation, _ = env.reset(seed=seed)
        examples.append(
            dspy.Example(problem=str(observation), seed=seed).with_inputs("problem", "seed")
        )

    if hasattr(env, "close"):
        env.close()
    return examples


def _run_heldout_eval(
    *,
    config: ExperimentConfig,
    task_model_id: str,
    instructions: str,
) -> tuple[list[GEMEpisodeResult], dict[str, Any]]:
    eval_runner = GEMEpisodeRunner(
        environment_id=config.environment.gem_env_id,
        model_id=task_model_id,
        temperature=config.eval_protocol.temperature_eval,
        max_steps=config.environment.max_steps,
        max_response_tokens=config.eval_protocol.max_response_tokens,
        seed=config.seeds.data_seed + config.environment.train_size,
        experiment_name=config.name,
        tools=[tool.value for tool in config.environment.tools if tool.value != "none"],
    )

    eval_examples = _eval_examples(config)
    rollouts = config.eval_protocol.rollouts_per_task

    results: list[GEMEpisodeResult] = []
    successes = 0
    for example_index, example in enumerate(eval_examples):
        for rollout_index in range(rollouts):
            seed = int(example.seed) + rollout_index
            result = eval_runner.run_episode(
                instructions,
                episode_index=example_index * rollouts + rollout_index,
                seed_override=seed,
                expected_observation=str(example.problem),
                persist=False,
            )
            results.append(result)
            if result.raw_metrics.success:
                successes += 1

    total = len(results)
    summary = {
        "episodes": total,
        "successes": successes,
        "success_rate": (successes / total) if total else 0.0,
        "temperature_eval": config.eval_protocol.temperature_eval,
    }
    return (results, summary)


def run_experiment(args: RunExperimentArgs) -> dict[str, Any]:
    """Run a production GEPA experiment across configured models and replications."""
    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config = ExperimentConfig.from_yaml(args.config_path)
    models = select_task_models(config, args.models)
    replication_seeds = select_replication_seeds(config, args.seeds)
    max_metric_calls = derive_max_metric_calls(config, args.max_metric_calls)

    if args.fresh:
        fresh_target = args.results_root / _safe_segment(config.name)
        if fresh_target.exists():
            logger.info("Removing existing results directory: %s", fresh_target)
            shutil.rmtree(fresh_target)

    trainset = _build_trainset(config)
    val_size = config.environment.effective_val_size
    valset = trainset[:val_size] if val_size <= len(trainset) else trainset

    global_started = _utc_now()
    run_id = f"{_safe_segment(config.name)}-{global_started.strftime('%Y%m%dT%H%M%SZ')}"
    config_hash = _config_hash(config)
    git_commit = _git_commit_hash()

    overall: dict[str, Any] = {
        "run_id": run_id,
        "config": config.name,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "started_at": _iso(global_started),
        "max_metric_calls": max_metric_calls,
        "models": {},
    }

    with _fitness_override_context(config.fitness_override):
        metric = TrajectoryFitnessMetric(return_feedback=True)

        for task_model in models:
            model_id = _resolve_task_model_id(task_model)
            logger.info("Starting model: %s (%s)", task_model.name, model_id)
            model_entries: dict[str, Any] = {}
            overall["models"][task_model.name] = model_entries

            for replication_seed in replication_seeds:
                replication_dir = _model_replication_dir(args, config, task_model, replication_seed)
                replication_dir.mkdir(parents=True, exist_ok=True)

                if _completed(replication_dir):
                    logger.info(
                        "Skipping completed replication model=%s seed=%s",
                        task_model.name,
                        replication_seed,
                    )
                    model_entries[str(replication_seed)] = {"status": "skipped"}
                    continue

                started_at = _utc_now()
                metadata = {
                    "run_id": run_id,
                    "status": "running",
                    "config_hash": config_hash,
                    "git_commit": git_commit,
                    "started_at": _iso(started_at),
                    "finished_at": None,
                    "seed": replication_seed,
                    "model_name": task_model.name,
                    "model_id": model_id,
                    "max_metric_calls": max_metric_calls,
                }
                _write_json(replication_dir / "run_metadata.json", metadata)

                config_snapshot = config.model_dump(mode="json", by_alias=True, exclude_none=True)
                (replication_dir / "config_snapshot.yaml").write_text(
                    yaml.dump(config_snapshot, default_flow_style=False, sort_keys=False),
                    encoding="utf-8",
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
                    seed=config.seeds.data_seed,
                    experiment_name=config.name,
                    tools=[tool.value for tool in config.environment.tools if tool.value != "none"],
                )
                module = GEMSolverModule(runner, default_instructions=args.seed_prompt)

                gepa_log_dir = replication_dir / "gepa_logs"
                gepa_log_dir.mkdir(parents=True, exist_ok=True)

                optimizer = dspy.GEPA(
                    metric=metric,
                    max_metric_calls=max_metric_calls,
                    num_threads=settings.gepa.num_threads,
                    log_dir=str(gepa_log_dir),
                    track_stats=True,
                    seed=replication_seed,
                    reflection_lm=reflection_lm,
                )

                try:
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

                    result = GEPARunResult.from_module(optimized_module, args.seed_prompt)
                    optimized_prompt = (
                        result.optimized_instructions
                        if result is not None
                        else getattr(optimized_module, "instructions", args.seed_prompt)
                    )
                    (replication_dir / "optimized_prompt.txt").write_text(
                        optimized_prompt,
                        encoding="utf-8",
                    )

                    fitness_history = _extract_fitness_history(optimized_module)
                    _write_json(replication_dir / "fitness_history.json", {"history": fitness_history})
                    _write_json(
                        replication_dir / "pareto_frontier.json",
                        {"pareto_frontier": _extract_pareto_frontier(optimized_module)},
                    )

                    training_results = list(runner.episode_history)
                    eval_results, eval_summary = _run_heldout_eval(
                        config=config,
                        task_model_id=model_id,
                        instructions=optimized_prompt,
                    )

                    all_rows = [r.raw_metrics for r in training_results] + [
                        r.raw_metrics for r in eval_results
                    ]
                    _write_csv(replication_dir / "raw_metrics.csv", all_rows)
                    _write_jsonl(replication_dir / "raw_metrics.jsonl", all_rows)
                    _write_json(
                        replication_dir / "raw_metrics_summary.json",
                        _raw_metrics_summary(all_rows),
                    )

                    task_tokens_train, task_cost_train = _extract_task_usage(training_results)
                    task_tokens_eval, task_cost_eval = _extract_task_usage(eval_results)
                    reflection_tokens, reflection_cost = _extract_reflection_usage(reflection_lm)

                    task_tokens = task_tokens_train + task_tokens_eval
                    task_cost = task_cost_train + task_cost_eval
                    total_tokens = task_tokens + reflection_tokens
                    total_cost = task_cost + reflection_cost

                    cost_summary = {
                        "task_model_tokens": task_tokens,
                        "task_model_cost": round(task_cost, 6),
                        "reflection_tokens": reflection_tokens,
                        "reflection_cost": round(reflection_cost, 6),
                        "total_tokens": total_tokens,
                        "total_cost": round(total_cost, 6),
                        "effective_budget_usd": config.cost_budget.effective_budget_usd,
                        "training_task_model_tokens": task_tokens_train,
                        "training_task_model_cost": round(task_cost_train, 6),
                        "eval_task_model_tokens": task_tokens_eval,
                        "eval_task_model_cost": round(task_cost_eval, 6),
                    }
                    _write_json(replication_dir / "cost_summary.json", cost_summary)

                    alert_level = config.cost_budget.effective_budget_usd * _budget_alert_fraction()
                    if total_cost >= alert_level:
                        logger.warning(
                            "Cost alert model=%s seed=%s total_cost=%.6f budget=%.6f",
                            task_model.name,
                            replication_seed,
                            total_cost,
                            config.cost_budget.effective_budget_usd,
                        )

                    if total_cost > config.cost_budget.effective_budget_usd:
                        logger.warning(
                            "Cost budget exceeded model=%s seed=%s total_cost=%.6f budget=%.6f",
                            task_model.name,
                            replication_seed,
                            total_cost,
                            config.cost_budget.effective_budget_usd,
                        )
                        if args.halt_on_budget_exceeded:
                            raise RuntimeError("Cost budget exceeded and halt_on_budget_exceeded=True")

                    finished = _utc_now()
                    metadata.update(
                        {
                            "status": "completed",
                            "finished_at": _iso(finished),
                            "elapsed_seconds": round((finished - started_at).total_seconds(), 3),
                            "result": result.model_dump(mode="json") if result is not None else None,
                            "eval": eval_summary,
                        }
                    )
                    _write_json(replication_dir / "run_metadata.json", metadata)

                    model_entries[str(replication_seed)] = {
                        "status": "completed",
                        "cost_summary": cost_summary,
                        "eval": eval_summary,
                    }
                except Exception:
                    finished = _utc_now()
                    metadata.update(
                        {
                            "status": "failed",
                            "finished_at": _iso(finished),
                            "elapsed_seconds": round((finished - started_at).total_seconds(), 3),
                        }
                    )
                    _write_json(replication_dir / "run_metadata.json", metadata)
                    logger.exception(
                        "Replication failed model=%s seed=%s",
                        task_model.name,
                        replication_seed,
                    )
                    raise

    finished_at = _utc_now()
    overall["finished_at"] = _iso(finished_at)
    overall["elapsed_seconds"] = round((finished_at - global_started).total_seconds(), 3)

    experiment_summary_path = args.results_root / _safe_segment(config.name) / "run_summary.json"
    _write_json(experiment_summary_path, overall)
    return overall
