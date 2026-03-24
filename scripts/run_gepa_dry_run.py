"""End-to-end GEPA dry-run: DSPy GEPA + GEM + trajectory fitness.

Loads an experiment config, builds a minimal trainset/valset from the GEM
environment, creates a GEMSolverModule + TrajectoryFitnessMetric, and runs
dspy.GEPA.compile() for one iteration on a tiny population.

This is the K4 deliverable — a successful small-scale full pipeline run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import cast

import dspy  # type: ignore[import-untyped]

from trajectory_aware_gym.adapters.dspy_adapter import TrajectoryFitnessMetric
from trajectory_aware_gym.adapters.gem_episode_runner import GEMEpisodeRunner
from trajectory_aware_gym.adapters.gem_solver_module import GEMSolverModule
from trajectory_aware_gym.config.llm_provider import (
    TaskModelName,
    get_reflection_lm,
    get_task_lm,
    get_task_model_id,
)
from trajectory_aware_gym.models.experiment import ExperimentConfig
from trajectory_aware_gym.models.gepa_result import GEPARunResult

DEFAULT_SEED_PROMPT = (
    "You are a math problem solver. "
    "Solve the problem step by step, then give your final answer "
    "inside \\boxed{}.  For example: \\boxed{42}"
)

LOG_DIR = Path("logs/gepa-dry-run")
# Dry-run-only cap — do NOT promote to the experiment runner.
# Real experiments should use the value from ExperimentConfig.
DRY_RUN_MAX_METRIC_CALLS = 32


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path)


def build_trainset(config: ExperimentConfig) -> list[dspy.Example]:
    """Build a reproducible trainset bound to specific GEM reset seeds."""
    import importlib

    gem = importlib.import_module("gem")
    importlib.import_module("gem.envs")

    env = gem.make(config.environment.gem_env_id)
    examples: list[dspy.Example] = []
    for i in range(config.environment.train_size):
        seed = config.seeds.data_seed + i
        observation, _info = env.reset(seed=seed)
        examples.append(
            dspy.Example(problem=str(observation), seed=seed).with_inputs("problem", "seed")
        )
    if hasattr(env, "close"):
        env.close()
    return examples


def build_runner(config: ExperimentConfig) -> GEMEpisodeRunner:
    task_model = config.task_models[0]
    tools = [t.value for t in config.environment.tools if t.value != "none"]
    model_id = task_model.model_id
    if task_model.provider == "bedrock":
        model_id = get_task_model_id(cast(TaskModelName, task_model.name))
    return GEMEpisodeRunner(
        environment_id=config.environment.gem_env_id,
        model_id=model_id,
        temperature=config.eval_protocol.temperature_train,
        max_steps=config.environment.max_steps,
        max_response_tokens=config.eval_protocol.max_response_tokens,
        seed=config.seeds.data_seed,
        experiment_name=config.name,
        tools=tools,
    )


def run_dry_run(
    config_path: Path,
    *,
    seed_prompt: str = DEFAULT_SEED_PROMPT,
    fresh: bool = False,
) -> dict:
    config = load_config(config_path)
    print(f"Loaded config: {config.name}")
    print(f"  Environment: {config.environment.gem_env_id}")
    print(f"  Task model:  {config.task_models[0].model_id}")
    print(
        "  GEPA budget: "
        f"max_metric_calls={DRY_RUN_MAX_METRIC_CALLS} "
        f"(config mode={config.gepa_budget.mode}, "
        f"iters={config.gepa_budget.iterations}, "
        f"pop={config.gepa_budget.population_size})"
    )

    # Build components
    runner = build_runner(config)
    module = GEMSolverModule(runner, default_instructions=seed_prompt)
    metric = TrajectoryFitnessMetric(return_feedback=True)

    # Build train/val sets
    print("\nBuilding trainset from GEM environment resets...")
    trainset = build_trainset(config)
    val_size = config.environment.effective_val_size
    valset = trainset[:val_size] if val_size <= len(trainset) else trainset
    print(f"  Train examples: {len(trainset)}, Val examples: {len(valset)}")
    for index, example in enumerate(trainset[: min(3, len(trainset))], start=1):
        question_preview = str(example.problem).replace("\n", " ")[:120]
        print(f"    Sample {index}: seed={example.seed} problem={question_preview}")

    # Resolve num_threads from centralized config
    from trajectory_aware_gym.config import settings as settings

    num_threads = settings.gepa.num_threads

    # Configure DSPy LM for the task model
    task_model_cfg = config.task_models[0]

    if task_model_cfg.provider == "bedrock":
        task_lm = get_task_lm(
            model=cast(TaskModelName, task_model_cfg.name),
            mode="train",
        )
    else:
        task_lm_kwargs: dict = {
            "model": task_model_cfg.model_id,
            "temperature": config.eval_protocol.temperature_train,
            "max_tokens": config.eval_protocol.max_response_tokens,
        }
        if task_model_cfg.provider == "ollama":
            task_lm_kwargs["api_base"] = settings.ollama.api_base
        task_lm = dspy.LM(**task_lm_kwargs)
    dspy.configure(lm=task_lm)

    # Build reflection LM from the experiment config so the run uses the same
    # model it prints and records in artifacts.
    reflection_lm = None
    try:
        reflection_lm = get_reflection_lm(
            config.reflection_model.model_id,
            temperature=config.reflection_model.temperature,
            max_tokens=config.reflection_model.max_tokens,
        )
        print(f"  Reflection LM: {config.reflection_model.model_id}")
    except Exception as e:
        print(f"  Reflection LM unavailable ({e}); GEPA will use task LM for reflection")

    # Set up GEPA optimizer
    if fresh and LOG_DIR.exists():
        print(f"\nRemoving existing GEPA run directory: {LOG_DIR}")
        shutil.rmtree(LOG_DIR)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    gepa_kwargs: dict = {
        "metric": metric,
        "max_metric_calls": DRY_RUN_MAX_METRIC_CALLS,
        "num_threads": num_threads,
        "log_dir": str(LOG_DIR),
        "track_stats": True,
        "seed": config.seeds.data_seed,
    }
    if reflection_lm is not None:
        gepa_kwargs["reflection_lm"] = reflection_lm

    optimizer = dspy.GEPA(**gepa_kwargs)

    # Run GEPA compile
    print(
        "\nStarting GEPA compile "
        f"(max_metric_calls={DRY_RUN_MAX_METRIC_CALLS}, "
        f"num_threads={num_threads})..."
    )
    start_time = time.monotonic()
    optimized_module = optimizer.compile(
        student=module,
        trainset=trainset,
        valset=valset,
    )
    elapsed = time.monotonic() - start_time

    # Extract results
    result = GEPARunResult.from_module(optimized_module, seed_prompt)

    print(f"\nGEPA compile completed in {elapsed:.1f}s")
    if result is not None:
        print(
            f"  Baseline fitness: {result.baseline_fitness:.4f}  |  "
            f"Final fitness: {result.final_fitness:.4f} (program {result.best_program_index})"
        )
        print(
            f"  Baseline accuracy: {result.baseline_accuracy:.2%}  |  "
            f"Final accuracy: {result.final_accuracy:.2%}"
        )
        print(f"  Optimized instructions ({len(result.optimized_instructions)} chars):")
        preview = result.optimized_instructions[:200]
        ellipsis = "..." if len(result.optimized_instructions) > 200 else ""
        print(f"    {preview}{ellipsis}")

    # Save summary
    summary = {
        "config_name": config.name,
        "environment": config.environment.gem_env_id,
        "task_model": config.task_models[0].model_id,
        "gepa_budget": {
            "config_mode": config.gepa_budget.mode,
            "max_metric_calls": DRY_RUN_MAX_METRIC_CALLS,
            "num_threads": num_threads,
        },
        "train_size": len(trainset),
        "val_size": len(valset),
        "seed_prompt": seed_prompt,
        "elapsed_seconds": round(elapsed, 2),
        **(
            {
                "baseline_fitness": result.baseline_fitness,
                "final_fitness": result.final_fitness,
                "baseline_accuracy": result.baseline_accuracy,
                "final_accuracy": result.final_accuracy,
                "optimized_instructions": result.optimized_instructions,
            }
            if result is not None
            else {}
        ),
    }
    summary_path = LOG_DIR / "dry_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary saved to: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA dry-run (K4 full pipeline)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/math-dry-run/config.yaml"),
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--seed-prompt",
        type=str,
        default=DEFAULT_SEED_PROMPT,
        help="Initial seed prompt for GEPA optimization",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove the existing GEPA log directory before running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dry_run(args.config, seed_prompt=args.seed_prompt, fresh=args.fresh)


if __name__ == "__main__":
    main()
