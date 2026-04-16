"""Diagnostic: run a handful of orz57k episodes with python_exec and dump
exactly what the model submits to the tool, what comes back, and the final
action. Use this when we suspect the tool description / seed prompt is the
reason tool runs under-perform the no-tool baseline.

Usage:

    uv run python scripts/diagnose_python_exec.py --num-episodes 8

Writes a text report to stdout and a JSON dump to
``results/_diagnostics/python_exec_<timestamp>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from trajectory_aware_gym.adapters.gem_episode_runner import (
    GEMEpisodeRunner,
)
from trajectory_aware_gym.adapters.tool_runtime import ToolRuntime
from trajectory_aware_gym.config import ProjectPaths
from trajectory_aware_gym.models.experiment import ExperimentConfig

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "experiments" / "orz57k-tool" / "config.yaml"
_DEFAULT_MODEL_NAME = "Llama-3.1-8B-Instruct"
_DEFAULT_NUM_EPISODES = 8


def _resolve_task_model(config: ExperimentConfig, model_name: str) -> str:
    for task_model in config.task_models:
        if task_model.name == model_name:
            return task_model.model_id
    available = ", ".join(m.name for m in config.task_models)
    raise ValueError(f"Model {model_name!r} not in config; available: {available}")


def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, total {len(text)} chars]"


def _format_tool_call(index: int, step_index: int, tool_call: dict) -> str:
    lines = [f"  [step {step_index} / tool call #{index}] {tool_call['tool_name']}"]
    try:
        args = json.loads(tool_call["tool_input"])
    except json.JSONDecodeError:
        args = {"_raw": tool_call["tool_input"]}
    code = args.get("code")
    if code is not None:
        lines.append("    --- code ---")
        for code_line in _truncate(str(code)).splitlines():
            lines.append(f"    | {code_line}")
    else:
        lines.append(f"    args: {json.dumps(args, ensure_ascii=False)[:500]}")

    try:
        output = json.loads(tool_call["tool_output"])
    except json.JSONDecodeError:
        output = {"_raw": tool_call["tool_output"]}
    status = output.get("status", "?")
    stdout = output.get("output") or output.get("stdout") or ""
    stderr = output.get("stderr") or output.get("error") or ""
    lines.append(f"    status: {status} | success={tool_call['success']}")
    if stdout:
        lines.append(f"    stdout: {_truncate(str(stdout), 300)!r}")
    if stderr:
        lines.append(f"    stderr: {_truncate(str(stderr), 300)!r}")
    if not stdout and not stderr and status == "success":
        lines.append("    stdout: (EMPTY — model did not print a result)")
    return "\n".join(lines)


def run_diagnostic(
    *,
    config_path: Path,
    model_name: str,
    num_episodes: int,
    seed_base: int,
    system_prompt: str | None,
) -> dict:
    config = ExperimentConfig.from_yaml(config_path)
    env_cfg = config.environment
    if not env_cfg.tools:
        raise ValueError(f"Config at {config_path} has no tools configured")

    model_id = _resolve_task_model(config, model_name)
    seed_prompt = system_prompt if system_prompt is not None else config.seed_prompt
    print(f"\nSystem prompt in use ({len(seed_prompt)} chars):\n{seed_prompt}\n")

    runner = GEMEpisodeRunner(
        environment_id=env_cfg.gem_env_id,
        model_id=model_id,
        temperature=config.eval_protocol.temperature_eval,
        max_steps=env_cfg.max_steps,
        max_response_tokens=config.eval_protocol.max_response_tokens,
        seed=seed_base,
        experiment_name=config.name,
        tools=list(env_cfg.tools),
        tool_runtime=ToolRuntime(),
    )

    episodes: list[dict] = []
    stats = {
        "num_episodes": num_episodes,
        "successes": 0,
        "episodes_with_tool_invocation": 0,
        "tool_calls_total": 0,
        "tool_calls_with_empty_stdout": 0,
        "tool_calls_errored": 0,
    }

    for ep in range(num_episodes):
        print(f"\n{'=' * 80}\nEpisode {ep + 1}/{num_episodes} (seed={seed_base + ep})\n{'=' * 80}")
        result = runner.run_episode(seed_prompt, episode_index=ep, persist=False)
        trajectory = result.trajectory

        print(f"Problem:\n  {_truncate(trajectory.initial_observation, 400)}")

        episode_tool_calls: list[dict] = []
        for step in trajectory.steps:
            for idx, tc in enumerate(step.tool_calls):
                tc_dict = {
                    "tool_name": tc.tool_name,
                    "tool_input": tc.tool_input,
                    "tool_output": tc.tool_output,
                    "success": tc.success,
                }
                episode_tool_calls.append({"step_index": step.step_index, **tc_dict})
                print(_format_tool_call(idx + 1, step.step_index, tc_dict))

                stats["tool_calls_total"] += 1
                try:
                    out = json.loads(tc.tool_output)
                except json.JSONDecodeError:
                    out = {}
                if out.get("status") == "success":
                    stdout = str(out.get("output") or "")
                    if not stdout.strip():
                        stats["tool_calls_with_empty_stdout"] += 1
                else:
                    stats["tool_calls_errored"] += 1

            print(f"  [step {step.step_index}] action: {_truncate(step.action, 300)!r}")
            print(f"  [step {step.step_index}] reward: {step.reward}, terminated={step.terminated}")

        success = trajectory.episode_outcome == "success"
        if success:
            stats["successes"] += 1
        if episode_tool_calls:
            stats["episodes_with_tool_invocation"] += 1

        print(
            f"\nOutcome: {trajectory.episode_outcome} | "
            f"total_reward={trajectory.total_reward} | "
            f"env_steps={len(trajectory.steps)} | "
            f"tool_calls={len(episode_tool_calls)} | "
            f"tokens={trajectory.total_tokens}"
        )

        episodes.append(
            {
                "seed": trajectory.seed,
                "outcome": trajectory.episode_outcome,
                "total_reward": trajectory.total_reward,
                "num_env_steps": len(trajectory.steps),
                "num_tool_calls": len(episode_tool_calls),
                "total_tokens": trajectory.total_tokens,
                "total_cost_usd": trajectory.total_cost_usd,
                "problem": trajectory.initial_observation,
                "steps": [
                    {
                        "step_index": s.step_index,
                        "action": s.action,
                        "reward": s.reward,
                        "terminated": s.terminated,
                        "truncated": s.truncated,
                        "tool_calls": [
                            {
                                "tool_name": tc.tool_name,
                                "tool_input": tc.tool_input,
                                "tool_output": tc.tool_output,
                                "success": tc.success,
                            }
                            for tc in s.tool_calls
                        ],
                    }
                    for s in trajectory.steps
                ],
            }
        )

    return {
        "config_path": str(config_path),
        "model_id": model_id,
        "model_name": model_name,
        "environment_id": env_cfg.gem_env_id,
        "tools": list(env_cfg.tools),
        "system_prompt": seed_prompt,
        "generated_at": datetime.now(UTC).isoformat(),
        "stats": stats,
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--model-name", type=str, default=_DEFAULT_MODEL_NAME)
    parser.add_argument("--num-episodes", type=int, default=_DEFAULT_NUM_EPISODES)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=(
            "Override system prompt. When omitted, uses the seed_prompt from "
            "the experiment config YAML."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ProjectPaths().root / "results" / "_diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    result = run_diagnostic(
        config_path=args.config,
        model_name=args.model_name,
        num_episodes=args.num_episodes,
        seed_base=args.seed_base,
        system_prompt=args.system_prompt,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dump_path = args.output_dir / f"python_exec_{timestamp}.json"
    dump_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key, value in result["stats"].items():
        print(f"  {key}: {value}")
    print(f"\nFull dump: {dump_path}")


if __name__ == "__main__":
    main()
