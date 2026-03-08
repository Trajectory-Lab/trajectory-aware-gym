"""Run a single GEM episode and persist a validated trajectory log.

Demonstrates the extended trajectory logging schema (F1/F2) with:
- Optional system prompt tracking
- Episode outcome derivation
- Per-step timestamps
"""

from __future__ import annotations

import argparse
import importlib
import json

from trajectory_aware_gym.adapters import TrajectoryLogger
from trajectory_aware_gym.adapters.tool_runtime import ToolRuntime
from trajectory_aware_gym.config import settings
from trajectory_aware_gym.utils.tool_setup import build_tool_registry


def choose_guess(observation: str, low: int, high: int) -> tuple[int, int, int]:
    """Choose next guess based on environment hint text."""
    normalized = observation.lower()
    if "higher than" in normalized:
        marker = "higher than"
        pivot = int(normalized.split(marker, maxsplit=1)[1].split(".", maxsplit=1)[0].strip())
        low = max(low, pivot + 1)
    elif "lower than" in normalized:
        marker = "lower than"
        pivot = int(normalized.split(marker, maxsplit=1)[1].split(".", maxsplit=1)[0].strip())
        high = min(high, pivot - 1)

    guess = (low + high) // 2
    return guess, low, high


def run_episode(
    environment_id: str,
    seed: int | None = None,
    system_prompt: str | None = None,
) -> tuple[float, int, str]:
    """Execute one episode and save a trajectory log to disk."""
    gem = importlib.import_module("gem")
    importlib.import_module("gem.envs")
    env = gem.make(environment_id)

    reset_kwargs = {"seed": seed} if seed is not None else {}
    observation, info = env.reset(**reset_kwargs)

    logger = TrajectoryLogger(environment_id=environment_id, seed=seed)

    tool_registry = build_tool_registry()
    tool_runtime = ToolRuntime(tool_registry)

    if system_prompt:
        logger.set_system_prompt(system_prompt)
    logger.set_initial_state(observation=observation, info=info)

    total_reward = 0.0
    low, high = 1, 10

    for _ in range(settings.gem.max_steps):
        # --- Compute guess using python tool ---

        tool_call = {
            "tool": "python_exec",
            "arguments": {
                "code": f"""
        low = {low}
        high = {high}
        print((low + high) // 2)
        """
            },
        }

        tool_result = tool_runtime.execute(tool_call)

        if tool_result["status"] == "success":
            guess = int(tool_result["output"].strip())
        else:
            # fallback to local logic if tool fails
            guess, low, high = choose_guess(observation, low, high)
        action = f"\\\\boxed{{{guess}}}"

        observation, reward, terminated, truncated, step_info = env.step(action)
        logger.add_step(
            action=action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=step_info,
        )

        total_reward += reward
        if terminated or truncated:
            break

    log_path = logger.save()

    if hasattr(env, "close"):
        env.close()

    return total_reward, len(logger.steps), str(log_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run one GEM episode and log trajectory.")
    parser.add_argument(
        "--environment",
        default="game:GuessTheNumber-v0-easy",
        help="GEM environment id",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed for environment reset")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt to log")
    parser.add_argument("--show-log", action="store_true", help="Print the full JSON log")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    total_reward, steps, log_path = run_episode(
        args.environment,
        args.seed,
        args.system_prompt,
    )
    print(f"Environment:    {args.environment}")
    print(f"Steps:          {steps}")
    print(f"Total reward:   {total_reward:.3f}")
    print(f"Trajectory log: {log_path}")

    if args.show_log:
        from pathlib import Path

        payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
        print(f"\nSchema version: {payload['schema_version']}")
        print(f"Outcome:        {payload['episode_outcome']}")
        print(f"System prompt:  {payload.get('system_prompt', '(none)')}")


if __name__ == "__main__":
    main()
