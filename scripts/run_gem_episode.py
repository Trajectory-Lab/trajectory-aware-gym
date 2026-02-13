"""Run a single GEM episode and persist a validated trajectory log."""

from __future__ import annotations

import argparse
import importlib

from trajectory_aware_gym.adapters import TrajectoryLogger
from trajectory_aware_gym.config import GEMConfig


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


def run_episode(environment_id: str, seed: int | None = None) -> tuple[float, int, str]:
    """Execute one episode and save a trajectory log to disk."""
    config = GEMConfig(_env_file=None)
    gem = importlib.import_module("gem")
    importlib.import_module("gem.envs")
    env = gem.make(environment_id)

    reset_kwargs = {"seed": seed} if seed is not None else {}
    observation, info = env.reset(**reset_kwargs)

    logger = TrajectoryLogger(environment_id=environment_id, seed=seed)
    logger.set_initial_state(observation=observation, info=info)

    total_reward = 0.0
    low, high = 1, 10

    for _ in range(config.gem_max_steps):
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
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    total_reward, steps, log_path = run_episode(args.environment, args.seed)
    print(f"Environment: {args.environment}")
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Trajectory log: {log_path}")


if __name__ == "__main__":
    main()
