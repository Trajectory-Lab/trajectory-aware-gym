#!/usr/bin/env python3
"""Run a single GEM episode with trajectory logging.

This script executes one complete episode in a GEM environment using a simple
agent and logs the full trajectory to disk with schema validation.
"""

import re
import sys
from pathlib import Path

import gem
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trajectory_aware_gym.logging import TrajectoryLogger

console = Console()


class SimpleGuessAgent:
    """Simple binary search agent for GuessTheNumber environment."""

    def __init__(self) -> None:
        self.low: int | None = None
        self.high: int | None = None
        self.last_action: int | None = None

    def parse_range(self, observation: str) -> None:
        """Parse the number range from observation."""
        numbers = list(map(int, re.findall(r"\d+", observation)))
        if len(numbers) >= 2:
            self.low, self.high = numbers[0], numbers[1]

    def act(self, observation: str | None) -> str:
        """Choose action based on observation using binary search."""
        # First step: read range
        if observation is not None and (self.low is None or self.high is None):
            self.parse_range(observation)

        # Update bounds from feedback
        if observation is not None and self.last_action is not None:
            if "higher" in observation:
                self.low = self.last_action + 1
            elif "lower" in observation:
                self.high = self.last_action - 1

        # Safety check
        if self.low is None or self.high is None:
            raise ValueError("Range not initialized from observation.")

        # Binary search guess
        self.last_action = (self.low + self.high) // 2
        return f"\\boxed{{{self.last_action}}}"


def run_gem_episode(
    env_id: str = "game:GuessTheNumber-v0-easy",
    log_dir: Path | str = "logs",
    verbose: bool = True,
) -> tuple[str, Path]:
    """Run a single GEM episode with trajectory logging.

    Args:
        env_id: GEM environment ID
        log_dir: Directory to save trajectory logs
        verbose: Whether to print progress

    Returns:
        Tuple of (episode_id, log_file_path)
    """
    # Initialize environment and logger
    env = gem.make(env_id)
    logger = TrajectoryLogger(log_dir=log_dir)

    # Start episode
    episode_id = logger.start_episode(env_id)
    observation, info = env.reset()

    if verbose:
        console.print(
            Panel.fit(
                f"[bold cyan]Starting Episode[/bold cyan]\n"
                f"Episode ID: {episode_id}\n"
                f"Environment: {env_id}",
                border_style="cyan",
            )
        )
        console.print(f"\n[bold]Initial Observation:[/bold]\n{observation}\n")

    # Initialize agent
    agent = SimpleGuessAgent()

    # Run episode
    step = 0
    while True:
        # Agent takes action
        action = agent.act(observation)

        # Environment steps
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Log this step
        logger.log_step(
            step=step,
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        # Extract guess for display
        guess = action.replace("\\boxed{", "").replace("}", "")

        if verbose:
            console.print(
                f"[bold yellow]Step {step}[/bold yellow]\n"
                f"  Action: {guess}\n"
                f"  Reward: {reward}\n"
                f"  Response: {next_obs}\n"
            )

        observation = next_obs
        step += 1

        if terminated or truncated:
            break

    # Save trajectory to disk
    log_file = logger.save_episode()

    # Print summary
    if verbose:
        episode_log = logger.load_episode(log_file)

        table = Table(title="Episode Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Episode ID", episode_log.episode_id)
        table.add_row("Environment", episode_log.env_id)
        table.add_row("Total Steps", str(episode_log.num_steps))
        table.add_row("Total Reward", f"{episode_log.total_reward:.2f}")
        table.add_row("Success", "✓" if episode_log.success else "✗")
        table.add_row("Log File", str(log_file))

        console.print()
        console.print(table)
        console.print()
        console.print(f"[green]✓[/green] Episode completed and saved to: {log_file}")
        console.print(f"[green]✓[/green] Schema validated successfully")

    return episode_id, log_file


def main() -> int:
    """Main entry point."""
    try:
        episode_id, log_file = run_gem_episode()
        return 0
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", style="red")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
