"""Trajectory logger for GEM episodes with schema validation."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TrajectoryStep(BaseModel):
    """Single step in a GEM episode trajectory."""

    step: int = Field(description="Step number in episode (0-indexed)")
    observation: str = Field(description="Observation from environment")
    action: str = Field(description="Action taken by agent")
    reward: float = Field(description="Reward received")
    terminated: bool = Field(description="Whether episode terminated")
    truncated: bool = Field(description="Whether episode was truncated")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional info from env")


class EpisodeLog(BaseModel):
    """Complete episode trajectory log with metadata."""

    episode_id: str = Field(description="Unique episode identifier")
    env_id: str = Field(description="GEM environment ID")
    timestamp: str = Field(description="ISO 8601 timestamp")
    steps: list[TrajectoryStep] = Field(default_factory=list, description="Episode steps")
    total_reward: float = Field(default=0.0, description="Cumulative reward")
    success: bool = Field(default=False, description="Whether episode succeeded")
    num_steps: int = Field(default=0, description="Number of steps taken")


class TrajectoryLogger:
    """Logger for GEM episode trajectories with schema validation."""

    def __init__(self, log_dir: Path | str = "logs"):
        """Initialize trajectory logger.

        Args:
            log_dir: Directory to save trajectory logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_episode: EpisodeLog | None = None

    def start_episode(self, env_id: str) -> str:
        """Start a new episode.

        Args:
            env_id: GEM environment ID

        Returns:
            Episode ID
        """
        timestamp = datetime.now(UTC)
        episode_id = f"{env_id}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

        self.current_episode = EpisodeLog(
            episode_id=episode_id,
            env_id=env_id,
            timestamp=timestamp.isoformat(),
        )

        return episode_id

    def log_step(
        self,
        step: int,
        observation: str,
        action: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Log a single step in the episode.

        Args:
            step: Step number
            observation: Observation from environment
            action: Action taken by agent
            reward: Reward received
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Additional info from environment
        """
        if self.current_episode is None:
            raise ValueError("No episode started. Call start_episode() first.")

        trajectory_step = TrajectoryStep(
            step=step,
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info or {},
        )

        self.current_episode.steps.append(trajectory_step)
        self.current_episode.total_reward += reward
        self.current_episode.num_steps = len(self.current_episode.steps)

        # Check success based on final reward
        if terminated and reward > 0:
            self.current_episode.success = True

    def save_episode(self) -> Path:
        """Save the current episode to disk with schema validation.

        Returns:
            Path to saved log file

        Raises:
            ValueError: If no episode is active
        """
        if self.current_episode is None:
            raise ValueError("No episode to save. Call start_episode() first.")

        # Validate schema (Pydantic will raise ValidationError if invalid)
        episode_json = self.current_episode.model_dump(mode="json")

        # Save to file
        log_file = self.log_dir / f"{self.current_episode.episode_id}.json"
        with open(log_file, "w") as f:
            json.dump(episode_json, f, indent=2)

        # Clear current episode
        self.current_episode = None

        return log_file

    def load_episode(self, log_file: Path | str) -> EpisodeLog:
        """Load and validate an episode log from disk.

        Args:
            log_file: Path to log file

        Returns:
            Validated episode log
        """
        with open(log_file) as f:
            data = json.load(f)

        return EpisodeLog.model_validate(data)
