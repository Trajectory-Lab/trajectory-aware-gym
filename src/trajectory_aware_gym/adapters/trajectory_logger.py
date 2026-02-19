"""Trajectory logging utilities for GEM environment episodes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from trajectory_aware_gym.config import ProjectPaths


class TrajectoryStep(BaseModel):
    """Single transition in an environment trajectory."""

    step_index: int = Field(ge=1)
    action: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action", "observation")
    @classmethod
    def strip_and_validate_text(cls, value: str) -> str:
        """Normalize and validate required text fields."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class TrajectoryLog(BaseModel):
    """Validated schema for a single GEM episode trajectory."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    environment_id: str = Field(min_length=1)
    seed: int | None = None
    started_at: datetime
    finished_at: datetime
    initial_observation: str = Field(min_length=1)
    initial_info: dict[str, Any] = Field(default_factory=dict)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    total_reward: float

    @field_validator("environment_id", "initial_observation")
    @classmethod
    def strip_and_validate_text(cls, value: str) -> str:
        """Normalize and validate required text fields."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_consistency(self) -> TrajectoryLog:
        """Ensure aggregate values match per-step records."""
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be greater than or equal to started_at")

        computed_total = sum(step.reward for step in self.steps)
        if abs(self.total_reward - computed_total) > 1e-9:
            raise ValueError("total_reward must equal the sum of step rewards")

        return self


class TrajectoryLogger:
    """Collects and persists validated trajectory logs for one episode."""

    def __init__(self, environment_id: str, seed: int | None = None):
        self.environment_id = environment_id
        self.seed = seed
        self.started_at = datetime.now(UTC)
        self.initial_observation: str | None = None
        self.initial_info: dict[str, Any] = {}
        self.steps: list[TrajectoryStep] = []

    def set_initial_state(self, observation: str, info: dict[str, Any] | None = None) -> None:
        """Store initial state from env.reset()."""
        self.initial_observation = observation
        self.initial_info = info or {}

    def add_step(
        self,
        *,
        action: str,
        observation: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> TrajectoryStep:
        """Append a validated transition to the trajectory."""
        step = TrajectoryStep(
            step_index=len(self.steps) + 1,
            action=action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info or {},
        )
        self.steps.append(step)
        return step

    def build_log(self) -> TrajectoryLog:
        """Build and validate a complete trajectory log."""
        if self.initial_observation is None:
            raise ValueError("initial state is not set; call set_initial_state() first")

        return TrajectoryLog(
            environment_id=self.environment_id,
            seed=self.seed,
            started_at=self.started_at,
            finished_at=datetime.now(UTC),
            initial_observation=self.initial_observation,
            initial_info=self.initial_info,
            steps=self.steps,
            total_reward=sum(step.reward for step in self.steps),
        )

    def save(self, project_paths: ProjectPaths | None = None) -> Path:
        """Persist trajectory log as JSON under logs/ with timestamped filename."""
        trajectory_log = self.build_log()
        paths = project_paths or ProjectPaths()

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        file_path = paths.logs / f"trajectory_{timestamp}_{trajectory_log.run_id}.json"
        file_path.write_text(trajectory_log.model_dump_json(indent=2), encoding="utf-8")
        return file_path
