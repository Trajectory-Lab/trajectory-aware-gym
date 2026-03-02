"""Trajectory logging utilities for GEM environment episodes.

Schema version history:
    1.0.0 - Initial schema with tool call tracking, LLM metadata, and cost aggregation.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from trajectory_aware_gym.config import ProjectPaths

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

type EpisodeOutcome = Literal["success", "failure", "truncated"]


class ToolCall(BaseModel):
    """A single tool invocation within an episode step."""

    tool_name: str = Field(min_length=1)
    tool_input: str
    tool_output: str
    success: bool
    duration_ms: float | None = None

    @field_validator("tool_name")
    @classmethod
    def strip_and_validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_name must not be blank")
        return normalized


class LLMCallMetadata(BaseModel):
    """Token usage and cost metadata for a single LLM call."""

    model_id: str = Field(min_length=1)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float | None = None
    latency_ms: float | None = None

    @model_validator(mode="after")
    def validate_token_sum(self) -> LLMCallMetadata:
        expected = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal "
                f"prompt_tokens + completion_tokens ({expected})"
            )
        return self


class TrajectoryStep(BaseModel):
    """Single transition in an environment trajectory."""

    step_index: int = Field(ge=1)
    action: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Assumes at most one LLM call per env step. F3 (DSPy integration) may need
    # llm_calls: list[LLMCallMetadata] if DSPy issues multiple forward() per step.
    llm_call: LLMCallMetadata | None = None

    @field_validator("reward")
    @classmethod
    def reject_non_finite_reward(cls, value: float) -> float:
        import math

        if not math.isfinite(value):
            raise ValueError(f"reward must be finite, got {value}")
        return value

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

    schema_version: str = SCHEMA_VERSION
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    environment_id: str = Field(min_length=1)
    seed: int | None = None
    # The GEPA-level prompt. Per-step LLM prompts (if needed) should be stored separately.
    system_prompt: str | None = None
    started_at: datetime
    finished_at: datetime
    initial_observation: str = Field(min_length=1)
    initial_info: dict[str, Any] = Field(default_factory=dict)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    total_reward: float
    num_steps: int = 0
    episode_outcome: EpisodeOutcome | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0

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

        if self.num_steps != len(self.steps):
            raise ValueError(
                f"num_steps ({self.num_steps}) must equal len(steps) ({len(self.steps)})"
            )

        computed_tokens = sum(step.llm_call.total_tokens for step in self.steps if step.llm_call)
        if self.total_tokens != computed_tokens:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal "
                f"sum of step tokens ({computed_tokens})"
            )

        computed_cost = sum(
            step.llm_call.cost_usd
            for step in self.steps
            if step.llm_call and step.llm_call.cost_usd is not None
        )
        if abs(self.total_cost_usd - computed_cost) > 1e-9:
            raise ValueError(
                f"total_cost_usd ({self.total_cost_usd}) must equal "
                f"sum of step costs ({computed_cost})"
            )

        return self


# GEM envs provide ground-truth success in info (e.g., MathEnv: {"correct": bool}).
# We check these keys first; reward > 0 is only a fallback when no explicit signal exists.
_SUCCESS_INFO_KEYS = ("correct", "is_correct", "success", "task_success")


def _derive_outcome(steps: list[TrajectoryStep]) -> EpisodeOutcome | None:
    """Derive episode outcome from the final step's info dict and termination signals.

    Priority order:
        1. Explicit success signal in info dict (ground truth from GEM env)
        2. truncated flag (takes precedence; GEM game envs may set both flags on final turn)
        3. terminated flag with reward > 0 as fallback heuristic
    """
    if not steps:
        return None
    final = steps[-1]

    for key in _SUCCESS_INFO_KEYS:
        if key in final.info:
            return "success" if bool(final.info[key]) else "failure"

    # truncated takes precedence: GEM game envs may set both flags on the final turn.
    if final.truncated:
        return "truncated"

    if final.terminated:
        return "success" if final.reward > 0 else "failure"

    return None


class TrajectoryLogger:
    """Collects and persists validated trajectory logs for one episode."""

    def __init__(
        self,
        environment_id: str,
        seed: int | None = None,
    ):
        self.environment_id = environment_id
        self.seed = seed
        self.system_prompt: str | None = None
        self.started_at = datetime.now(UTC)
        self.initial_observation: str | None = None
        self.initial_info: dict[str, Any] = {}
        self.steps: list[TrajectoryStep] = []

    def set_initial_state(self, observation: str, info: dict[str, Any] | None = None) -> None:
        if self.initial_observation is not None:
            raise RuntimeError("Initial state has already been set; cannot overwrite.")
        self.initial_observation = observation
        self.initial_info = info or {}

    def set_system_prompt(self, prompt: str) -> None:
        """Store the system prompt used for this episode (the GEPA independent variable)."""
        self.system_prompt = prompt

    def add_step(
        self,
        *,
        action: str,
        observation: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
        tool_calls: list[ToolCall] | None = None,
        llm_call: LLMCallMetadata | None = None,
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
            timestamp=timestamp or datetime.now(UTC),
            tool_calls=tool_calls or [],
            llm_call=llm_call,
        )
        self.steps.append(step)
        return step

    def build_log(self) -> TrajectoryLog:
        """Build and validate a complete trajectory log."""
        if self.initial_observation is None:
            raise ValueError("initial state is not set; call set_initial_state() first")

        total_tokens = sum(s.llm_call.total_tokens for s in self.steps if s.llm_call)
        total_cost = sum(
            s.llm_call.cost_usd
            for s in self.steps
            if s.llm_call and s.llm_call.cost_usd is not None
        )

        return TrajectoryLog(
            environment_id=self.environment_id,
            seed=self.seed,
            system_prompt=self.system_prompt,
            started_at=self.started_at,
            finished_at=datetime.now(UTC),
            initial_observation=self.initial_observation,
            initial_info=self.initial_info,
            steps=self.steps,
            total_reward=sum(step.reward for step in self.steps),
            num_steps=len(self.steps),
            episode_outcome=_derive_outcome(self.steps),
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
        )

    def save(self, project_paths: ProjectPaths | None = None) -> Path:
        """Persist trajectory log as JSON under logs/ with timestamped filename.
        Creates the logs directory if it doesn't exist.
        Writes to a temporary file first and then atomically replaces the existing file.
        This ensures the file is always in a valid state and avoids partial writes.
        """
        trajectory_log = self.build_log()
        paths = project_paths or ProjectPaths()

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        file_path = paths.logs / f"trajectory_{timestamp}_{trajectory_log.run_id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = file_path.with_suffix(".json.tmp")
        tmp_path.write_text(trajectory_log.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp_path, file_path)
        return file_path


# ---------------------------------------------------------------------------
# Trajectory loading utilities (F2)
# ---------------------------------------------------------------------------


def load_trajectory(path: Path | str) -> TrajectoryLog:
    """Deserialize a trajectory log from a JSON file."""
    p = Path(path)
    return TrajectoryLog.model_validate_json(p.read_text(encoding="utf-8"))


def load_all_trajectories(directory: Path | str) -> list[TrajectoryLog]:
    """Load all trajectory JSON logs from a directory, sorted by started_at."""
    d = Path(directory)
    logs: list[TrajectoryLog] = []
    for p in d.glob("trajectory_*.json"):
        try:
            logs.append(TrajectoryLog.model_validate_json(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Skipping corrupt trajectory %s: %s", p, exc)
    return sorted(logs, key=lambda t: t.started_at)


def filter_trajectories(
    logs: list[TrajectoryLog],
    *,
    outcome: EpisodeOutcome | None = None,
    environment_id: str | None = None,
) -> list[TrajectoryLog]:
    """Filter trajectory logs by outcome or environment ID."""
    result = logs
    if outcome is not None:
        result = [t for t in result if t.episode_outcome == outcome]
    if environment_id is not None:
        result = [t for t in result if t.environment_id == environment_id]
    return result
