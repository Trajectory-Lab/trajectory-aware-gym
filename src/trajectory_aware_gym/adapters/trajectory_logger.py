"""Trajectory logging utilities for GEM environment episodes.

Public API
----------
- ``TrajectoryLogger``              -- collect and persist per-step trajectory data
- ``extract_llm_calls_from_tracker`` -- convert ``dspy.track_usage()`` output to our schema

DSPy integration example::

    from trajectory_aware_gym.adapters import TrajectoryLogger, extract_llm_calls_from_tracker

    logger = TrajectoryLogger(environment_id="math:Math12K", seed=42)
    logger.set_initial_state(observation, info)

    for step in range(max_steps):
        with dspy.track_usage() as tracker:
            prediction = module(observation=observation)

        llm_calls = extract_llm_calls_from_tracker(tracker)   # <-- convert DSPy usage

        observation, reward, terminated, truncated, info = env.step(action)

        logger.add_step(                                       # <-- record the step
            action=action, observation=observation, reward=reward,
            terminated=terminated, truncated=truncated, info=info,
            llm_calls=llm_calls,
        )
        if terminated or truncated:
            break

    trajectory_log = logger.build_log()

Schema version history:
    1.1.0 - llm_call -> llm_calls (list) to support multiple LM forwards per env step (DSPy).
    1.0.0 - Initial schema with tool call tracking, LLM metadata, and cost aggregation.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from trajectory_aware_gym.config import ProjectPaths

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.1.0"

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
        if self.total_tokens < expected:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must be at least "
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
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)

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
    episode_outcome: EpisodeOutcome | None = None

    @computed_field
    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @computed_field
    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for step in self.steps for call in step.llm_calls)

    @computed_field
    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd or 0.0 for step in self.steps for call in step.llm_calls)

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
    """Collects and persists validated trajectory logs for one episode.

    See module docstring for DSPy integration example.
    """

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
        llm_calls: list[LLMCallMetadata] | None = None,
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
            llm_calls=llm_calls or [],
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
            system_prompt=self.system_prompt,
            started_at=self.started_at,
            finished_at=datetime.now(UTC),
            initial_observation=self.initial_observation,
            initial_info=self.initial_info,
            steps=self.steps,
            total_reward=sum(step.reward for step in self.steps),
            episode_outcome=_derive_outcome(self.steps),
        )

    def save(self, project_paths: ProjectPaths | None = None) -> Path:
        """Persist trajectory log to SQLite under ``logs/trajectories.db``.

        Returns the path to the database file.  The ``run_id`` of the saved
        episode is available via :attr:`last_run_id` after this call.
        """
        from trajectory_aware_gym.storage import save_trajectory

        trajectory_log = self.build_log()
        paths = project_paths or ProjectPaths()

        db_path = paths.logs / "trajectories.db"
        save_trajectory(db_path, trajectory_log)
        self._last_run_id = trajectory_log.run_id
        return db_path

    @property
    def last_run_id(self) -> str | None:
        """The ``run_id`` of the most recently saved trajectory, or None."""
        return getattr(self, "_last_run_id", None)


# ---------------------------------------------------------------------------
# Trajectory loading utilities (F2)
# ---------------------------------------------------------------------------


def load_trajectory(path: Path | str, *, run_id: str | None = None) -> TrajectoryLog:
    """Load a trajectory from a JSON file or SQLite database.

    When *path* points to a ``.db`` file, *run_id* is required.
    When *path* points to a ``.json`` file, the JSON is deserialized directly.
    """
    p = Path(path)
    if p.suffix == ".db":
        if run_id is None:
            raise ValueError("run_id is required when loading from a .db file")
        from trajectory_aware_gym.storage import load_trajectory_by_id

        return load_trajectory_by_id(p, run_id)
    return TrajectoryLog.model_validate_json(p.read_text(encoding="utf-8"))


def load_all_trajectories(directory: Path | str) -> list[TrajectoryLog]:
    """Load all trajectories from SQLite (preferred) or JSON files.

    If ``trajectories.db`` exists in *directory*, loads from SQLite.
    Otherwise falls back to globbing ``trajectory_*.json`` files.
    """
    d = Path(directory)
    db_path = d / "trajectories.db"
    if db_path.exists():
        from trajectory_aware_gym.storage import (
            load_all_trajectories as load_all_from_db,
        )

        return load_all_from_db(db_path)

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


# ---------------------------------------------------------------------------
# DSPy usage tracking helpers (F3)
# ---------------------------------------------------------------------------


def extract_llm_calls_from_tracker(tracker: Any) -> list[LLMCallMetadata]:
    """Convert a ``dspy.track_usage()`` tracker into LLMCallMetadata entries.

    ``tracker.usage_data`` is ``dict[str, list[dict]]`` where each key is a
    model name and each dict contains ``prompt_tokens``, ``completion_tokens``,
    and ``total_tokens`` from LiteLLM.

    Usage::

        with dspy.track_usage() as tracker:
            prediction = module(observation=obs)
        llm_calls = extract_llm_calls_from_tracker(tracker)
        logger.add_step(..., llm_calls=llm_calls)
    """
    calls: list[LLMCallMetadata] = []
    for model_id, usages in getattr(tracker, "usage_data", {}).items():
        for usage in usages:
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            calls.append(
                LLMCallMetadata(
                    model_id=model_id,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=prompt + completion,
                    cost_usd=usage.get("cost", None),
                )
            )
    return calls
