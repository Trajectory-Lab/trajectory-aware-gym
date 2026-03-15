"""Baseline experiment configuration models and loaders."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class BaselineModelConfig(BaseModel):
    """Model selection for a baseline experiment run."""

    provider: str = Field(min_length=1)
    model_ref: str = Field(min_length=1)


class BaselineRuntimeConfig(BaseModel):
    """Runtime controls that should match evaluation protocol settings."""

    max_steps: int = Field(ge=1)
    max_response_tokens: int = Field(ge=1)
    temperature_train: float = Field(ge=0.0)
    temperature_eval: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)


class BaselineBudgetConfig(BaseModel):
    """Experiment budget and rollout counts for baseline runs."""

    train_episodes: int = Field(ge=1)
    eval_rollouts_per_task: int = Field(default=5, ge=1)
    num_replications: int = Field(default=3, ge=1)


class BaselineExperimentConfig(BaseModel):
    """Validated baseline experiment configuration."""

    config_name: str = Field(min_length=1)
    algorithm: Literal["grpo", "ppo"]
    environment: Literal["math12k", "codecontest", "hotpotqa"]
    environment_id: str = Field(min_length=1)
    output_subdir: str = Field(min_length=1)

    model: BaselineModelConfig
    runtime: BaselineRuntimeConfig
    budget: BaselineBudgetConfig


def load_baseline_config(config_path: Path) -> BaselineExperimentConfig:
    """Load and validate one baseline experiment TOML config file."""
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return BaselineExperimentConfig.model_validate(payload)


def load_baseline_configs(config_dir: Path) -> dict[Path, BaselineExperimentConfig]:
    """Load all baseline TOML configs in a directory."""
    config_paths = sorted(config_dir.glob("*.toml"))
    if not config_paths:
        raise FileNotFoundError(f"No baseline config files found in {config_dir}")

    return {config_path: load_baseline_config(config_path) for config_path in config_paths}
