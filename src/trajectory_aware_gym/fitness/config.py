"""Fitness function configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FitnessConfig(BaseSettings):
    """Hyperparameters for trajectory-aware fitness computation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    fitness_gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Discount factor for reverse-time weighting",
    )
    fitness_lambda: float = Field(
        default=0.1,
        ge=0.0,
        description="Scaling factor for auxiliary per-turn rewards",
    )
    fitness_loop_penalty_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for loop detection penalty term",
    )
    fitness_step_efficiency_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for step efficiency bonus term",
    )
    fitness_max_steps: int = Field(
        default=50,
        ge=1,
        description="Maximum expected steps for efficiency normalization",
    )
    fitness_loop_window: int = Field(
        default=3,
        ge=1,
        description="Sliding window size for consecutive loop detection",
    )
