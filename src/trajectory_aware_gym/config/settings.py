"""General project settings and configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    """Experiment configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    experiment_name: str = Field(
        default="baseline_experiment",
        description="Name of the experiment",
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    num_replications: int = Field(
        default=3,
        description="Number of experiment replications",
    )


class GEPAConfig(BaseSettings):
    """GEPA optimizer configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    gepa_budget: Literal["light", "medium", "heavy"] = Field(
        default="medium",
        description="GEPA optimization budget",
    )
    gepa_population_size: int = Field(
        default=6,
        description="GEPA population size",
    )
    gepa_iterations: int = Field(
        default=75,
        description="Number of GEPA iterations",
    )
    gepa_reflection_model: str = Field(
        default="anthropic.claude-sonnet-4-5-v2:0",
        description="Model to use for GEPA reflection",
    )


class GEMConfig(BaseSettings):
    """GEM environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    gem_max_steps: int = Field(
        default=50,
        description="Maximum steps per episode",
    )
    gem_temperature_train: float = Field(
        default=1.0,
        description="Temperature for training",
    )
    gem_temperature_eval: float = Field(
        default=0.0,
        description="Temperature for evaluation",
    )


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(
        default="logs/training.log",
        description="Log file path",
    )


class CostTrackingConfig(BaseSettings):
    """Cost tracking configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    track_costs: bool = Field(default=True, description="Enable cost tracking")
    cost_alert_threshold: float = Field(
        default=100.0,
        description="Cost alert threshold in USD",
    )


class ProjectPaths:
    """Project directory paths."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).parent.parent.parent.parent
        self.src = self.root / "src"
        self.tests = self.root / "tests"
        self.logs = self.root / "logs"
        self.results = self.root / "results"
        self.data = self.root / "data"
        self.experiments = self.root / "experiments"

        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for path in [self.logs, self.results, self.data, self.experiments]:
            path.mkdir(parents=True, exist_ok=True)
