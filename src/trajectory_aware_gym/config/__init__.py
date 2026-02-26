"""Configuration management for trajectory-aware-gym project."""

from trajectory_aware_gym.config.core import (
    AWSModel,
    CostTrackingModel,
    ExperimentModel,
    FitnessModel,
    GEMModel,
    GEPAModel,
    LoggingModel,
    OllamaModel,
    Settings,
)
from trajectory_aware_gym.config.settings import ProjectPaths

settings = Settings()

__all__ = [
    "AWSModel",
    "CostTrackingModel",
    "ExperimentModel",
    "FitnessModel",
    "GEMModel",
    "GEPAModel",
    "LoggingModel",
    "OllamaModel",
    "ProjectPaths",
    "Settings",
    "settings",
]
