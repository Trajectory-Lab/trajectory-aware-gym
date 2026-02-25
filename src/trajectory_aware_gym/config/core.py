"""Centralized configuration loaded from .env and YAML.

Priority (highest to lowest):
    1. Environment variables / .env file
    2. trajectory-aware-gym.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, get_origin

import dotenv
import yaml
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_YAML_PATH = _PROJECT_ROOT / "trajectory-aware-gym.yaml"
_DOTENV_PATH = _PROJECT_ROOT / ".env"

dotenv.load_dotenv(_DOTENV_PATH)


# ── Sub-models (pure schema — all values from .env or YAML) ─────


class AWSModel(BaseModel):
    """AWS and Bedrock configuration."""

    region: str
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: str = ""
    bedrock_claude_sonnet_4_5: str
    bedrock_llama_1b: str
    bedrock_llama_3b: str
    bedrock_llama_8b: str
    s3_bucket: str
    s3_prefix: str

    def get_bedrock_client_config(self) -> dict[str, str]:
        """Build boto3 Bedrock client kwargs."""
        config: dict[str, str] = {"region_name": self.region}
        if self.access_key_id:
            config["aws_access_key_id"] = self.access_key_id
        if self.secret_access_key:
            config["aws_secret_access_key"] = self.secret_access_key
        if self.session_token:
            config["aws_session_token"] = self.session_token
        return config

    def get_s3_client_config(self) -> dict[str, str]:
        """Build boto3 S3 client kwargs."""
        return self.get_bedrock_client_config()


class OllamaModel(BaseModel):
    """Local Ollama task model configuration."""

    api_base: str
    task_model_1_7b: str
    task_model_4b: str


class GEMModel(BaseModel):
    """GEM environment configuration."""

    max_steps: int
    temperature_train: float
    temperature_eval: float


class GEPAModel(BaseModel):
    """GEPA optimizer configuration."""

    budget: Literal["light", "medium", "heavy"]
    population_size: int
    iterations: int
    reflection_model: str


class ExperimentModel(BaseModel):
    """Experiment configuration."""

    name: str
    random_seed: int
    num_replications: int


class LoggingModel(BaseModel):
    """Logging configuration."""

    level: str
    file: str


class CostTrackingModel(BaseModel):
    """Cost tracking configuration."""

    enabled: bool
    alert_threshold: float


class FitnessModel(BaseModel):
    """Hyperparameters for trajectory-aware fitness computation."""

    model_config = {"populate_by_name": True}

    gamma: float = Field(ge=0.0, le=1.0)
    lambda_: float = Field(alias="lambda", ge=0.0)
    loop_penalty_weight: float = Field(ge=0.0)
    step_efficiency_weight: float = Field(ge=0.0)
    max_steps: int = Field(ge=1)
    loop_window: int = Field(ge=1)


# ── Settings ─────────────────────────────────────────────────────

_SECTION_MAP: list[tuple[str, str, type[BaseModel]]] = [
    ("aws", "AWS", AWSModel),
    ("ollama", "OLLAMA", OllamaModel),
    ("gem", "GEM", GEMModel),
    ("gepa", "GEPA", GEPAModel),
    ("experiment", "EXPERIMENT", ExperimentModel),
    ("logging", "LOG", LoggingModel),
    ("cost_tracking", "COST_TRACKING", CostTrackingModel),
    ("fitness", "FITNESS", FitnessModel),
]


class Settings:
    """Centralized configuration.

    Loads from .env / env vars (priority 1) with YAML fallback.
    No defaults are hardcoded — all values come from external sources.
    """

    _config_cache: dict[str, Any] = {}
    _loaded: bool = False

    _aws: AWSModel
    _ollama: OllamaModel
    _gem: GEMModel
    _gepa: GEPAModel
    _experiment: ExperimentModel
    _logging: LoggingModel
    _cost_tracking: CostTrackingModel
    _fitness: FitnessModel

    def __init__(self, yaml_path: Path | None = None) -> None:
        if not Settings._loaded:
            self._load(yaml_path or _YAML_PATH)

    @classmethod
    def _load(cls, yaml_path: Path) -> None:
        yaml_data: dict[str, Any] = {}
        if yaml_path.exists():
            yaml_data = yaml.safe_load(yaml_path.read_text()) or {}
        cls._config_cache = yaml_data

        for yaml_key, env_prefix, model_cls in _SECTION_MAP:
            merged = _with_env_overrides(yaml_data.get(yaml_key, {}), env_prefix, model_cls)
            instance = model_cls(**merged)
            setattr(cls, f"_{yaml_key}", instance)

        cls._loaded = True

    def validate_aws(self) -> None:
        """Raise if Bedrock models are configured but AWS credentials are missing.

        Call this explicitly before making Bedrock API calls, not at load time,
        since not all code paths require AWS access.
        """
        uses_bedrock = self._gepa.reflection_model.startswith(
            "anthropic."
        ) or self._gepa.reflection_model.startswith("us.meta.")
        if uses_bedrock and not self._aws.access_key_id:
            raise ValueError(
                "AWS credentials required when Bedrock models are configured. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env or environment."
            )

    @classmethod
    def reset(cls) -> None:
        """Clear cached config (for testing)."""
        cls._loaded = False
        cls._config_cache = {}

    @property
    def aws(self) -> AWSModel:
        return self._aws

    @property
    def ollama(self) -> OllamaModel:
        return self._ollama

    @property
    def gem(self) -> GEMModel:
        return self._gem

    @property
    def gepa(self) -> GEPAModel:
        return self._gepa

    @property
    def experiment(self) -> ExperimentModel:
        return self._experiment

    @property
    def logging(self) -> LoggingModel:
        return self._logging

    @property
    def cost_tracking(self) -> CostTrackingModel:
        return self._cost_tracking

    @property
    def fitness(self) -> FitnessModel:
        return self._fitness


# ── Helpers ──────────────────────────────────────────────────────


def _with_env_overrides(
    yaml_section: dict[str, Any],
    prefix: str,
    model_cls: type[BaseModel],
) -> dict[str, Any]:
    """Merge YAML values with env var overrides for all model fields.

    Checks ``PREFIX_FIELD`` env vars (e.g. ``AWS_REGION``) for every
    field defined on *model_cls*, not just those present in the YAML.
    Env vars take priority over YAML values.
    """
    result = dict(yaml_section)
    for field_name, field_info in model_cls.model_fields.items():
        # Use alias for env var lookup if set (e.g. lambda_ -> FITNESS_LAMBDA)
        lookup_name = field_info.alias or field_name
        env_key = f"{prefix}_{lookup_name}".upper()
        env_val = os.getenv(env_key)
        if env_val is not None:
            # Use alias as dict key when alias exists (matches YAML key)
            dict_key = field_info.alias or field_name
            result[dict_key] = _coerce(env_val, field_info.annotation)
    return result


def _coerce(value: str, annotation: Any) -> Any:
    """Coerce a string env var to the target type."""
    if annotation is bool:
        return value.lower() in ("true", "1", "yes")
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if get_origin(annotation) is Literal:
        return value
    return value
