"""Tests for centralized configuration (core.py + settings.py)."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from trajectory_aware_gym.config.core import (
    AWSModel,
    GEMModel,
    GEPAModel,
    RetryModel,
    Settings,
    _coerce,
    _with_env_overrides,
)
from trajectory_aware_gym.config.settings import ProjectPaths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_settings() -> Settings:
    """Load Settings from the production YAML (resets singleton first)."""
    Settings.reset()
    return Settings()


# ===========================================================================
# Settings loading from YAML
# ===========================================================================


class TestSettingsLoad:
    """Settings loads all sections from the production YAML."""

    def test_loads_gem_section(self):
        s = _load_settings()
        assert s.gem.max_steps == 50
        assert s.gem.temperature_train == 1.0
        assert s.gem.temperature_eval == 0.0

    def test_loads_aws_section(self):
        s = _load_settings()
        assert s.aws.region == "us-east-1"
        assert s.aws.s3_bucket == "trajectory-aware-gym-results"

    def test_loads_ollama_section(self):
        s = _load_settings()
        assert s.ollama.api_base == "http://localhost:11434"
        assert s.ollama.task_model_1_7b == "ollama/qwen3-1.7b-base"

    def test_loads_sagemaker_section(self):
        s = _load_settings()
        assert s.sagemaker.region == "us-east-1"
        assert s.sagemaker.endpoint_1_7b == "qwen3-1-7b-base"
        assert s.sagemaker.endpoint_4b == "qwen3-4b-base"
        assert s.sagemaker.model_id_1_7b == "Qwen/Qwen3-1.7B-Base"
        assert s.sagemaker.model_id_4b == "Qwen/Qwen3-4B-Base"

    def test_loads_gepa_section(self):
        s = _load_settings()
        assert s.gepa.num_threads >= 1
        assert s.gepa.reflection_model

    def test_loads_experiment_section(self):
        s = _load_settings()
        assert s.experiment.name == "baseline_experiment"
        assert s.experiment.random_seed == 42
        assert s.experiment.num_replications == 3

    def test_loads_logging_section(self):
        s = _load_settings()
        assert s.logging.level == "INFO"
        assert s.logging.file == "logs/training.log"

    def test_loads_cost_tracking_section(self):
        s = _load_settings()
        assert s.cost_tracking.enabled is True
        assert s.cost_tracking.alert_threshold == 100.0

    def test_loads_fitness_section(self):
        s = _load_settings()
        assert s.fitness.gamma == 0.99
        assert s.fitness.lambda_ == 0.1
        assert s.fitness.loop_penalty_weight == 1.0
        assert s.fitness.step_efficiency_weight == 1.0
        assert s.fitness.call_efficiency_weight == 1.0
        assert s.fitness.max_steps == 50
        assert s.fitness.loop_window == 3
        assert s.fitness.call_budget_per_step == 8

    def test_loads_retry_section(self):
        s = _load_settings()
        assert s.retry.max_attempts == 4
        assert s.retry.initial_wait_seconds == 1.0
        assert s.retry.max_wait_seconds == 30.0
        assert s.retry.exponential_base == 2.0
        assert s.retry.jitter is True
        assert s.retry.litellm_num_retries == 0
        assert s.retry.boto3_retry_mode == "standard"
        assert s.retry.boto3_max_attempts == 3
        assert s.retry.sagemaker_read_timeout_seconds == 90
        assert s.retry.inference_semaphore_size >= 1

    def test_missing_yaml_raises_on_required_fields(self, tmp_path):
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")
        Settings.reset()
        with pytest.raises(ValidationError):
            Settings(yaml_path=empty_yaml)

    def test_singleton_caches_across_instances(self):
        s1 = _load_settings()
        s2 = Settings()
        assert s1.gem.max_steps == s2.gem.max_steps

    def test_reset_clears_cache(self):
        _load_settings()
        assert Settings._loaded is True
        Settings.reset()
        assert Settings._loaded is False


# ===========================================================================
# Env var overrides
# ===========================================================================


class TestEnvVarOverrides:
    """Env vars (from .env or environment) override YAML values."""

    def test_env_overrides_string_field(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        s = _load_settings()
        assert s.aws.region == "eu-west-1"

    def test_env_overrides_int_field(self, monkeypatch):
        monkeypatch.setenv("GEM_MAX_STEPS", "999")
        s = _load_settings()
        assert s.gem.max_steps == 999

    def test_env_overrides_float_field(self, monkeypatch):
        monkeypatch.setenv("GEM_TEMPERATURE_TRAIN", "0.5")
        s = _load_settings()
        assert s.gem.temperature_train == 0.5

    def test_env_overrides_bool_field(self, monkeypatch):
        monkeypatch.setenv("COST_TRACKING_ENABLED", "false")
        s = _load_settings()
        assert s.cost_tracking.enabled is False

    def test_env_provides_secret_not_in_yaml(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        s = _load_settings()
        assert s.aws.access_key_id == "AKIATEST"

    def test_env_overrides_retry_int_field(self, monkeypatch):
        monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "6")
        s = _load_settings()
        assert s.retry.max_attempts == 6

    def test_env_overrides_retry_float_field(self, monkeypatch):
        monkeypatch.setenv("RETRY_INITIAL_WAIT_SECONDS", "2.5")
        s = _load_settings()
        assert s.retry.initial_wait_seconds == 2.5

    def test_yaml_value_used_when_env_absent(self):
        assert os.getenv("GEM_MAX_STEPS") is None
        s = _load_settings()
        assert s.gem.max_steps == 50


# ===========================================================================
# _with_env_overrides helper
# ===========================================================================


class TestWithEnvOverrides:
    """Direct tests for the _with_env_overrides merge function."""

    def test_env_overrides_yaml_value(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
        result = _with_env_overrides(
            {
                "region": "us-east-1",
                "s3_bucket": "b",
                "s3_prefix": "p",
                "bedrock_claude_sonnet_4_5": "x",
                "bedrock_llama_1b": "x",
                "bedrock_llama_3b": "x",
                "bedrock_llama_8b": "x",
            },
            "AWS",
            AWSModel,
        )
        assert result["region"] == "ap-southeast-1"

    def test_env_adds_field_not_in_yaml(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA123")
        result = _with_env_overrides(
            {
                "region": "us-east-1",
                "s3_bucket": "b",
                "s3_prefix": "p",
                "bedrock_claude_sonnet_4_5": "x",
                "bedrock_llama_1b": "x",
                "bedrock_llama_3b": "x",
                "bedrock_llama_8b": "x",
            },
            "AWS",
            AWSModel,
        )
        assert result["access_key_id"] == "AKIA123"

    def test_no_env_preserves_yaml(self):
        yaml_data = {"max_steps": 100, "temperature_train": 0.8, "temperature_eval": 0.1}
        result = _with_env_overrides(yaml_data, "GEM", GEMModel)
        assert result == yaml_data


# ===========================================================================
# _coerce helper
# ===========================================================================


class TestCoerce:
    """Type coercion from string env var values."""

    @pytest.mark.parametrize(
        ("value", "annotation", "expected"),
        [
            ("42", int, 42),
            ("0", int, 0),
            ("-1", int, -1),
            ("3.14", float, 3.14),
            ("0.0", float, 0.0),
            ("true", bool, True),
            ("True", bool, True),
            ("1", bool, True),
            ("yes", bool, True),
            ("false", bool, False),
            ("False", bool, False),
            ("0", bool, False),
            ("no", bool, False),
            ("hello", str, "hello"),
            ("", str, ""),
        ],
    )
    def test_coercion(self, value, annotation, expected):
        assert _coerce(value, annotation) == expected

    def test_invalid_int_raises(self):
        with pytest.raises(ValueError, match="invalid literal"):
            _coerce("not_a_number", int)

    def test_invalid_float_raises(self):
        with pytest.raises(ValueError, match="could not convert"):
            _coerce("not_a_float", float)


# ===========================================================================
# AWS credential validation
# ===========================================================================


class TestValidateAWS:
    """validate_aws() checks credentials when Bedrock models configured."""

    def test_raises_when_bedrock_model_without_credentials(self):
        s = _load_settings()
        with pytest.raises(ValueError, match="AWS credentials required"):
            s.validate_aws()

    def test_passes_when_credentials_provided(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")  # pragma: allowlist secret
        s = _load_settings()
        s.validate_aws()

    def test_passes_when_non_bedrock_model(self, monkeypatch):
        monkeypatch.setenv("GEPA_REFLECTION_MODEL", "local/my-model")
        s = _load_settings()
        s.validate_aws()


# ===========================================================================
# AWSModel client config generation
# ===========================================================================


class TestAWSModelClientConfig:
    """Tests for get_bedrock_client_config and get_s3_client_config."""

    @pytest.mark.parametrize(
        ("key_id", "secret", "session", "expected_keys"),
        [
            ("", "", "", {"region_name"}),
            (
                "AKIA",
                "secret",
                "",
                {"region_name", "aws_access_key_id", "aws_secret_access_key"},
            ),  # pragma: allowlist secret
            (
                "AKIA",
                "secret",
                "tok",
                {"region_name", "aws_access_key_id", "aws_secret_access_key", "aws_session_token"},
            ),  # pragma: allowlist secret
        ],
    )
    def test_bedrock_client_config_keys(self, key_id, secret, session, expected_keys):
        model = AWSModel(
            region="us-east-1",
            access_key_id=key_id,
            secret_access_key=secret,
            session_token=session,
            bedrock_claude_sonnet_4_5="x",
            bedrock_llama_1b="x",
            bedrock_llama_3b="x",
            bedrock_llama_8b="x",
            s3_bucket="b",
            s3_prefix="p",
        )
        config = model.get_bedrock_client_config()
        assert set(config.keys()) == expected_keys

    def test_s3_config_delegates_to_bedrock_config(self):
        model = AWSModel(
            region="eu-west-1",
            access_key_id="AKIA",
            secret_access_key="sec",  # pragma: allowlist secret
            bedrock_claude_sonnet_4_5="x",
            bedrock_llama_1b="x",
            bedrock_llama_3b="x",
            bedrock_llama_8b="x",
            s3_bucket="b",
            s3_prefix="p",
        )
        assert model.get_s3_client_config() == model.get_bedrock_client_config()


# ===========================================================================
# GEPA budget literal validation
# ===========================================================================


class TestGEPAModel:
    """GEPAModel runtime knobs validation."""

    def test_valid_construction(self):
        model = GEPAModel(num_threads=4, reflection_model="test")
        assert model.num_threads == 4
        assert model.reflection_model == "test"

    def test_num_threads_must_be_positive(self):
        with pytest.raises(ValidationError, match="num_threads"):
            GEPAModel(num_threads=0, reflection_model="test")


# ===========================================================================
# ProjectPaths
# ===========================================================================


class TestProjectPaths:
    """Tests for project paths."""

    def test_paths_created(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        assert paths.root == tmp_path
        assert paths.logs.exists()
        assert paths.results.exists()
        assert paths.data.exists()
        assert paths.experiments.exists()

    @pytest.mark.parametrize(
        ("attr", "suffix"),
        [
            ("src", "src"),
            ("tests", "tests"),
            ("logs", "logs"),
            ("results", "results"),
            ("data", "data"),
            ("experiments", "experiments"),
        ],
    )
    def test_path_attributes(self, tmp_path, attr, suffix):
        paths = ProjectPaths(root=tmp_path)
        assert getattr(paths, attr) == tmp_path / suffix
