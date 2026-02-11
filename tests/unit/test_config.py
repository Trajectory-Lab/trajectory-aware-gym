"""Tests for configuration modules."""

import os

import pytest

from trajectory_aware_gym.config.aws_config import AWSConfig
from trajectory_aware_gym.config.settings import (
    CostTrackingConfig,
    ExperimentConfig,
    GEMConfig,
    GEPAConfig,
    LoggingConfig,
    ProjectPaths,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clear environment variables that might interfere with config tests."""
    env_prefixes = ["AWS_", "BEDROCK_", "GEPA_", "GEM_", "S3_", "LITELLM_"]
    for key in list(os.environ.keys()):
        for prefix in env_prefixes:
            if key.startswith(prefix):
                monkeypatch.delenv(key, raising=False)


class TestAWSConfig:
    """Tests for AWS configuration."""

    def test_default_region(self):
        """Test default AWS region."""
        config = AWSConfig(_env_file=None)
        assert config.aws_region == "us-east-1"

    def test_get_bedrock_client_config(self):
        """Test Bedrock client config generation."""
        config = AWSConfig(_env_file=None)
        client_config = config.get_bedrock_client_config()
        assert "region_name" in client_config
        assert client_config["region_name"] == "us-east-1"

    def test_get_s3_client_config(self):
        """Test S3 client config generation."""
        config = AWSConfig(_env_file=None)
        client_config = config.get_s3_client_config()
        assert "region_name" in client_config


class TestExperimentConfig:
    """Tests for experiment configuration."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("experiment_name", "baseline_experiment"),
            ("random_seed", 42),
            ("num_replications", 3),
        ],
    )
    def test_default_values(self, field, expected):
        """Test default configuration values."""
        config = ExperimentConfig(_env_file=None)
        assert getattr(config, field) == expected


class TestGEPAConfig:
    """Tests for GEPA configuration."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("gepa_budget", "medium"),
            ("gepa_population_size", 6),
            ("gepa_iterations", 75),
        ],
    )
    def test_default_values(self, field, expected):
        """Test default configuration values."""
        config = GEPAConfig(_env_file=None)
        assert getattr(config, field) == expected

    @pytest.mark.parametrize("budget", ["light", "medium", "heavy"])
    def test_budget_literal_values(self, budget):
        """Test budget accepts valid literal values."""
        config = GEPAConfig(gepa_budget=budget, _env_file=None)
        assert config.gepa_budget == budget


class TestGEMConfig:
    """Tests for GEM configuration."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("gem_max_steps", 50),
            ("gem_temperature_train", 1.0),
            ("gem_temperature_eval", 0.0),
        ],
    )
    def test_default_values(self, field, expected):
        """Test default configuration values."""
        config = GEMConfig(_env_file=None)
        assert getattr(config, field) == expected


class TestLoggingConfig:
    """Tests for logging configuration."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("log_level", "INFO"),
            ("log_file", "logs/training.log"),
        ],
    )
    def test_default_values(self, field, expected):
        """Test default configuration values."""
        config = LoggingConfig(_env_file=None)
        assert getattr(config, field) == expected


class TestCostTrackingConfig:
    """Tests for cost tracking configuration."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("track_costs", True),
            ("cost_alert_threshold", 100.0),
        ],
    )
    def test_default_values(self, field, expected):
        """Test default configuration values."""
        config = CostTrackingConfig(_env_file=None)
        assert getattr(config, field) == expected


class TestProjectPaths:
    """Tests for project paths."""

    def test_paths_created(self, tmp_path):
        """Test that required directories are created."""
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
        """Test path attributes are set correctly."""
        paths = ProjectPaths(root=tmp_path)
        assert getattr(paths, attr) == tmp_path / suffix
