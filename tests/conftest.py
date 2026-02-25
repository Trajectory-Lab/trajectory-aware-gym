"""Shared test fixtures and configuration."""

from __future__ import annotations

import os

import pytest

from trajectory_aware_gym.config.core import Settings

_CONFIG_ENV_PREFIXES = (
    "AWS_",
    "OLLAMA_",
    "GEM_",
    "GEPA_",
    "EXPERIMENT_",
    "LOG_",
    "COST_TRACKING_",
)


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    """Reset Settings singleton and clean config env vars before each test.

    dotenv.load_dotenv() runs at core.py import time, polluting os.environ
    with .env values. This fixture clears them so tests load cleanly from
    the production YAML. Use monkeypatch.setenv() in individual tests to
    override specific values.
    """
    Settings.reset()
    for key in list(os.environ.keys()):
        if any(key.startswith(prefix) for prefix in _CONFIG_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    yield
    Settings.reset()
