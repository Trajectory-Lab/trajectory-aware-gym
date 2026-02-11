"""Tests for package structure and imports."""

import pytest


class TestPackageImports:
    """Tests for verifying package imports work correctly."""

    def test_import_main_package(self):
        """Test main package can be imported."""
        import trajectory_aware_gym

        assert trajectory_aware_gym is not None

    def test_import_config_module(self):
        """Test config module can be imported."""
        from trajectory_aware_gym import config

        assert config is not None

    def test_import_adapters_module(self):
        """Test adapters module can be imported."""
        from trajectory_aware_gym import adapters

        assert adapters is not None

    def test_import_fitness_module(self):
        """Test fitness module can be imported."""
        from trajectory_aware_gym import fitness

        assert fitness is not None

    def test_import_optimizers_module(self):
        """Test optimizers module can be imported."""
        from trajectory_aware_gym import optimizers

        assert optimizers is not None

    def test_import_utils_module(self):
        """Test utils module can be imported."""
        from trajectory_aware_gym import utils

        assert utils is not None
