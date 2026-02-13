"""Integration tests for GEM episode execution with trajectory logging."""

import json
from pathlib import Path

import pytest

from scripts.run_gem_episode import run_gem_episode
from trajectory_aware_gym.logging import TrajectoryLogger


class TestGEMEpisodeExecution:
    """Integration tests for running GEM episodes with logging."""

    def test_run_gem_episode_creates_log(self, tmp_path):
        """Test that run_gem_episode creates a valid log file."""
        episode_id, log_file = run_gem_episode(
            env_id="game:GuessTheNumber-v0-easy",
            log_dir=tmp_path,
            verbose=False,
        )

        # Verify log file exists
        assert log_file.exists()
        assert log_file.suffix == ".json"
        assert episode_id in log_file.name

        # Verify log file can be loaded and validated
        logger = TrajectoryLogger(log_dir=tmp_path)
        episode_log = logger.load_episode(log_file)

        # Verify episode metadata
        assert episode_log.episode_id == episode_id
        assert episode_log.env_id == "game:GuessTheNumber-v0-easy"
        assert episode_log.num_steps > 0
        assert len(episode_log.steps) == episode_log.num_steps

    def test_episode_log_contains_all_steps(self, tmp_path):
        """Test that episode log contains complete trajectory."""
        _, log_file = run_gem_episode(
            env_id="game:GuessTheNumber-v0-easy",
            log_dir=tmp_path,
            verbose=False,
        )

        logger = TrajectoryLogger(log_dir=tmp_path)
        episode_log = logger.load_episode(log_file)

        # Verify each step has required fields
        for i, step in enumerate(episode_log.steps):
            assert step.step == i
            assert isinstance(step.observation, str)
            assert isinstance(step.action, str)
            assert isinstance(step.reward, float)
            assert isinstance(step.terminated, bool)
            assert isinstance(step.truncated, bool)
            assert isinstance(step.info, dict)

    def test_episode_success_recorded(self, tmp_path):
        """Test that episode success is correctly recorded."""
        _, log_file = run_gem_episode(
            env_id="game:GuessTheNumber-v0-easy",
            log_dir=tmp_path,
            verbose=False,
        )

        logger = TrajectoryLogger(log_dir=tmp_path)
        episode_log = logger.load_episode(log_file)

        # GuessTheNumber should succeed with binary search agent
        assert episode_log.success is True
        assert episode_log.total_reward > 0

        # Verify final step has termination
        final_step = episode_log.steps[-1]
        assert final_step.terminated is True

    def test_log_file_is_valid_json(self, tmp_path):
        """Test that log file is valid JSON."""
        _, log_file = run_gem_episode(
            env_id="game:GuessTheNumber-v0-easy",
            log_dir=tmp_path,
            verbose=False,
        )

        # Verify file can be parsed as JSON
        with open(log_file) as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "episode_id" in data
        assert "env_id" in data
        assert "timestamp" in data
        assert "steps" in data
        assert "total_reward" in data
        assert "success" in data
        assert "num_steps" in data
