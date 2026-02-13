"""Tests for trajectory logging module."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trajectory_aware_gym.logging import EpisodeLog, TrajectoryLogger, TrajectoryStep


class TestTrajectoryStep:
    """Tests for TrajectoryStep model."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("step", 0),
            ("observation", "test observation"),
            ("action", "test action"),
            ("reward", 1.0),
            ("terminated", True),
            ("truncated", False),
        ],
    )
    def test_valid_fields(self, field, value):
        """Test valid field values."""
        data = {
            "step": 0,
            "observation": "obs",
            "action": "act",
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        data[field] = value
        step = TrajectoryStep(**data)
        assert getattr(step, field) == value

    def test_info_defaults_to_empty_dict(self):
        """Test info field defaults to empty dict."""
        step = TrajectoryStep(
            step=0,
            observation="obs",
            action="act",
            reward=0.0,
            terminated=False,
            truncated=False,
        )
        assert step.info == {}

    def test_info_custom_dict(self):
        """Test info field accepts custom dict."""
        info = {"key": "value", "nested": {"data": 123}}
        step = TrajectoryStep(
            step=0,
            observation="obs",
            action="act",
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )
        assert step.info == info


class TestEpisodeLog:
    """Tests for EpisodeLog model."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("steps", []),
            ("total_reward", 0.0),
            ("success", False),
            ("num_steps", 0),
        ],
    )
    def test_default_values(self, field, expected):
        """Test default field values."""
        episode = EpisodeLog(
            episode_id="test_123",
            env_id="game:GuessTheNumber-v0-easy",
            timestamp="2026-01-30T12:00:00Z",
        )
        assert getattr(episode, field) == expected

    def test_steps_list(self):
        """Test steps field accepts list of TrajectoryStep."""
        steps = [
            TrajectoryStep(
                step=0,
                observation="obs1",
                action="act1",
                reward=0.0,
                terminated=False,
                truncated=False,
            ),
            TrajectoryStep(
                step=1,
                observation="obs2",
                action="act2",
                reward=1.0,
                terminated=True,
                truncated=False,
            ),
        ]
        episode = EpisodeLog(
            episode_id="test_123",
            env_id="game:GuessTheNumber-v0-easy",
            timestamp="2026-01-30T12:00:00Z",
            steps=steps,
        )
        assert episode.steps == steps
        assert len(episode.steps) == 2


class TestTrajectoryLogger:
    """Tests for TrajectoryLogger."""

    def test_init_creates_log_dir(self, tmp_path):
        """Test logger creates log directory."""
        log_dir = tmp_path / "logs"
        logger = TrajectoryLogger(log_dir=log_dir)
        assert log_dir.exists()
        assert logger.log_dir == log_dir

    def test_start_episode(self, tmp_path):
        """Test starting an episode."""
        logger = TrajectoryLogger(log_dir=tmp_path)
        env_id = "game:GuessTheNumber-v0-easy"
        episode_id = logger.start_episode(env_id)

        assert episode_id.startswith(env_id)
        assert logger.current_episode is not None
        assert logger.current_episode.env_id == env_id
        assert logger.current_episode.episode_id == episode_id

    def test_log_step_without_start_raises_error(self, tmp_path):
        """Test logging step without starting episode raises error."""
        logger = TrajectoryLogger(log_dir=tmp_path)

        with pytest.raises(ValueError, match="No episode started"):
            logger.log_step(
                step=0,
                observation="obs",
                action="act",
                reward=0.0,
                terminated=False,
                truncated=False,
            )

    def test_log_step_updates_episode(self, tmp_path):
        """Test logging step updates episode data."""
        logger = TrajectoryLogger(log_dir=tmp_path)
        logger.start_episode("game:GuessTheNumber-v0-easy")

        logger.log_step(
            step=0,
            observation="obs",
            action="act",
            reward=0.5,
            terminated=False,
            truncated=False,
        )

        assert len(logger.current_episode.steps) == 1
        assert logger.current_episode.total_reward == 0.5
        assert logger.current_episode.num_steps == 1

    def test_log_step_accumulates_reward(self, tmp_path):
        """Test logging multiple steps accumulates reward."""
        logger = TrajectoryLogger(log_dir=tmp_path)
        logger.start_episode("game:GuessTheNumber-v0-easy")

        logger.log_step(0, "obs1", "act1", 0.3, False, False)
        logger.log_step(1, "obs2", "act2", 0.7, False, False)
        logger.log_step(2, "obs3", "act3", 1.0, True, False)

        assert logger.current_episode.total_reward == 2.0
        assert logger.current_episode.num_steps == 3

    def test_log_step_marks_success_on_positive_reward(self, tmp_path):
        """Test logging step marks success when terminated with positive reward."""
        logger = TrajectoryLogger(log_dir=tmp_path)
        logger.start_episode("game:GuessTheNumber-v0-easy")

        logger.log_step(0, "obs", "act", 1.0, True, False)

        assert logger.current_episode.success is True

    def test_log_step_no_success_on_zero_reward(self, tmp_path):
        """Test logging step doesn't mark success with zero reward."""
        logger = TrajectoryLogger(log_dir=tmp_path)
        logger.start_episode("game:GuessTheNumber-v0-easy")

        logger.log_step(0, "obs", "act", 0.0, True, False)

        assert logger.current_episode.success is False

    def test_save_episode_without_start_raises_error(self, tmp_path):
        """Test saving episode without starting raises error."""
        logger = TrajectoryLogger(log_dir=tmp_path)

        with pytest.raises(ValueError, match="No episode to save"):
            logger.save_episode()

    def test_save_episode_creates_json_file(self, tmp_path):
        """Test saving episode creates JSON file."""
        logger = TrajectoryLogger(log_dir=tmp_path)
        episode_id = logger.start_episode("game:GuessTheNumber-v0-easy")
        logger.log_step(0, "obs", "act", 1.0, True, False)

        log_file = logger.save_episode()

        assert log_file.exists()
        assert log_file.suffix == ".json"
        assert log_file.name == f"{episode_id}.json"

    def test_save_episode_clears_current_episode(self, tmp_path):
        """Test saving episode clears current episode."""
        logger = TrajectoryLogger(log_dir=tmp_path)
        logger.start_episode("game:GuessTheNumber-v0-easy")
        logger.log_step(0, "obs", "act", 1.0, True, False)

        logger.save_episode()

        assert logger.current_episode is None

    def test_save_and_load_episode_roundtrip(self, tmp_path):
        """Test saving and loading episode preserves data."""
        logger = TrajectoryLogger(log_dir=tmp_path)
        env_id = "game:GuessTheNumber-v0-easy"
        episode_id = logger.start_episode(env_id)

        logger.log_step(0, "obs1", "act1", 0.0, False, False, {"key": "value"})
        logger.log_step(1, "obs2", "act2", 1.0, True, False)

        log_file = logger.save_episode()
        loaded_episode = logger.load_episode(log_file)

        assert loaded_episode.episode_id == episode_id
        assert loaded_episode.env_id == env_id
        assert len(loaded_episode.steps) == 2
        assert loaded_episode.total_reward == 1.0
        assert loaded_episode.success is True
        assert loaded_episode.num_steps == 2
        assert loaded_episode.steps[0].info == {"key": "value"}

    def test_load_episode_validates_schema(self, tmp_path):
        """Test loading episode validates schema."""
        log_file = tmp_path / "invalid.json"
        log_file.write_text('{"invalid": "schema"}')

        logger = TrajectoryLogger(log_dir=tmp_path)

        with pytest.raises(ValidationError):
            logger.load_episode(log_file)

    @pytest.mark.parametrize(
        "missing_field",
        ["episode_id", "env_id", "timestamp"],
    )
    def test_episode_log_requires_fields(self, missing_field):
        """Test EpisodeLog requires essential fields."""
        data = {
            "episode_id": "test_123",
            "env_id": "game:GuessTheNumber-v0-easy",
            "timestamp": "2026-01-30T12:00:00Z",
        }
        del data[missing_field]

        with pytest.raises(ValidationError):
            EpisodeLog(**data)
