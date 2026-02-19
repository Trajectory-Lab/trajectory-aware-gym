"""Tests for trajectory logging and schema validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from trajectory_aware_gym.adapters.trajectory_logger import (
    TrajectoryLog,
    TrajectoryLogger,
    TrajectoryStep,
)
from trajectory_aware_gym.config.settings import ProjectPaths


class TestTrajectoryStep:
    """Tests for trajectory step schema."""

    @pytest.mark.parametrize(
        ("action", "observation", "reward", "terminated", "truncated"),
        [
            ("\\\\boxed{5}", "hint", 0.0, False, False),
            ("\\\\boxed{1}", "done", 1.0, True, False),
            ("action", "truncated", -0.5, False, True),
        ],
    )
    def test_valid_step(self, action, observation, reward, terminated, truncated):
        """Test valid step records."""
        step = TrajectoryStep(
            step_index=1,
            action=action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"suffix": "next"},
        )
        assert step.step_index == 1
        assert step.action == action

    @pytest.mark.parametrize(
        ("action", "observation"),
        [
            ("", "valid"),
            ("   ", "valid"),
            ("valid", ""),
            ("valid", "   "),
        ],
    )
    def test_blank_text_rejected(self, action, observation):
        """Test blank action/observation raises validation error."""
        with pytest.raises(ValidationError):
            TrajectoryStep(
                step_index=1,
                action=action,
                observation=observation,
                reward=0.0,
                terminated=False,
                truncated=False,
            )


class TestTrajectoryLog:
    """Tests for aggregate trajectory log schema."""

    def test_total_reward_must_match_steps(self):
        """Test total reward consistency validation."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            TrajectoryLog(
                environment_id="game:GuessTheNumber-v0-easy",
                seed=1,
                started_at=now,
                finished_at=now + timedelta(seconds=1),
                initial_observation="start",
                steps=[
                    TrajectoryStep(
                        step_index=1,
                        action="\\\\boxed{5}",
                        observation="higher",
                        reward=0.5,
                        terminated=False,
                        truncated=False,
                    )
                ],
                total_reward=0.4,
            )

    def test_finished_at_cannot_precede_started_at(self):
        """Test timestamp ordering validation."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            TrajectoryLog(
                environment_id="game:GuessTheNumber-v0-easy",
                seed=1,
                started_at=now,
                finished_at=now - timedelta(seconds=1),
                initial_observation="start",
                steps=[],
                total_reward=0.0,
            )


class TestTrajectoryLogger:
    """Tests for logger collection and persistence."""

    def test_save_writes_json_log(self, tmp_path):
        """Test save persists a valid JSON trajectory in logs directory."""
        paths = ProjectPaths(root=tmp_path)
        logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy", seed=42)
        logger.set_initial_state("start", {"suffix": "next"})
        logger.add_step(
            action="\\\\boxed{5}",
            observation="lower",
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"suffix": "next"},
        )
        logger.add_step(
            action="\\\\boxed{1}",
            observation="win",
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"suffix": "next"},
        )

        file_path = logger.save(project_paths=paths)
        assert file_path.exists()
        assert file_path.parent == paths.logs

        payload = json.loads(file_path.read_text(encoding="utf-8"))
        assert payload["environment_id"] == "game:GuessTheNumber-v0-easy"
        assert payload["total_reward"] == 1.0
        assert len(payload["steps"]) == 2

    def test_build_log_requires_initial_state(self):
        """Test logger raises if reset state was never captured."""
        logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy")
        with pytest.raises(ValueError, match="initial state is not set"):
            logger.build_log()
