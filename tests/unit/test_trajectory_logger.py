"""Tests for trajectory logging and schema validation.

Deferred test ideas (P4+, not critical for current project phase):
- TODO(P4): Two loggers saving to same directory don't collide (UUID4 in filename should prevent)
- TODO(P4): 1000-step episode builds/serializes correctly (stress test)
- TODO(P4): Round-trip test with full model_dump() equality instead of field-by-field
- TODO(P4): episode_outcome="unknown" rejected by Literal type
- TODO(P4): Decouple _make_log helper from production aggregation logic
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from trajectory_aware_gym.adapters.trajectory_logger import (
    SCHEMA_VERSION,
    LLMCallMetadata,
    ToolCall,
    TrajectoryLog,
    TrajectoryLogger,
    TrajectoryStep,
    _derive_outcome,
    extract_llm_calls_from_tracker,
    filter_trajectories,
    load_all_trajectories,
    load_trajectory,
)
from trajectory_aware_gym.config import ProjectPaths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    step_index: int = 1,
    *,
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    info: dict[str, Any] | None = None,
    tool_calls: list[ToolCall] | None = None,
    llm_calls: list[LLMCallMetadata] | None = None,
    timestamp: datetime | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=step_index,
        action="act",
        observation="obs",
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info or {},
        tool_calls=tool_calls or [],
        llm_calls=llm_calls or [],
        timestamp=timestamp,
    )


def _make_llm_call(
    prompt: int = 10,
    completion: int = 20,
    cost: float | None = 0.001,
) -> LLMCallMetadata:
    return LLMCallMetadata(
        model_id="bedrock/qwen3-1.7b",
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cost_usd=cost,
    )


def _make_log(
    steps: list[TrajectoryStep] | None = None,
    **overrides,
) -> TrajectoryLog:
    now = datetime.now(UTC)
    steps = steps or []
    defaults = {
        "environment_id": "game:GuessTheNumber-v0-easy",
        "seed": 1,
        "started_at": now,
        "finished_at": now + timedelta(seconds=1),
        "initial_observation": "start",
        "steps": steps,
        "total_reward": sum(s.reward for s in steps),
    }
    defaults.update(overrides)
    return TrajectoryLog(**defaults)


# ===========================================================================
# F1: Schema Model Tests
# ===========================================================================


class TestToolCall:
    """Tests for the ToolCall schema."""

    @pytest.mark.parametrize(
        ("name", "inp", "out", "success", "dur"),
        [
            ("python_exec", "print(1)", "1", True, 50.0),
            ("web_search", "query", "result", True, None),
            ("shell", "ls", "file.txt", False, 120.5),
        ],
    )
    def test_valid_tool_call(self, name, inp, out, success, dur):
        tc = ToolCall(
            tool_name=name,
            tool_input=inp,
            tool_output=out,
            success=success,
            duration_ms=dur,
        )
        assert tc.tool_name == name
        assert tc.success is success

    @pytest.mark.parametrize("blank_name", ["", "   "])
    def test_blank_tool_name_rejected(self, blank_name):
        with pytest.raises(ValidationError):
            ToolCall(
                tool_name=blank_name,
                tool_input="x",
                tool_output="y",
                success=True,
            )

    def test_empty_input_output_allowed(self):
        tc = ToolCall(tool_name="noop", tool_input="", tool_output="", success=True)
        assert tc.tool_input == ""


class TestLLMCallMetadata:
    """Tests for the LLMCallMetadata schema."""

    @pytest.mark.parametrize(
        ("prompt", "completion", "cost", "latency"),
        [
            (10, 20, 0.001, 150.0),
            (0, 0, None, None),
            (100, 500, 0.05, 2000.0),
        ],
    )
    def test_valid_metadata(self, prompt, completion, cost, latency):
        meta = LLMCallMetadata(
            model_id="bedrock/qwen3-1.7b",
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cost_usd=cost,
            latency_ms=latency,
        )
        assert meta.total_tokens == prompt + completion

    def test_total_below_sum_rejected(self):
        with pytest.raises(ValidationError, match="total_tokens"):
            LLMCallMetadata(
                model_id="bedrock/qwen3-1.7b",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=25,
            )

    def test_thinking_tokens_accepted(self):
        """total_tokens may exceed prompt + completion (e.g. Qwen3 thinking tokens)."""
        meta = LLMCallMetadata(
            model_id="ollama/qwen3-4b-base",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=300,
        )
        assert meta.total_tokens == 300

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            LLMCallMetadata(
                model_id="bedrock/qwen3-1.7b",
                prompt_tokens=-1,
                completion_tokens=10,
                total_tokens=9,
            )

    def test_blank_model_id_rejected(self):
        with pytest.raises(ValidationError):
            LLMCallMetadata(
                model_id="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )


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
        with pytest.raises(ValidationError):
            TrajectoryStep(
                step_index=1,
                action=action,
                observation=observation,
                reward=0.0,
                terminated=False,
                truncated=False,
            )

    def test_step_with_tool_calls(self):
        tc = ToolCall(
            tool_name="python_exec",
            tool_input="print(1+1)",
            tool_output="2",
            success=True,
            duration_ms=42.0,
        )
        step = _make_step(tool_calls=[tc])
        assert len(step.tool_calls) == 1
        assert step.tool_calls[0].tool_name == "python_exec"

    def test_step_with_llm_calls(self):
        meta = _make_llm_call()
        step = _make_step(llm_calls=[meta])
        assert len(step.llm_calls) == 1
        assert step.llm_calls[0].total_tokens == 30

    def test_step_with_multiple_llm_calls(self):
        calls = [_make_llm_call(10, 20, 0.01), _make_llm_call(5, 15, 0.005)]
        step = _make_step(llm_calls=calls)
        assert len(step.llm_calls) == 2
        assert step.llm_calls[0].total_tokens == 30
        assert step.llm_calls[1].total_tokens == 20

    def test_step_defaults_no_tool_calls_or_llm(self):
        step = _make_step()
        assert step.tool_calls == []
        assert step.llm_calls == []
        assert step.timestamp is None

    def test_step_with_timestamp(self):
        ts = datetime(2026, 2, 22, 12, 0, 0, tzinfo=UTC)
        step = _make_step(timestamp=ts)
        assert step.timestamp == ts

    @pytest.mark.parametrize("bad_index", [0, -1, -100])
    def test_step_index_below_one_rejected(self, bad_index):
        with pytest.raises(ValidationError):
            TrajectoryStep(
                step_index=bad_index,
                action="a",
                observation="o",
                reward=0.0,
                terminated=False,
                truncated=False,
            )

    @pytest.mark.parametrize("bad_reward", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_reward_rejected(self, bad_reward):
        with pytest.raises(ValidationError, match="finite"):
            TrajectoryStep(
                step_index=1,
                action="a",
                observation="o",
                reward=bad_reward,
                terminated=False,
                truncated=False,
            )


class TestTrajectoryLog:
    """Tests for aggregate trajectory log schema."""

    def test_total_reward_must_match_steps(self):
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            TrajectoryLog(
                environment_id="game:GuessTheNumber-v0-easy",
                seed=1,
                started_at=now,
                finished_at=now + timedelta(seconds=1),
                initial_observation="start",
                steps=[_make_step(reward=0.5)],
                total_reward=0.4,
            )

    def test_finished_at_cannot_precede_started_at(self):
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

    def test_schema_version_defaults(self):
        log = _make_log()
        assert log.schema_version == SCHEMA_VERSION

    def test_num_steps_is_derived_from_steps(self):
        log = _make_log(steps=[_make_step(), _make_step()])
        assert log.num_steps == 2

    def test_total_tokens_is_derived_from_steps(self):
        step = _make_step(llm_calls=[_make_llm_call(prompt=10, completion=20)])
        log = _make_log(steps=[step])
        assert log.total_tokens == 30

    def test_total_cost_is_derived_from_steps(self):
        step = _make_step(llm_calls=[_make_llm_call(cost=0.05)])
        log = _make_log(steps=[step])
        assert log.total_cost_usd == 0.05

    def test_system_prompt_stored(self):
        log = _make_log(system_prompt="You are a helpful math tutor.")
        assert log.system_prompt == "You are a helpful math tutor."

    @pytest.mark.parametrize(
        "outcome",
        ["success", "failure", "truncated"],
    )
    def test_episode_outcome_values(self, outcome):
        log = _make_log(episode_outcome=outcome)
        assert log.episode_outcome == outcome

    def test_log_with_multiple_llm_steps(self):
        s1 = _make_step(step_index=1, llm_calls=[_make_llm_call(10, 20, 0.01)])
        s2 = _make_step(step_index=2, llm_calls=[_make_llm_call(15, 25, 0.02)])
        log = _make_log(steps=[s1, s2])
        assert log.total_tokens == 70
        assert abs(log.total_cost_usd - 0.03) < 1e-9

    def test_log_with_none_cost_steps(self):
        s1 = _make_step(step_index=1, llm_calls=[_make_llm_call(10, 20, cost=None)])
        s2 = _make_step(step_index=2, llm_calls=[_make_llm_call(5, 5, cost=0.01)])
        log = _make_log(steps=[s1, s2])
        assert abs(log.total_cost_usd - 0.01) < 1e-9

    def test_backward_compatible_empty_log(self):
        log = _make_log()
        assert log.steps == []
        assert log.total_tokens == 0
        assert log.total_cost_usd == 0.0
        assert log.episode_outcome is None
        assert log.system_prompt is None

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_environment_id_rejected(self, blank):
        with pytest.raises(ValidationError):
            _make_log(environment_id=blank)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_initial_observation_rejected(self, blank):
        with pytest.raises(ValidationError):
            _make_log(initial_observation=blank)

    def test_steps_without_llm_calls_aggregate_to_zero(self):
        s1 = _make_step(step_index=1, reward=0.5)
        s2 = _make_step(step_index=2, reward=0.5)
        log = _make_log(steps=[s1, s2])
        assert log.total_tokens == 0
        assert log.total_cost_usd == 0.0

    def test_multiple_llm_calls_per_step_aggregation(self):
        s1 = _make_step(
            step_index=1,
            llm_calls=[_make_llm_call(10, 20, 0.01), _make_llm_call(5, 15, 0.005)],
        )
        s2 = _make_step(
            step_index=2,
            llm_calls=[_make_llm_call(8, 12, 0.008)],
        )
        log = _make_log(steps=[s1, s2])
        assert log.total_tokens == 30 + 20 + 20
        assert abs(log.total_cost_usd - 0.023) < 1e-9

    def test_cost_zero_differs_from_none(self):
        s1 = _make_step(step_index=1, llm_calls=[_make_llm_call(10, 20, cost=0.0)])
        s2 = _make_step(step_index=2, llm_calls=[_make_llm_call(10, 20, cost=None)])
        log = _make_log(steps=[s1, s2])
        assert log.total_cost_usd == 0.0
        assert log.steps[0].llm_calls[0].cost_usd == 0.0
        assert log.steps[1].llm_calls[0].cost_usd is None


# ===========================================================================
# Outcome derivation
# ===========================================================================


class TestDeriveOutcome:
    """Tests for _derive_outcome helper."""

    def test_empty_steps(self):
        assert _derive_outcome([]) is None

    def test_success_from_reward(self):
        step = _make_step(terminated=True, truncated=False, reward=1.0)
        assert _derive_outcome([step]) == "success"

    def test_failure_from_reward(self):
        step = _make_step(terminated=True, truncated=False, reward=0.0)
        assert _derive_outcome([step]) == "failure"

    def test_negative_reward_failure(self):
        step = _make_step(terminated=True, truncated=False, reward=-0.1)
        assert _derive_outcome([step]) == "failure"

    def test_truncated(self):
        step = _make_step(terminated=False, truncated=True, reward=0.5)
        assert _derive_outcome([step]) == "truncated"

    def test_truncated_takes_precedence_over_terminated(self):
        step = _make_step(terminated=True, truncated=True, reward=1.0)
        assert _derive_outcome([step]) == "truncated"

    def test_neither(self):
        step = _make_step(terminated=False, truncated=False)
        assert _derive_outcome([step]) is None

    @pytest.mark.parametrize(
        ("info_key", "info_val", "expected"),
        [
            ("correct", True, "success"),
            ("correct", False, "failure"),
            ("is_correct", True, "success"),
            ("is_correct", False, "failure"),
            ("success", True, "success"),
            ("success", False, "failure"),
            ("task_success", True, "success"),
            ("task_success", False, "failure"),
        ],
    )
    def test_info_key_takes_precedence(self, info_key, info_val, expected):
        step = _make_step(
            terminated=True,
            truncated=False,
            reward=0.0,
            info={info_key: info_val},
        )
        assert _derive_outcome([step]) == expected

    def test_info_correct_overrides_reward(self):
        """GEM MathEnv returns correct=True with reward=1.0, but correct=False with reward=0.0.
        The info key should be authoritative even if reward contradicts it."""
        step = _make_step(
            terminated=True,
            truncated=False,
            reward=1.0,
            info={"correct": False},
        )
        assert _derive_outcome([step]) == "failure"

    def test_info_key_not_present_falls_back_to_reward(self):
        step = _make_step(
            terminated=True,
            truncated=False,
            reward=1.0,
            info={"suffix": "next"},
        )
        assert _derive_outcome([step]) == "success"

    def test_info_truthy_int_value(self):
        step = _make_step(terminated=True, truncated=False, reward=0.0, info={"correct": 1})
        assert _derive_outcome([step]) == "success"

    def test_info_true_overrides_negative_reward(self):
        step = _make_step(terminated=True, truncated=False, reward=-1.0, info={"correct": True})
        assert _derive_outcome([step]) == "success"


# ===========================================================================
# F1/F2: TrajectoryLogger Tests
# ===========================================================================


class TestTrajectoryLogger:
    """Tests for logger collection and persistence."""

    def test_save_writes_to_sqlite(self, tmp_path):
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

        db_path = logger.save(project_paths=paths)
        assert db_path.exists()
        assert db_path.suffix == ".db"
        assert db_path.parent == paths.logs

        loaded = load_trajectory(db_path, run_id=logger.last_run_id)
        assert loaded.environment_id == "game:GuessTheNumber-v0-easy"
        assert loaded.total_reward == 1.0
        assert loaded.num_steps == 2

    def test_build_log_requires_initial_state(self):
        logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy")
        with pytest.raises(ValueError, match="initial state is not set"):
            logger.build_log()

    def test_set_initial_state_cannot_be_called_twice(self):
        logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy")
        logger.set_initial_state("first")
        with pytest.raises(RuntimeError, match="already been set"):
            logger.set_initial_state("second")

    def test_system_prompt_stored(self):
        logger = TrajectoryLogger(environment_id="math:Math12K")
        logger.set_system_prompt("Solve step by step.")
        logger.set_initial_state("What is 2+2?")
        logger.add_step(
            action="\\\\boxed{4}",
            observation="done",
            reward=1.0,
            terminated=True,
            truncated=False,
        )
        log = logger.build_log()
        assert log.system_prompt == "Solve step by step."

    def test_build_log_schema_version(self):
        logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy")
        logger.set_initial_state("start")
        log = logger.build_log()
        assert log.schema_version == SCHEMA_VERSION

    def test_build_log_outcome_success(self):
        logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy")
        logger.set_initial_state("start")
        logger.add_step(action="a", observation="o", reward=1.0, terminated=True, truncated=False)
        log = logger.build_log()
        assert log.episode_outcome == "success"

    def test_build_log_outcome_failure(self):
        logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy")
        logger.set_initial_state("start")
        logger.add_step(action="a", observation="o", reward=0.0, terminated=True, truncated=False)
        log = logger.build_log()
        assert log.episode_outcome == "failure"

    def test_build_log_outcome_truncated(self):
        logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy")
        logger.set_initial_state("start")
        logger.add_step(action="a", observation="o", reward=0.5, terminated=False, truncated=True)
        log = logger.build_log()
        assert log.episode_outcome == "truncated"

    def test_build_log_token_and_cost_aggregation(self):
        logger = TrajectoryLogger(environment_id="math:Math12K")
        logger.set_initial_state("problem")
        logger.add_step(
            action="a1",
            observation="o1",
            reward=0.0,
            terminated=False,
            truncated=False,
            llm_calls=[_make_llm_call(10, 20, 0.01)],
        )
        logger.add_step(
            action="a2",
            observation="o2",
            reward=1.0,
            terminated=True,
            truncated=False,
            llm_calls=[_make_llm_call(15, 25, 0.02)],
        )
        log = logger.build_log()
        assert log.total_tokens == 70
        assert abs(log.total_cost_usd - 0.03) < 1e-9
        assert log.num_steps == 2

    def test_build_log_multi_call_per_step_aggregation(self):
        logger = TrajectoryLogger(environment_id="math:Math12K")
        logger.set_initial_state("problem")
        logger.add_step(
            action="think",
            observation="hint",
            reward=0.0,
            terminated=False,
            truncated=False,
            llm_calls=[
                _make_llm_call(10, 20, 0.01),
                _make_llm_call(5, 15, 0.005),
            ],
        )
        logger.add_step(
            action="answer",
            observation="correct",
            reward=1.0,
            terminated=True,
            truncated=False,
            llm_calls=[_make_llm_call(8, 12, 0.008)],
        )
        log = logger.build_log()
        assert log.total_tokens == 30 + 20 + 20
        assert abs(log.total_cost_usd - 0.023) < 1e-9

    def test_add_step_with_tool_calls(self):
        logger = TrajectoryLogger(environment_id="code:CodeContest")
        logger.set_initial_state("problem")
        tc = ToolCall(
            tool_name="python_exec",
            tool_input="print(42)",
            tool_output="42",
            success=True,
            duration_ms=100.0,
        )
        step = logger.add_step(
            action="run code",
            observation="output",
            reward=1.0,
            terminated=True,
            truncated=False,
            tool_calls=[tc],
        )
        assert len(step.tool_calls) == 1
        assert step.tool_calls[0].tool_name == "python_exec"

    def test_add_step_auto_timestamp(self):
        logger = TrajectoryLogger(environment_id="math:Math12K")
        logger.set_initial_state("start")
        before = datetime.now(UTC)
        step = logger.add_step(
            action="a", observation="o", reward=0.0, terminated=False, truncated=False
        )
        after = datetime.now(UTC)
        assert step.timestamp is not None
        assert before <= step.timestamp <= after

    def test_save_includes_new_fields(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        logger = TrajectoryLogger(environment_id="math:Math12K", seed=1)
        logger.set_system_prompt("Be precise.")
        logger.set_initial_state("What is 1+1?")
        logger.add_step(
            action="\\\\boxed{2}",
            observation="correct",
            reward=1.0,
            terminated=True,
            truncated=False,
            llm_calls=[_make_llm_call(5, 10, 0.005)],
        )
        db_path = logger.save(project_paths=paths)
        loaded = load_trajectory(db_path, run_id=logger.last_run_id)

        assert loaded.schema_version == SCHEMA_VERSION
        assert loaded.system_prompt == "Be precise."
        assert loaded.episode_outcome == "success"
        assert loaded.num_steps == 1
        assert loaded.total_tokens == 15
        assert abs(loaded.total_cost_usd - 0.005) < 1e-9

    def test_save_failure_leaves_no_tmp_file(self, tmp_path):
        logger = TrajectoryLogger(environment_id="game:GuessTheNumber-v0-easy")
        paths = ProjectPaths(root=tmp_path)
        with pytest.raises(ValueError, match="initial state is not set"):
            logger.save(project_paths=paths)
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == []


# ===========================================================================
# F2: Trajectory Loading and Filtering Tests
# ===========================================================================


class TestTrajectoryLoadAndFilter:
    """Tests for load_trajectory, load_all_trajectories, and filter_trajectories."""

    def _save_log(self, tmp_path, logger: TrajectoryLogger, name: str) -> None:
        paths = ProjectPaths(root=tmp_path)
        log = logger.build_log()
        file_path = paths.logs / f"trajectory_{name}_{log.run_id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(log.model_dump_json(indent=2), encoding="utf-8")

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trajectory(tmp_path / "does_not_exist.json")

    def test_load_corrupt_json_raises(self, tmp_path):
        bad_file = tmp_path / "trajectory_bad.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        with pytest.raises((json.JSONDecodeError, ValidationError)):
            load_trajectory(bad_file)

    def test_load_all_skips_corrupt_files(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        paths.logs.mkdir(parents=True, exist_ok=True)

        logger = TrajectoryLogger(environment_id="math:Math12K", seed=1)
        logger.set_initial_state("problem")
        logger.add_step(action="a", observation="o", reward=1.0, terminated=True, truncated=False)
        logger.save(project_paths=paths)

        (paths.logs / "trajectory_corrupt_badid.json").write_text("{not json", encoding="utf-8")
        (paths.logs / "trajectory_empty_badid.json").write_text("", encoding="utf-8")

        logs = load_all_trajectories(paths.logs)
        assert len(logs) == 1
        assert logs[0].environment_id == "math:Math12K"

    def test_load_single_trajectory(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        logger = TrajectoryLogger(environment_id="math:Math12K", seed=1)
        logger.set_initial_state("problem")
        logger.add_step(action="a", observation="o", reward=1.0, terminated=True, truncated=False)
        db_path = logger.save(project_paths=paths)

        loaded = load_trajectory(db_path, run_id=logger.last_run_id)
        assert loaded.environment_id == "math:Math12K"
        assert loaded.total_reward == 1.0
        assert loaded.num_steps == 1

    def test_load_all_trajectories(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        for i in range(3):
            logger = TrajectoryLogger(environment_id="math:Math12K", seed=i)
            logger.set_initial_state(f"problem_{i}")
            logger.add_step(
                action="a",
                observation="o",
                reward=float(i),
                terminated=True,
                truncated=False,
            )
            logger.save(project_paths=paths)

        logs = load_all_trajectories(paths.logs)
        assert len(logs) == 3
        assert logs[0].started_at <= logs[1].started_at <= logs[2].started_at

    def test_load_all_empty_directory(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        logs = load_all_trajectories(paths.logs)
        assert logs == []

    def test_round_trip_serialization(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        logger = TrajectoryLogger(environment_id="code:CodeContest", seed=42)
        logger.set_system_prompt("Write efficient code.")
        logger.set_initial_state("Implement fizzbuzz")
        tc = ToolCall(
            tool_name="python_exec",
            tool_input="code",
            tool_output="ok",
            success=True,
            duration_ms=50.0,
        )
        logger.add_step(
            action="solution",
            observation="passed",
            reward=1.0,
            terminated=True,
            truncated=False,
            tool_calls=[tc],
            llm_calls=[_make_llm_call(20, 30, 0.02)],
        )
        db_path = logger.save(project_paths=paths)

        loaded = load_trajectory(db_path, run_id=logger.last_run_id)
        assert loaded.system_prompt == "Write efficient code."
        assert loaded.episode_outcome == "success"
        assert len(loaded.steps[0].tool_calls) == 1
        assert loaded.steps[0].tool_calls[0].tool_name == "python_exec"
        assert len(loaded.steps[0].llm_calls) == 1
        assert loaded.steps[0].llm_calls[0].total_tokens == 50

    def test_round_trip_multiple_llm_calls_per_step(self, tmp_path):
        paths = ProjectPaths(root=tmp_path)
        logger = TrajectoryLogger(environment_id="math:Math12K", seed=7)
        logger.set_initial_state("problem")
        logger.add_step(
            action="think+act",
            observation="result",
            reward=1.0,
            terminated=True,
            truncated=False,
            llm_calls=[
                _make_llm_call(10, 20, 0.01),
                _make_llm_call(5, 15, None),
            ],
        )
        db_path = logger.save(project_paths=paths)

        loaded = load_trajectory(db_path, run_id=logger.last_run_id)
        assert len(loaded.steps[0].llm_calls) == 2
        assert loaded.steps[0].llm_calls[0].total_tokens == 30
        assert loaded.steps[0].llm_calls[0].cost_usd == 0.01
        assert loaded.steps[0].llm_calls[1].total_tokens == 20
        assert loaded.steps[0].llm_calls[1].cost_usd is None
        assert loaded.total_tokens == 50
        assert abs(loaded.total_cost_usd - 0.01) < 1e-9

    @pytest.mark.parametrize(
        ("outcome", "expected_count"),
        [
            ("success", 2),
            ("failure", 1),
            ("truncated", 0),
            (None, 3),
        ],
    )
    def test_filter_by_outcome(self, outcome, expected_count):
        now = datetime.now(UTC)
        logs = [
            _make_log(
                environment_id="math:Math12K",
                episode_outcome="success",
                started_at=now,
                finished_at=now + timedelta(seconds=1),
            ),
            _make_log(
                environment_id="math:GSM8K",
                episode_outcome="failure",
                started_at=now,
                finished_at=now + timedelta(seconds=1),
            ),
            _make_log(
                environment_id="game:GuessTheNumber-v0-easy",
                episode_outcome="success",
                started_at=now,
                finished_at=now + timedelta(seconds=1),
            ),
        ]

        filtered = filter_trajectories(logs, outcome=outcome)
        assert len(filtered) == expected_count

    def test_filter_by_environment_id(self):
        now = datetime.now(UTC)
        logs = [
            _make_log(
                environment_id="math:Math12K",
                started_at=now,
                finished_at=now + timedelta(seconds=1),
            ),
            _make_log(
                environment_id="math:GSM8K",
                started_at=now,
                finished_at=now + timedelta(seconds=1),
            ),
        ]
        filtered = filter_trajectories(logs, environment_id="math:Math12K")
        assert len(filtered) == 1
        assert filtered[0].environment_id == "math:Math12K"


# ===========================================================================
# F3: extract_llm_calls_from_tracker Tests
# ===========================================================================


class _FakeTracker:
    """Mimics dspy.track_usage() tracker with a usage_data dict."""

    def __init__(self, usage_data: dict[str, list[dict]]):
        self.usage_data = usage_data


class TestExtractLLMCallsFromTracker:
    """Tests for the DSPy usage tracker conversion helper."""

    def test_single_model_single_call(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [
                    {"prompt_tokens": 10, "completion_tokens": 20},
                ],
            }
        )
        calls = extract_llm_calls_from_tracker(tracker)
        assert len(calls) == 1
        assert calls[0].model_id == "bedrock/qwen3-1.7b"
        assert calls[0].prompt_tokens == 10
        assert calls[0].completion_tokens == 20
        assert calls[0].total_tokens == 30

    def test_single_model_multiple_calls(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [
                    {"prompt_tokens": 10, "completion_tokens": 20},
                    {"prompt_tokens": 15, "completion_tokens": 25},
                ],
            }
        )
        calls = extract_llm_calls_from_tracker(tracker)
        assert len(calls) == 2
        assert calls[0].total_tokens == 30
        assert calls[1].total_tokens == 40

    def test_multiple_models(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [
                    {"prompt_tokens": 10, "completion_tokens": 20},
                ],
                "bedrock/claude-sonnet": [
                    {"prompt_tokens": 100, "completion_tokens": 200},
                ],
            }
        )
        calls = extract_llm_calls_from_tracker(tracker)
        assert len(calls) == 2
        model_ids = {c.model_id for c in calls}
        assert model_ids == {"bedrock/qwen3-1.7b", "bedrock/claude-sonnet"}

    def test_empty_usage_data(self):
        tracker = _FakeTracker({})
        calls = extract_llm_calls_from_tracker(tracker)
        assert calls == []

    def test_no_usage_data_attribute(self):
        tracker = object()
        calls = extract_llm_calls_from_tracker(tracker)
        assert calls == []

    def test_cost_included_when_present(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [
                    {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.005},
                ],
            }
        )
        calls = extract_llm_calls_from_tracker(tracker)
        assert calls[0].cost_usd == 0.005

    def test_cost_none_when_absent(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [
                    {"prompt_tokens": 10, "completion_tokens": 20},
                ],
            }
        )
        calls = extract_llm_calls_from_tracker(tracker)
        assert calls[0].cost_usd is None

    def test_missing_token_fields_default_to_zero(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [{}],
            }
        )
        calls = extract_llm_calls_from_tracker(tracker)
        assert len(calls) == 1
        assert calls[0].prompt_tokens == 0
        assert calls[0].completion_tokens == 0
        assert calls[0].total_tokens == 0

    def test_integrates_with_add_step(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [
                    {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.01},
                    {"prompt_tokens": 5, "completion_tokens": 15, "cost": 0.005},
                ],
            }
        )
        llm_calls = extract_llm_calls_from_tracker(tracker)

        logger = TrajectoryLogger(environment_id="math:Math12K")
        logger.set_initial_state("problem")
        logger.add_step(
            action="answer",
            observation="correct",
            reward=1.0,
            terminated=True,
            truncated=False,
            llm_calls=llm_calls,
        )
        log = logger.build_log()
        assert log.total_tokens == 50
        assert abs(log.total_cost_usd - 0.015) < 1e-9

    def test_extra_keys_in_usage_dict_ignored(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "future_field": "whatever",
                        "cache_hits": 3,
                    },
                ],
            }
        )
        calls = extract_llm_calls_from_tracker(tracker)
        assert len(calls) == 1
        assert calls[0].total_tokens == 30

    def test_cost_zero_is_preserved(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [
                    {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                ],
            }
        )
        calls = extract_llm_calls_from_tracker(tracker)
        assert calls[0].cost_usd == 0.0
        assert calls[0].cost_usd is not None

    def test_multiple_models_multiple_calls_each(self):
        tracker = _FakeTracker(
            {
                "bedrock/qwen3-1.7b": [
                    {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.01},
                    {"prompt_tokens": 5, "completion_tokens": 15, "cost": 0.005},
                ],
                "bedrock/claude-sonnet": [
                    {"prompt_tokens": 100, "completion_tokens": 200, "cost": 0.1},
                ],
            }
        )
        calls = extract_llm_calls_from_tracker(tracker)
        assert len(calls) == 3
        total_tokens = sum(c.total_tokens for c in calls)
        assert total_tokens == 30 + 20 + 300
        total_cost = sum(c.cost_usd for c in calls if c.cost_usd is not None)
        assert abs(total_cost - 0.115) < 1e-9
