"""Tests for frozen experiment configuration schema (models/experiment.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trajectory_aware_gym.models.experiment import (
    ComparisonProtocol,
    CostBudget,
    DatasetSplit,
    DSPyModuleType,
    EnvironmentConfig,
    EnvironmentType,
    EvalProtocol,
    ExperimentConfig,
    FitnessOverride,
    GEPABudgetConfig,
    PromptBaselineConfig,
    ReflectionModelConfig,
    RLBaselineResult,
    RLTrainingDetails,
    SeedConfig,
    TaskModelConfig,
    ToolType,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"


def _minimal_experiment(**overrides) -> dict:
    """Return minimal valid ExperimentConfig kwargs."""
    base = {
        "name": "test",
        "seed_prompt": "You are a test solver. Answer with \\boxed{...}.",
        "environment": {
            "gem_env_id": "math:Orz57K",
            "env_type": "math",
            "dspy_module": "react",
            "train_size": 100,
            "max_steps": 10,
            "discount_gamma": 1.0,
            "tools": ["python_exec"],
        },
        "task_models": [
            {
                "name": "Qwen3-1.7B",
                "model_id": "ollama/qwen3-1.7b-base",
                "provider": "ollama",
                "parameter_count": "1.7B",
            },
        ],
        "reflection_model": {
            "model_id": "anthropic.claude-sonnet-4-5-v2:0",
        },
        "gepa_budget": {
            "mode": "medium",
            "tasks_per_minibatch": 3,
        },
        "seeds": {"data_seed": 42, "replication_seeds": [42, 123, 456]},
        "num_replications": 3,
    }
    base.update(overrides)
    return base


class TestDatasetSplit:
    """Tests for reproducible dataset partition metadata."""

    def test_valid_dataset_split(self):
        dataset = DatasetSplit(
            hf_dataset_id="axon-rl/ORZ-57k",
            total_train_size=57_000,
            subsample_size=500,
            subsample_strategy="uniform",
            eval_dataset_id="axon-rl/math-eval",
            eval_split="MATH500",
            eval_size=500,
        )
        assert dataset.hf_dataset_id == "axon-rl/ORZ-57k"
        assert dataset.subsample_size == 500

    def test_subsample_must_not_exceed_total(self):
        with pytest.raises(ValidationError, match="subsample_size"):
            DatasetSplit(
                hf_dataset_id="axon-rl/ORZ-57k",
                total_train_size=100,
                subsample_size=101,
                subsample_strategy="uniform",
                eval_dataset_id="axon-rl/math-eval",
                eval_split="MATH500",
                eval_size=500,
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("total_train_size", 0),
            ("subsample_size", 0),
            ("eval_size", 0),
        ],
    )
    def test_rejects_invalid_sizes(self, field, value):
        kwargs = {
            "hf_dataset_id": "axon-rl/HotpotQA",
            "total_train_size": 90_400,
            "subsample_size": 500,
            "subsample_strategy": "uniform",
            "eval_dataset_id": "axon-rl/search-eval",
            "eval_split": "hotpotqa",
            "eval_size": 512,
            field: value,
        }
        with pytest.raises(ValidationError):
            DatasetSplit(**kwargs)


class TestEnvironmentConfig:
    """Tests for per-environment configuration."""

    def test_orz57k_factory(self):
        env = EnvironmentConfig.orz57k()
        assert env.gem_env_id == "math:Orz57K"
        assert env.env_type == EnvironmentType.MATH
        assert env.dspy_module == DSPyModuleType.REACT
        assert env.train_size == 500
        assert env.max_steps == 10
        assert env.discount_gamma == 1.0
        assert env.tools == [ToolType.PYTHON_EXEC]
        assert env.dataset is not None
        assert env.dataset.eval_split == "MATH500"

    def test_hotpotqa_factory(self):
        env = EnvironmentConfig.hotpotqa()
        assert env.gem_env_id == "qa:HotpotQA"
        assert env.env_type == EnvironmentType.QA
        assert env.dspy_module == DSPyModuleType.REACT
        assert env.discount_gamma == 0.9
        assert env.tools == [ToolType.WEB_SEARCH]
        assert env.dataset is not None
        assert env.dataset.eval_size == 512

    def test_effective_val_size_defaults_to_10_pct(self):
        env = EnvironmentConfig.orz57k()
        assert env.effective_val_size == 50

    def test_effective_val_size_explicit(self):
        env = EnvironmentConfig(
            gem_env_id="math:Orz57K",
            env_type="math",
            dspy_module="react",
            train_size=500,
            val_size=25,
            max_steps=10,
            discount_gamma=1.0,
        )
        assert env.effective_val_size == 25

    def test_effective_val_size_minimum_is_one(self):
        env = EnvironmentConfig(
            gem_env_id="math:Orz57K",
            env_type="math",
            dspy_module="react",
            train_size=5,
            max_steps=10,
            discount_gamma=1.0,
        )
        assert env.effective_val_size == 1

    def test_frozen(self):
        env = EnvironmentConfig.orz57k()
        with pytest.raises(ValidationError):
            env.train_size = 999

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("train_size", 0),
            ("train_size", -1),
            ("max_steps", 0),
            ("discount_gamma", -0.1),
            ("discount_gamma", 1.1),
        ],
    )
    def test_rejects_invalid_values(self, field, value):
        kwargs = {
            "gem_env_id": "math:Orz57K",
            "env_type": "math",
            "dspy_module": "react",
            "train_size": 100,
            "max_steps": 10,
            "discount_gamma": 1.0,
            field: value,
        }
        with pytest.raises(ValidationError):
            EnvironmentConfig(**kwargs)

    def test_default_test_split(self):
        env = EnvironmentConfig.orz57k()
        assert env.test_split == "test"


class TestEvalProtocol:
    """Evaluation protocol defaults match GEM paper."""

    def test_gem_defaults(self):
        proto = EvalProtocol()
        assert proto.max_response_tokens == 4096
        assert proto.temperature_train == 1.0
        assert proto.temperature_eval == 0.0
        assert proto.top_p == 1.0
        assert proto.top_k == -1
        assert proto.rollouts_per_task == 5
        assert proto.tost_margin == 0.05
        assert proto.tost_alpha == 0.05
        assert proto.bootstrap_iterations == 1000

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_response_tokens", 0),
            ("top_p", 1.5),
            ("tost_margin", 0.0),
            ("tost_alpha", 1.0),
            ("rollouts_per_task", 0),
            ("bootstrap_iterations", 50),
        ],
    )
    def test_rejects_invalid_values(self, field, value):
        with pytest.raises(ValidationError):
            EvalProtocol(**{field: value})


class TestGEPABudgetConfig:
    """GEPA budget config — maps to dspy.GEPA(auto=mode, ...)."""

    @pytest.mark.parametrize("mode", ["light", "medium", "heavy"])
    def test_valid_modes(self, mode):
        budget = GEPABudgetConfig(mode=mode, tasks_per_minibatch=3)
        assert budget.mode == mode
        assert budget.tasks_per_minibatch == 3

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValidationError, match="mode"):
            GEPABudgetConfig(mode="invalid", tasks_per_minibatch=3)

    def test_tasks_per_minibatch_must_be_positive(self):
        with pytest.raises(ValidationError, match="tasks_per_minibatch"):
            GEPABudgetConfig(mode="medium", tasks_per_minibatch=0)


class TestSeedConfig:
    """Seed configuration for reproducibility."""

    def test_defaults(self):
        seeds = SeedConfig()
        assert seeds.data_seed == 42
        assert seeds.replication_seeds == (42, 123, 456)

    def test_custom_seeds(self):
        seeds = SeedConfig(data_seed=99, replication_seeds=(1, 2))
        assert seeds.data_seed == 99
        assert seeds.replication_seeds == (1, 2)


class TestRLBaselines:
    """RL baseline reference data from GEM paper."""

    def test_training_details_defaults(self):
        details = RLTrainingDetails()
        assert details.learning_rate == 1e-6
        assert details.optimizer == "AdamW"
        assert details.adam_betas == (0.9, 0.95)
        assert details.policy_clip == 0.2
        assert details.kl_coefficient == 0.0
        assert details.inner_epochs == 2
        assert details.gae_lambda == 0.95
        assert details.training_steps == 500

    def test_baseline_result(self):
        result = RLBaselineResult(
            algorithm="REINFORCE+ReBN",
            environment="math:Orz57K",
            model_name="Qwen3-4B",
            model_size="4B",
            success_rate=0.71,
            tool_augmented=True,
            training_condition="single",
            metric_name="pass@1",
            source_detail="Table 1, MATH500, Base+RL (with tool)",
            estimated=False,
        )
        assert result.success_rate == 0.71
        assert result.tool_augmented is True
        assert result.training_condition == "single"
        assert result.estimated is False
        assert result.source == "GEM (Liu et al., 2025)"

    def test_success_rate_bounded(self):
        with pytest.raises(ValidationError):
            RLBaselineResult(
                algorithm="REINFORCE+ReBN",
                environment="math:Orz57K",
                model_name="Qwen3-4B",
                model_size="4B",
                success_rate=1.5,
            )


class TestPromptBaselineConfig:
    """Prompt baseline metadata and validation."""

    def test_zero_shot_factory(self):
        baseline = PromptBaselineConfig.zero_shot()
        assert baseline.name == "zero-shot"
        assert baseline.optimizer is None
        assert baseline.requires_optimization is False

    def test_miprov2_factory(self):
        baseline = PromptBaselineConfig.miprov2()
        assert baseline.name == "miprov2"
        assert baseline.optimizer == "MIPROv2"
        assert baseline.requires_optimization is True

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "name": "zero-shot",
                "description": "bad zero-shot",
                "optimizer": "MIPROv2",
                "requires_optimization": False,
            },
            {
                "name": "miprov2",
                "description": "bad miprov2",
                "optimizer": None,
                "requires_optimization": True,
            },
        ],
    )
    def test_rejects_inconsistent_configuration(self, kwargs):
        with pytest.raises(ValidationError):
            PromptBaselineConfig(**kwargs)


class TestComparisonProtocol:
    """How GEPA is compared against baselines."""

    def test_defaults(self):
        protocol = ComparisonProtocol()
        assert protocol.primary_metric == "success_rate"
        assert protocol.equivalence_test == "TOST"
        assert protocol.primary_rl_baseline == "REINFORCE+ReBN"
        assert [baseline.name for baseline in protocol.prompt_baselines] == ["zero-shot", "miprov2"]

    def test_custom_prompt_baselines(self):
        protocol = ComparisonProtocol(
            primary_rl_baseline="REINFORCE+ReBN (+tool, single)",
            prompt_baselines=[PromptBaselineConfig.zero_shot()],
        )
        assert protocol.primary_rl_baseline == "REINFORCE+ReBN (+tool, single)"
        assert len(protocol.prompt_baselines) == 1

    def test_requires_prompt_baseline(self):
        with pytest.raises(ValidationError):
            ComparisonProtocol(prompt_baselines=[])


class TestCostBudget:
    """Cost budget with buffer computation."""

    def test_defaults(self):
        budget = CostBudget()
        assert budget.total_budget_usd == 2500.0
        assert budget.buffer_pct == 0.25

    def test_effective_budget(self):
        budget = CostBudget(total_budget_usd=1000.0, buffer_pct=0.25)
        assert budget.effective_budget_usd == 1250.0

    def test_zero_buffer(self):
        budget = CostBudget(total_budget_usd=100.0, buffer_pct=0.0)
        assert budget.effective_budget_usd == 100.0


class TestFitnessOverride:
    """Per-environment fitness parameter overrides."""

    def test_all_none_by_default(self):
        override = FitnessOverride()
        assert override.gamma is None
        assert override.lambda_ is None
        assert override.loop_penalty_weight is None

    def test_partial_override(self):
        override = FitnessOverride(gamma=0.9)
        assert override.gamma == 0.9
        assert override.lambda_ is None

    def test_rejects_invalid_gamma(self):
        with pytest.raises(ValidationError):
            FitnessOverride(gamma=1.5)


class TestExperimentConfig:
    """Top-level experiment configuration."""

    def test_minimal_valid(self):
        config = ExperimentConfig(**_minimal_experiment())
        assert config.name == "test"
        assert config.environment.gem_env_id == "math:Orz57K"
        assert len(config.task_models) == 1
        assert config.num_replications == 3
        assert config.comparison_protocol is None

    def test_seeds_must_match_replications(self):
        with pytest.raises(ValidationError, match="replication_seeds"):
            ExperimentConfig(
                **_minimal_experiment(
                    seeds={"data_seed": 42, "replication_seeds": [42, 123]},
                    num_replications=3,
                )
            )

    def test_frozen(self):
        config = ExperimentConfig(**_minimal_experiment())
        with pytest.raises(ValidationError):
            config.name = "changed"

    def test_requires_at_least_one_task_model(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(**_minimal_experiment(task_models=[]))

    def test_default_eval_protocol(self):
        config = ExperimentConfig(**_minimal_experiment())
        assert config.eval_protocol.temperature_eval == 0.0
        assert config.eval_protocol.rollouts_per_task == 5

    def test_optional_comparison_protocol(self):
        config = ExperimentConfig(
            **_minimal_experiment(
                comparison_protocol={
                    "primary_metric": "success_rate",
                    "equivalence_test": "TOST",
                    "primary_rl_baseline": "REINFORCE+ReBN (+tool)",
                    "prompt_baselines": [
                        {
                            "name": "zero-shot",
                            "description": "Unoptimized DSPy agent with hand-written instructions",
                            "optimizer": None,
                            "requires_optimization": False,
                        },
                    ],
                }
            )
        )
        assert config.comparison_protocol is not None
        assert config.comparison_protocol.primary_metric == "success_rate"


class TestYAMLRoundTrip:
    """Serialization and deserialization to/from YAML."""

    def test_round_trip(self, tmp_path):
        original = ExperimentConfig(
            **_minimal_experiment(
                comparison_protocol=ComparisonProtocol().model_dump(mode="json", exclude_none=True)
            )
        )
        yaml_path = tmp_path / "test_config.yaml"
        original.to_yaml(yaml_path)

        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.name == original.name
        assert loaded.environment.gem_env_id == original.environment.gem_env_id
        assert loaded.num_replications == original.num_replications
        assert loaded.seeds.replication_seeds == original.seeds.replication_seeds
        assert loaded.comparison_protocol is not None

    def test_creates_parent_dirs(self, tmp_path):
        config = ExperimentConfig(**_minimal_experiment())
        nested_path = tmp_path / "a" / "b" / "config.yaml"
        config.to_yaml(nested_path)
        assert nested_path.exists()


class TestProductionConfigs:
    """Validate all experiment YAML files in experiments/."""

    @pytest.mark.parametrize(
        "experiment_dir",
        [
            "orz57k-tool",
            "orz57k-notool",
            "hotpotqa-tool",
            "hotpotqa-notool",
            "quick-test",
            "math-dry-run",
        ],
    )
    def test_loads_and_validates(self, experiment_dir):
        config_path = EXPERIMENTS_DIR / experiment_dir / "config.yaml"
        config = ExperimentConfig.from_yaml(config_path)
        assert config.name == experiment_dir

    def test_orz57k_tool_values(self):
        config = ExperimentConfig.from_yaml(EXPERIMENTS_DIR / "orz57k-tool" / "config.yaml")
        assert config.environment.gem_env_id == "math:Orz57K"
        assert config.environment.discount_gamma == 1.0
        assert config.environment.max_steps == 10
        assert config.environment.tools == [ToolType.PYTHON_EXEC]
        assert config.environment.dataset is not None
        assert config.environment.dataset.hf_dataset_id == "axon-rl/ORZ-57k"
        assert config.rl_baselines[0].success_rate == pytest.approx(0.71)
        assert config.rl_baselines[0].tool_augmented is True
        assert config.comparison_protocol is not None

    def test_orz57k_notool_values(self):
        config = ExperimentConfig.from_yaml(EXPERIMENTS_DIR / "orz57k-notool" / "config.yaml")
        assert config.environment.gem_env_id == "math:Orz57K"
        assert config.environment.discount_gamma == 1.0
        assert config.environment.tools == []
        assert config.rl_baselines[0].success_rate == pytest.approx(0.674)
        assert config.rl_baselines[0].tool_augmented is False

    def test_hotpotqa_tool_values(self):
        config = ExperimentConfig.from_yaml(EXPERIMENTS_DIR / "hotpotqa-tool" / "config.yaml")
        assert config.environment.gem_env_id == "qa:HotpotQA"
        assert config.environment.discount_gamma == 0.9
        assert config.environment.tools == [ToolType.WEB_SEARCH]
        assert config.environment.dataset is not None
        assert config.environment.dataset.eval_split == "hotpotqa"
        assert config.rl_baselines[0].success_rate == pytest.approx(0.432)
        assert config.rl_baselines[0].training_condition == "single"

    def test_hotpotqa_notool_values(self):
        config = ExperimentConfig.from_yaml(EXPERIMENTS_DIR / "hotpotqa-notool" / "config.yaml")
        assert config.environment.gem_env_id == "qa:HotpotQA"
        assert config.environment.discount_gamma == 0.9
        assert config.environment.tools == []
        assert config.rl_baselines[0].success_rate == pytest.approx(0.211)
        assert config.rl_baselines[0].tool_augmented is False

    def test_quick_test_is_lightweight(self):
        config = ExperimentConfig.from_yaml(EXPERIMENTS_DIR / "quick-test" / "config.yaml")
        assert config.num_replications == 1
        assert config.gepa_budget.mode == "light"
        assert config.environment.train_size == 50
        assert config.environment.gem_env_id == "math:Orz57K"
        assert config.environment.tools == [ToolType.PYTHON_EXEC]
        assert len(config.task_models) == 1

    def test_math_dry_run_is_minimal(self):
        config = ExperimentConfig.from_yaml(EXPERIMENTS_DIR / "math-dry-run" / "config.yaml")
        assert config.num_replications == 1
        assert config.environment.gem_env_id == "math:Orz57K"
        assert config.environment.max_steps == 5
        assert config.environment.train_size == 12
        assert config.environment.val_size == 3
        assert config.environment.tools == [ToolType.PYTHON_EXEC]
        assert config.eval_protocol.max_response_tokens == 1536
        assert len(config.task_models) == 1

    @pytest.mark.parametrize(
        "config_path",
        sorted(EXPERIMENTS_DIR.glob("*/config.yaml")),
        ids=lambda p: p.parent.name,
    )
    def test_seed_prompt_is_present_and_mentions_configured_tools(self, config_path):
        """Every experiment must ship a non-empty seed_prompt, and tool-enabled
        experiments must reference each enabled tool by the name the model
        will see. The runner applies ``_TOOL_NAME_ALIASES`` before showing
        tool names to the model, so we check the aliased (runtime) name.

        Prevents the failure mode that caused orz57k-tool to score far below
        orz57k-notool in earlier runs: a seed_prompt that never cues the
        model to use the tool at all.
        """
        from trajectory_aware_gym.adapters.gem_episode_runner import (
            _TOOL_NAME_ALIASES,
        )

        config = ExperimentConfig.from_yaml(config_path)
        assert config.seed_prompt.strip(), f"{config_path} has empty seed_prompt"

        for tool_name in config.environment.active_tool_names:
            runtime_name = _TOOL_NAME_ALIASES.get(tool_name, tool_name)
            assert runtime_name in config.seed_prompt, (
                f"{config_path} enables tool {tool_name!r} (runtime name: "
                f"{runtime_name!r}) but its seed_prompt never mentions it — "
                f"the model won't know to use it"
            )
