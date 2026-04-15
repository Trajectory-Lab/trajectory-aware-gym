"""Frozen experiment configuration schema for reproducible research.

This module defines the complete experiment specification — environments,
models, seeds, budgets, evaluation protocol, and RL baseline references.
All models are frozen (immutable) to guarantee reproducibility.

Separate from the runtime Settings singleton in config/core.py:
- Settings = infrastructure config (AWS creds, endpoints, log levels)
- ExperimentConfig = science config (what experiment to run)
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

# ── Enums ────────────────────────────────────────────────────────


class EnvironmentType(StrEnum):
    MATH = "math"
    CODE = "code"
    QA = "qa"


class DSPyModuleType(StrEnum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"


class ToolType(StrEnum):
    NONE = "none"
    PYTHON_EXEC = "python_exec"
    WEB_SEARCH = "web_search"


# ── Sub-models (all frozen) ──────────────────────────────────────


class DatasetSplit(BaseModel):
    """Exact dataset partition used for a single experiment."""

    model_config = {"frozen": True}

    hf_dataset_id: str = Field(description="HuggingFace training dataset ID")
    total_train_size: int = Field(ge=1, description="Full size of the source training dataset")
    subsample_size: int = Field(ge=1, description="Number of training tasks we actually use")
    subsample_strategy: Literal["uniform", "stratified"]
    eval_dataset_id: str = Field(description="HuggingFace evaluation dataset ID")
    eval_split: str = Field(description="Primary held-out evaluation split name")
    eval_size: int = Field(ge=1, description="Number of evaluation tasks in the primary split")

    @model_validator(mode="after")
    def _subsample_within_train_size(self) -> Self:
        if self.subsample_size > self.total_train_size:
            msg = (
                f"subsample_size ({self.subsample_size}) must be <= "
                f"total_train_size ({self.total_train_size})"
            )
            raise ValueError(msg)
        return self


class EnvironmentConfig(BaseModel):
    """Configuration for a single GEM environment."""

    model_config = {"frozen": True}

    gem_env_id: str = Field(description="GEM environment identifier, e.g. 'math:Orz57K'")
    env_type: EnvironmentType
    dspy_module: DSPyModuleType
    train_size: int = Field(ge=1, description="Number of training tasks (subsampled if needed)")
    val_size: int | None = Field(
        default=None,
        description="Validation set size. None means 10% of train_size.",
    )
    test_split: str = Field(
        default="test",
        description="Name of GEM's held-out test partition",
    )
    max_steps: int = Field(ge=1, description="Max steps per episode")
    discount_gamma: float = Field(ge=0.0, le=1.0, description="Discount factor from GEM paper")
    tools: list[ToolType] = Field(default_factory=list)
    dataset: DatasetSplit | None = None

    @property
    def effective_val_size(self) -> int:
        """GEPA validation size: explicit value or 10% of train_size."""
        if self.val_size is not None:
            return self.val_size
        return max(1, self.train_size // 10)

    @property
    def effective_eval_size(self) -> int:
        """Held-out eval size from dataset split, or falls back to effective_val_size."""
        if self.dataset is not None:
            return self.dataset.eval_size
        return self.effective_val_size

    @property
    def active_tool_names(self) -> list[str]:
        """Tool names excluding 'none', ready for GEMEpisodeRunner."""
        return [tool.value for tool in self.tools if tool.value != "none"]

    @classmethod
    def orz57k(cls) -> EnvironmentConfig:
        """Canonical Orz57K config matching the GEM paper's math training setup."""
        return cls(
            gem_env_id="math:Orz57K",
            env_type=EnvironmentType.MATH,
            dspy_module=DSPyModuleType.REACT,
            train_size=500,
            max_steps=10,
            discount_gamma=1.0,
            tools=[ToolType.PYTHON_EXEC],
            dataset=DatasetSplit(
                hf_dataset_id="axon-rl/ORZ-57k",
                total_train_size=57_000,
                subsample_size=500,
                subsample_strategy="uniform",
                eval_dataset_id="axon-rl/math-eval",
                eval_split="MATH500",
                eval_size=500,
            ),
        )

    @classmethod
    def hotpotqa(cls) -> EnvironmentConfig:
        """Canonical HotpotQA config matching GEM paper."""
        return cls(
            gem_env_id="qa:HotpotQA",
            env_type=EnvironmentType.QA,
            dspy_module=DSPyModuleType.REACT,
            train_size=500,
            max_steps=10,
            discount_gamma=0.9,
            tools=[ToolType.WEB_SEARCH],
            dataset=DatasetSplit(
                hf_dataset_id="axon-rl/HotpotQA",
                total_train_size=90_400,
                subsample_size=500,
                subsample_strategy="uniform",
                eval_dataset_id="axon-rl/search-eval",
                eval_split="hotpotqa",
                eval_size=512,
            ),
        )


class EvalProtocol(BaseModel):
    """Evaluation protocol matching GEM paper settings."""

    model_config = {"frozen": True}

    max_response_tokens: int = Field(default=4096, ge=1)
    temperature_train: float = Field(default=1.0, ge=0.0)
    temperature_eval: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, description="-1 means disabled")
    rollouts_per_task: int = Field(default=5, ge=1)
    max_eval_workers: int = Field(
        default=32, ge=1, description="Parallel workers for held-out eval"
    )
    tost_margin: float = Field(default=0.05, gt=0.0, le=1.0, description="TOST equivalence margin")
    tost_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    bootstrap_iterations: int = Field(default=1000, ge=100)


class TaskModelConfig(BaseModel):
    """Task model (the LLM being evaluated)."""

    model_config = {"frozen": True}

    name: str = Field(description="Human-readable name, e.g. 'Qwen3-1.7B'")
    model_id: str = Field(description="LiteLLM routing ID, e.g. 'ollama/qwen3-1.7b-base'")
    provider: Literal["ollama", "bedrock", "sagemaker"]
    parameter_count: str = Field(description="For reporting, e.g. '1.7B'")


class ReflectionModelConfig(BaseModel):
    """Reflection model for GEPA mutation proposals."""

    model_config = {"frozen": True}

    model_id: str = Field(description="LiteLLM routing ID")
    provider: Literal["bedrock"] = "bedrock"
    temperature: float = Field(default=1.0, ge=0.0)
    max_tokens: int = Field(default=4096, ge=1)


class GEPABudgetConfig(BaseModel):
    """GEPA optimizer budget configuration.

    Maps directly to ``dspy.GEPA(auto=mode, reflection_minibatch_size=...)``.
    DSPy's ``auto`` parameter sets the number of candidate prompts to explore
    (light=6, medium=12, heavy=18) and derives the metric-call budget
    internally from trainset/valset sizes.
    """

    model_config = {"frozen": True}

    mode: Literal["light", "medium", "heavy"]
    tasks_per_minibatch: int = Field(
        ge=1, description="dspy.GEPA reflection_minibatch_size — paper uses 3"
    )


class SeedConfig(BaseModel):
    """Seed configuration for reproducibility at three levels."""

    model_config = {"frozen": True}

    data_seed: int = Field(
        default=42, description="Fixed across all experiments for identical splits"
    )
    replication_seeds: tuple[int, ...] = Field(
        default=(42, 123, 456),
        description="One per replication, varied across runs",
    )


class RLTrainingDetails(BaseModel):
    """RL training hyperparameters from GEM paper (reference only)."""

    model_config = {"frozen": True}

    learning_rate: float = 1e-6
    optimizer: str = "AdamW"
    adam_betas: tuple[float, float] = (0.9, 0.95)
    gradient_clip: float = 1.0
    policy_clip: float = 0.2
    kl_coefficient: float = 0.0
    inner_epochs: int = 2
    gae_lambda: float = 0.95
    training_steps: int = 500
    infrastructure: str = "8x A100 GPUs, ~1 day"


class RLBaselineResult(BaseModel):
    """Published RL baseline result from GEM paper (reference data)."""

    model_config = {"frozen": True}

    algorithm: str = Field(description="e.g. 'GRPO', 'PPO', 'REINFORCE+ReBN'")
    environment: str = Field(description="GEM environment ID")
    model_name: str = Field(description="e.g. 'Qwen3-4B'")
    model_size: str = Field(description="e.g. '4B'")
    success_rate: float = Field(ge=0.0, le=1.0)
    tool_augmented: bool = Field(default=False, description="Whether the agent had tool access")
    training_condition: Literal["single", "mixed"] = Field(
        default="single",
        description="Whether RL training used a single environment or a mixed dataset",
    )
    metric_name: str = Field(default="pass@1", description="Published evaluation metric")
    source: str = Field(default="GEM (Liu et al., 2025)")
    source_detail: str = Field(
        default="", description="Exact table/cell provenance in the GEM paper"
    )
    estimated: bool = Field(
        default=False, description="True if value is visually estimated, not tabulated"
    )
    training_details: RLTrainingDetails = Field(default_factory=RLTrainingDetails)


class CostBudget(BaseModel):
    """Cost budget estimation from the proposal."""

    model_config = {"frozen": True}

    estimated_tokens_per_experiment: int = Field(default=94_000_000, ge=0)
    total_budget_usd: float = Field(default=2500.0, ge=0.0)
    buffer_pct: float = Field(default=0.25, ge=0.0, le=1.0)

    @property
    def effective_budget_usd(self) -> float:
        return self.total_budget_usd * (1 + self.buffer_pct)


class FitnessOverride(BaseModel):
    """Per-environment fitness parameter overrides."""

    model_config = {"frozen": True}

    gamma: float | None = Field(default=None, ge=0.0, le=1.0)
    lambda_: float | None = Field(default=None, alias="lambda", ge=0.0)
    loop_penalty_weight: float | None = Field(default=None, ge=0.0)
    step_efficiency_weight: float | None = Field(default=None, ge=0.0)
    call_efficiency_weight: float | None = Field(default=None, ge=0.0)
    max_steps: int | None = Field(default=None, ge=1)
    call_budget_per_step: int | None = Field(default=None, ge=1)


class PromptBaselineConfig(BaseModel):
    """Prompt-optimization baseline we run ourselves."""

    model_config = {"frozen": True}

    name: Literal["zero-shot", "miprov2"]
    description: str
    optimizer: Literal["MIPROv2"] | None = None
    requires_optimization: bool

    @model_validator(mode="after")
    def _validate_baseline_shape(self) -> Self:
        if self.name == "zero-shot":
            if self.optimizer is not None or self.requires_optimization:
                msg = "zero-shot baseline must not declare an optimizer and cannot require optimization"
                raise ValueError(msg)
            return self

        if self.optimizer != "MIPROv2" or not self.requires_optimization:
            msg = "miprov2 baseline must use optimizer='MIPROv2' and require optimization"
            raise ValueError(msg)
        return self

    @classmethod
    def zero_shot(cls) -> PromptBaselineConfig:
        return cls(
            name="zero-shot",
            description="Unoptimized DSPy agent with hand-written instructions",
            optimizer=None,
            requires_optimization=False,
        )

    @classmethod
    def miprov2(cls) -> PromptBaselineConfig:
        return cls(
            name="miprov2",
            description="DSPy's Bayesian prompt optimizer baseline",
            optimizer="MIPROv2",
            requires_optimization=True,
        )


class ComparisonProtocol(BaseModel):
    """Formal comparison settings for GEPA versus baselines."""

    model_config = {"frozen": True}

    primary_metric: Literal["success_rate"] = "success_rate"
    equivalence_test: Literal["TOST"] = "TOST"
    primary_rl_baseline: str = Field(default="REINFORCE+ReBN")
    prompt_baselines: list[PromptBaselineConfig] = Field(
        default_factory=lambda: [PromptBaselineConfig.zero_shot(), PromptBaselineConfig.miprov2()],
        min_length=1,
    )


# ── Top-level ExperimentConfig ───────────────────────────────────


class ExperimentConfig(BaseModel):
    """Complete experiment specification for reproducible research.

    Each config targets a single environment. One YAML per experiment:
        config = ExperimentConfig.from_yaml("experiments/orz57k-tool/config.yaml")
    """

    model_config = {"frozen": True, "populate_by_name": True}

    name: str
    description: str = ""
    environment: EnvironmentConfig
    # Seed system prompt that initializes GEPA and is used by baselines.
    # Required in every experiment YAML because the correct prompt depends on
    # the environment (math vs. QA) and on whether tools are enabled — there
    # is no single sensible default.
    seed_prompt: str = Field(min_length=1)
    eval_protocol: EvalProtocol = Field(default_factory=EvalProtocol)
    task_models: list[TaskModelConfig] = Field(min_length=1)
    reflection_model: ReflectionModelConfig
    gepa_budget: GEPABudgetConfig
    seeds: SeedConfig = Field(default_factory=SeedConfig)
    num_replications: int = Field(default=3, ge=1)
    rl_baselines: list[RLBaselineResult] = Field(default_factory=list)
    comparison_protocol: ComparisonProtocol | None = None
    cost_budget: CostBudget = Field(default_factory=CostBudget)
    fitness_override: FitnessOverride = Field(default_factory=FitnessOverride)

    @model_validator(mode="after")
    def _seeds_match_replications(self) -> Self:
        if len(self.seeds.replication_seeds) != self.num_replications:
            msg = (
                f"len(replication_seeds)={len(self.seeds.replication_seeds)} "
                f"!= num_replications={self.num_replications}"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate an experiment config from a YAML file."""
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize experiment config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
