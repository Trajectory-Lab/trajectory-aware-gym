"""LLM provider factory for task and reflection models."""

from typing import Literal, TypeAlias

import dspy  # type: ignore[import-untyped]

from trajectory_aware_gym.config import settings

# All supported task models
TaskModelName: TypeAlias = Literal[  # noqa: UP040
    "qwen3:1.7b",
    "qwen3:4b",
    "llama:1b",
    "llama:3b",
    "llama:8b",
    # "gpt-oss:20b",   # placeholder
    # "gpt-oss:120b",  # placeholder
]


def get_task_lm(
    model: TaskModelName = "qwen3:1.7b",
    mode: Literal["train", "eval"] = "train",
) -> dspy.LM:
    """Get a task model LM instance.

    Routes to the correct provider (Ollama for local models,
    Bedrock for AWS models) based on the model name.

    Args:
        model: Which task model to use.
        mode: "train" (temp=1.0) or "eval" (temp=0.0) per GEM protocol.
    """
    temperature = (
        settings.gem.temperature_train if mode == "train" else settings.gem.temperature_eval
    )

    match model:
        # Local models (Ollama)
        case "qwen3:1.7b":
            return dspy.LM(
                model=settings.ollama.task_model_1_7b,
                api_base=settings.ollama.api_base,
                temperature=temperature,
                max_tokens=4096,
            )
        case "qwen3:4b":
            return dspy.LM(
                model=settings.ollama.task_model_4b,
                api_base=settings.ollama.api_base,
                temperature=temperature,
                max_tokens=4096,
            )
        # AWS Bedrock models
        case "llama:1b":
            return dspy.LM(
                model=f"bedrock/{settings.aws.bedrock_llama_1b}",
                temperature=temperature,
                max_tokens=4096,
            )
        case "llama:3b":
            return dspy.LM(
                model=f"bedrock/{settings.aws.bedrock_llama_3b}",
                temperature=temperature,
                max_tokens=4096,
            )
        case "llama:8b":
            return dspy.LM(
                model=f"bedrock/{settings.aws.bedrock_llama_8b}",
                temperature=temperature,
                max_tokens=4096,
            )


def get_reflection_lm() -> dspy.LM:
    """Get the GEPA reflection model LM (Claude Sonnet 4.5 via Bedrock)."""
    return dspy.LM(
        model=f"bedrock/{settings.gepa.reflection_model}",
        temperature=1.0,
        max_tokens=4096,
    )
