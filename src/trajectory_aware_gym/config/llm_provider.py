"""LLM provider factory for task and reflection models."""

from typing import Literal, TypeAlias

import dspy  # type: ignore[import-untyped]

from trajectory_aware_gym.config import settings

# All supported task models
TaskModelName: TypeAlias = Literal[  # noqa: UP040
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3-base:1.7b",
    "qwen3-base:4b",
    "llama:1b",
    "llama:3b",
    "llama:8b",
    # "gpt-oss:20b",   # placeholder
    # "gpt-oss:120b",  # placeholder
]


def get_task_model_id(model: TaskModelName = "qwen3:1.7b") -> str | None:
    """Resolve a task model alias to the underlying LiteLLM model ID."""
    match model:
        case "qwen3:1.7b":
            return settings.ollama.task_model_1_7b
        case "qwen3:4b":
            return settings.ollama.task_model_4b
        case "qwen3-base:1.7b":
            return f"sagemaker/{settings.sagemaker.endpoint_1_7b}"
        case "qwen3-base:4b":
            return f"sagemaker/{settings.sagemaker.endpoint_4b}"
        case "llama:1b":
            return f"bedrock/{settings.aws.bedrock_llama_1b}"
        case "llama:3b":
            return f"bedrock/{settings.aws.bedrock_llama_3b}"
        case "llama:8b":
            return f"bedrock/{settings.aws.bedrock_llama_8b}"


def get_task_lm(
    model: TaskModelName = "qwen3:1.7b",
    mode: Literal["train", "eval"] = "train",
) -> dspy.LM | None:
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
    model_id = get_task_model_id(model)
    if model_id is None:
        return None
    kwargs = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if model_id.startswith(("bedrock/", "sagemaker/")):
        aws_region = getattr(settings.aws, "region", None)
        if aws_region is not None:
            kwargs["aws_region_name"] = aws_region
    if model_id.startswith("ollama_chat/"):
        kwargs["api_base"] = settings.ollama.api_base
    return dspy.LM(**kwargs)


def get_reflection_lm(
    model_id: str | None = None,
    *,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> dspy.LM:
    """Get the GEPA reflection model LM.

    Defaults to the infrastructure-level settings value, but callers may pass an
    explicit experiment-scoped model ID and generation settings.
    """
    kwargs = {
        "model": f"bedrock/{model_id or settings.gepa.reflection_model}",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    aws_region = getattr(settings.aws, "region", None)
    if aws_region is not None:
        kwargs["aws_region_name"] = aws_region
    return dspy.LM(**kwargs)
