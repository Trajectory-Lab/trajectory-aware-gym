"""Ollama configuration for local task models."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaConfig(BaseSettings):
    """Configuration for local Ollama task models (Qwen3 1.7B/4B)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    ollama_api_base: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    local_task_model_1_7b: str = Field(
        default="ollama_chat/qwen3:1.7b",
        description="LiteLLM model string for Qwen3 1.7B",
    )
    local_task_model_4b: str = Field(
        default="ollama_chat/qwen3:4b",
        description="LiteLLM model string for Qwen3 4B",
    )