"""AWS Configuration for Bedrock and S3 services."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AWSConfig(BaseSettings):
    """AWS configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: str = Field(default="", description="AWS access key ID")
    aws_secret_access_key: str = Field(default="", description="AWS secret access key")
    aws_session_token: str = Field(default="", description="AWS session token (optional)")

    bedrock_claude_sonnet_4_5: str = Field(
        default="anthropic.claude-sonnet-4-5-v2:0",
        description="Bedrock model ID for Claude Sonnet 4.5",
    )
    bedrock_llama_1b: str = Field(
        default="us.meta.llama3-2-1b-instruct-v1:0",
        description="Bedrock model ID for Llama 3.2 1B",
    )
    bedrock_llama_3b: str = Field(
        default="us.meta.llama3-2-3b-instruct-v1:0",
        description="Bedrock model ID for Llama 3.2 3B",
    )
    bedrock_llama_8b: str = Field(
        default="us.meta.llama3-1-8b-instruct-v1:0",
        description="Bedrock model ID for Llama 3.1 8B",
    )

    s3_bucket: str = Field(
        default="trajectory-aware-gym-results",
        description="S3 bucket for storing results",
    )
    s3_prefix: str = Field(
        default="experiments/",
        description="S3 prefix for experiment data",
    )

    def get_bedrock_client_config(self) -> dict:
        """Get configuration for boto3 Bedrock client."""
        config = {
            "region_name": self.aws_region,
        }

        if self.aws_access_key_id:
            config["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            config["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            config["aws_session_token"] = self.aws_session_token

        return config

    def get_s3_client_config(self) -> dict:
        """Get configuration for boto3 S3 client."""
        return self.get_bedrock_client_config()
