"""Tests for AWS client configuration branches."""

import pytest

from trajectory_aware_gym.config.core import AWSModel


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param(
            {
                "region": "us-east-1",
                "bedrock_claude_sonnet_4_5": "anthropic.claude-sonnet-4-5-v2:0",
                "bedrock_llama_1b": "us.meta.llama3-2-1b-instruct-v1:0",
                "bedrock_llama_3b": "us.meta.llama3-2-3b-instruct-v1:0",
                "bedrock_llama_8b": "us.meta.llama3-1-8b-instruct-v1:0",
                "bedrock_gemma_4b": "google.gemma-3-4b-it",
                "bedrock_mistral_7b": "mistral.mistral-7b-instruct-v0:2",
                "bedrock_nemotron_9b": "nvidia.nemotron-nano-9b-v2",
                "s3_bucket": "trajectory-aware-gym-results",
                "s3_prefix": "experiments/",
            },
            {"region_name": "us-east-1"},
            id="defaults-exclude-empty-credentials",
        ),
        pytest.param(
            {
                "region": "us-west-2",
                "access_key_id": "AKIA_TEST",
                "secret_access_key": "secret",  # pragma: allowlist secret
                "session_token": "session",
                "bedrock_claude_sonnet_4_5": "anthropic.claude-sonnet-4-5-v2:0",
                "bedrock_llama_1b": "us.meta.llama3-2-1b-instruct-v1:0",
                "bedrock_llama_3b": "us.meta.llama3-2-3b-instruct-v1:0",
                "bedrock_llama_8b": "us.meta.llama3-1-8b-instruct-v1:0",
                "bedrock_gemma_4b": "google.gemma-3-4b-it",
                "bedrock_mistral_7b": "mistral.mistral-7b-instruct-v0:2",
                "bedrock_nemotron_9b": "nvidia.nemotron-nano-9b-v2",
                "s3_bucket": "trajectory-aware-gym-results",
                "s3_prefix": "experiments/",
            },
            {
                "region_name": "us-west-2",
                "aws_access_key_id": "AKIA_TEST",
                "aws_secret_access_key": "secret",  # pragma: allowlist secret
                "aws_session_token": "session",
            },
            id="all-credentials-included",
        ),
        pytest.param(
            {
                "region": "eu-central-1",
                "access_key_id": "AKIA_PARTIAL",
                "secret_access_key": "",
                "bedrock_claude_sonnet_4_5": "anthropic.claude-sonnet-4-5-v2:0",
                "bedrock_llama_1b": "us.meta.llama3-2-1b-instruct-v1:0",
                "bedrock_llama_3b": "us.meta.llama3-2-3b-instruct-v1:0",
                "bedrock_llama_8b": "us.meta.llama3-1-8b-instruct-v1:0",
                "bedrock_gemma_4b": "google.gemma-3-4b-it",
                "bedrock_mistral_7b": "mistral.mistral-7b-instruct-v0:2",
                "bedrock_nemotron_9b": "nvidia.nemotron-nano-9b-v2",
                "s3_bucket": "trajectory-aware-gym-results",
                "s3_prefix": "experiments/",
            },
            {
                "region_name": "eu-central-1",
                "aws_access_key_id": "AKIA_PARTIAL",
            },
            id="partial-credentials-key-only",
        ),
        pytest.param(
            {
                "region": "ap-southeast-1",
                "session_token": "tok",
                "bedrock_claude_sonnet_4_5": "anthropic.claude-sonnet-4-5-v2:0",
                "bedrock_llama_1b": "us.meta.llama3-2-1b-instruct-v1:0",
                "bedrock_llama_3b": "us.meta.llama3-2-3b-instruct-v1:0",
                "bedrock_llama_8b": "us.meta.llama3-1-8b-instruct-v1:0",
                "bedrock_gemma_4b": "google.gemma-3-4b-it",
                "bedrock_mistral_7b": "mistral.mistral-7b-instruct-v0:2",
                "bedrock_nemotron_9b": "nvidia.nemotron-nano-9b-v2",
                "s3_bucket": "trajectory-aware-gym-results",
                "s3_prefix": "experiments/",
            },
            {
                "region_name": "ap-southeast-1",
                "aws_session_token": "tok",
            },
            id="partial-credentials-session-token-only",
        ),
    ],
)
def test_bedrock_client_config(kwargs, expected):
    """Bedrock client config includes credentials only when non-empty."""
    config = AWSModel(**kwargs)
    assert config.get_bedrock_client_config() == expected


def test_s3_client_config_delegates_to_bedrock_payload():
    """S3 config reuses Bedrock client config construction."""
    config = AWSModel(
        region="eu-central-1",
        bedrock_claude_sonnet_4_5="anthropic.claude-sonnet-4-5-v2:0",
        bedrock_llama_1b="us.meta.llama3-2-1b-instruct-v1:0",
        bedrock_llama_3b="us.meta.llama3-2-3b-instruct-v1:0",
        bedrock_llama_8b="us.meta.llama3-1-8b-instruct-v1:0",
        bedrock_gemma_4b="google.gemma-3-4b-it",
        bedrock_mistral_7b="mistral.mistral-7b-instruct-v0:2",
        bedrock_nemotron_9b="nvidia.nemotron-nano-9b-v2",
        s3_bucket="trajectory-aware-gym-results",
        s3_prefix="experiments/",
    )
    assert config.get_s3_client_config() == {"region_name": "eu-central-1"}
