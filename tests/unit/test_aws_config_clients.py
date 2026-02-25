"""Tests for AWS client configuration branches."""

import pytest

from trajectory_aware_gym.config.aws_config import AWSConfig


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param(
            {"aws_region": "us-east-1"},
            {"region_name": "us-east-1"},
            id="defaults-exclude-empty-credentials",
        ),
        pytest.param(
            {
                "aws_region": "us-west-2",
                "aws_access_key_id": "AKIA_TEST",
                "aws_secret_access_key": "secret",  # pragma: allowlist secret
                "aws_session_token": "session",
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
                "aws_region": "eu-central-1",
                "aws_access_key_id": "AKIA_PARTIAL",
                "aws_secret_access_key": "",
            },
            {
                "region_name": "eu-central-1",
                "aws_access_key_id": "AKIA_PARTIAL",
            },
            id="partial-credentials-key-only",
        ),
        pytest.param(
            {
                "aws_region": "ap-southeast-1",
                "aws_session_token": "tok",
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
    config = AWSConfig(**kwargs, _env_file=None)
    assert config.get_bedrock_client_config() == expected


def test_s3_client_config_delegates_to_bedrock_payload():
    """S3 config reuses Bedrock client config construction."""
    config = AWSConfig(aws_region="eu-central-1", _env_file=None)
    assert config.get_s3_client_config() == {"region_name": "eu-central-1"}
