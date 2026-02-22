"""Tests for AWS client configuration branches."""

from trajectory_aware_gym.config.aws_config import AWSConfig


def test_bedrock_client_config_includes_optional_credentials():
    """Optional AWS credential fields are included when configured."""
    config = AWSConfig(
        aws_region="us-west-2",
        aws_access_key_id="AKIA_TEST",
        aws_secret_access_key="secret",
        aws_session_token="session",
        _env_file=None,
    )

    payload = config.get_bedrock_client_config()

    assert payload == {
        "region_name": "us-west-2",
        "aws_access_key_id": "AKIA_TEST",
        "aws_secret_access_key": "secret",
        "aws_session_token": "session",
    }


def test_s3_client_config_delegates_to_bedrock_payload():
    """S3 config reuses Bedrock client config construction."""
    config = AWSConfig(aws_region="eu-central-1", _env_file=None)

    assert config.get_s3_client_config() == {"region_name": "eu-central-1"}
