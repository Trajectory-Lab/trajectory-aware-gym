"""Tests for S3 artifact upload/download module."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from trajectory_aware_gym.storage.s3_upload import (
    download_artifact,
    list_remote_runs,
    upload_artifact_bundle,
    upload_artifact_bundle_detailed,
)

BUCKET = "test-bucket"
PREFIX = "experiments/"


@pytest.fixture
def mock_s3(monkeypatch):
    """Provide a mocked boto3 S3 client.

    boto3 is imported lazily inside ``_get_s3_client``, so we inject
    a fake boto3 module into ``sys.modules`` before the function runs.
    """
    client = MagicMock()

    class FakeClientError(Exception):
        def __init__(self, response, operation_name):
            super().__init__(response, operation_name)
            self.response = response
            self.operation_name = operation_name

    # Custom exception types that behave like real exceptions
    client.exceptions.ClientError = FakeClientError
    client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

    # Default: head_object raises ClientError (key does not exist)
    client.head_object.side_effect = client.exceptions.ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )

    fake_boto3 = ModuleType("boto3")
    fake_boto3.client = MagicMock(return_value=client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    return client


class TestUploadArtifactBundle:
    def test_uploads_all_artifacts(self, mock_s3, tmp_path):
        """8.4: put_object called with correct key/body for each artifact."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test\n")
        summary_file = tmp_path / "summary.json"
        summary_file.write_text('{"accuracy": 0.85}')

        artifacts = {
            "config_snapshot.yaml": config_file,
            "run_summary.json": summary_file,
        }

        result = upload_artifact_bundle(
            "run-001",
            artifacts,
            bucket=BUCKET,
            prefix=PREFIX,
        )

        assert len(result) == 2
        assert f"{PREFIX}run-001/config_snapshot.yaml" in result
        assert f"{PREFIX}run-001/run_summary.json" in result
        assert mock_s3.put_object.call_count == 2

        # Verify the actual call args
        calls = {
            call.kwargs["Key"]: call.kwargs["Body"] for call in mock_s3.put_object.call_args_list
        }
        assert calls[f"{PREFIX}run-001/config_snapshot.yaml"] == b"model: test\n"
        assert calls[f"{PREFIX}run-001/run_summary.json"] == b'{"accuracy": 0.85}'

    def test_skips_existing_key(self, mock_s3, tmp_path):
        """8.5: existing key → skip upload, log warning."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test\n")

        # head_object succeeds → key exists
        mock_s3.head_object.side_effect = None
        mock_s3.head_object.return_value = {}

        result = upload_artifact_bundle(
            "run-001",
            {"config.yaml": config_file},
            bucket=BUCKET,
            prefix=PREFIX,
        )

        assert result == []
        mock_s3.put_object.assert_not_called()

    def test_partial_skip(self, mock_s3, tmp_path):
        """One artifact exists, one doesn't — only the new one gets uploaded."""
        existing_file = tmp_path / "existing.yaml"
        existing_file.write_text("old")
        new_file = tmp_path / "new.json"
        new_file.write_text("new")

        def head_side_effect(*, Bucket, Key):  # noqa: N803
            if "existing.yaml" in Key:
                return {}  # exists
            raise mock_s3.exceptions.ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )

        mock_s3.head_object.side_effect = head_side_effect

        result = upload_artifact_bundle(
            "run-002",
            {"existing.yaml": existing_file, "new.json": new_file},
            bucket=BUCKET,
            prefix=PREFIX,
        )

        assert len(result) == 1
        assert f"{PREFIX}run-002/new.json" in result
        mock_s3.put_object.assert_called_once()

    def test_invalid_run_id_raises(self, mock_s3, tmp_path):
        artifact = tmp_path / "config.yaml"
        artifact.write_text("model: test\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid experiment_run_id"):
            upload_artifact_bundle(
                "../run-005",
                {"config.yaml": artifact},
                bucket=BUCKET,
                prefix=PREFIX,
            )

        mock_s3.put_object.assert_not_called()

    def test_empty_artifacts(self, mock_s3):
        """No artifacts → no uploads, empty result."""
        result = upload_artifact_bundle("run-003", {}, bucket=BUCKET, prefix=PREFIX)
        assert result == []
        mock_s3.put_object.assert_not_called()

    def test_client_init_failure_returns_failed_keys(self, tmp_path):
        """Client/bootstrap failures are recorded per artifact instead of raising."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test\n", encoding="utf-8")
        summary_file = tmp_path / "summary.json"
        summary_file.write_text("{}", encoding="utf-8")

        artifacts = {
            "config_snapshot.yaml": config_file,
            "run_summary.json": summary_file,
        }

        with patch(
            "trajectory_aware_gym.storage.s3_upload._get_s3_client",
            side_effect=RuntimeError("bad creds"),
        ):
            result = upload_artifact_bundle_detailed(
                "run-004",
                artifacts,
                bucket=BUCKET,
                prefix=PREFIX,
            )

        assert result["uploaded_keys"] == []
        assert result["skipped_existing_keys"] == []
        assert result["failed_keys"] == [
            {
                "filename": "config_snapshot.yaml",
                "key": f"{PREFIX}run-004/config_snapshot.yaml",
                "error": "RuntimeError('bad creds')",
            },
            {
                "filename": "run_summary.json",
                "key": f"{PREFIX}run-004/run_summary.json",
                "error": "RuntimeError('bad creds')",
            },
        ]

    def test_head_errors_other_than_missing_are_not_treated_as_absent(self, mock_s3, tmp_path):
        artifact = tmp_path / "config.yaml"
        artifact.write_text("model: test\n", encoding="utf-8")
        mock_s3.head_object.side_effect = mock_s3.exceptions.ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}},
            "HeadObject",
        )

        result = upload_artifact_bundle_detailed(
            "run-006",
            {"config.yaml": artifact},
            bucket=BUCKET,
            prefix=PREFIX,
        )

        assert result["uploaded_keys"] == []
        assert result["skipped_existing_keys"] == []
        assert len(result["failed_keys"]) == 1
        assert result["failed_keys"][0]["filename"] == "config.yaml"
        assert result["failed_keys"][0]["key"] == f"{PREFIX}run-006/config.yaml"
        assert "403" in result["failed_keys"][0]["error"]
        mock_s3.put_object.assert_not_called()


class TestListRemoteRuns:
    def test_lists_run_ids(self, mock_s3):
        """8.7: list_remote_runs extracts run IDs from S3 prefixes."""
        paginator = MagicMock()
        mock_s3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "experiments/run-a/"},
                    {"Prefix": "experiments/run-b/"},
                ]
            },
            {
                "CommonPrefixes": [
                    {"Prefix": "experiments/run-c/"},
                ]
            },
        ]

        result = list_remote_runs(bucket=BUCKET, prefix=PREFIX)
        assert result == ["run-a", "run-b", "run-c"]
        mock_s3.get_paginator.assert_called_once_with("list_objects_v2")

    def test_empty_bucket(self, mock_s3):
        paginator = MagicMock()
        mock_s3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]

        result = list_remote_runs(bucket=BUCKET, prefix=PREFIX)
        assert result == []


class TestDownloadArtifact:
    def test_downloads_artifact(self, mock_s3, tmp_path):
        """8.8: download_artifact writes file to dest_dir."""
        body = MagicMock()
        body.read.return_value = b"content here"
        mock_s3.get_object.return_value = {"Body": body}

        dest = tmp_path / "downloads"
        result = download_artifact(
            "run-001",
            "config.yaml",
            dest,
            bucket=BUCKET,
            prefix=PREFIX,
        )

        assert result == dest / "config.yaml"
        assert result.read_text() == "content here"
        mock_s3.get_object.assert_called_once_with(
            Bucket=BUCKET,
            Key=f"{PREFIX}run-001/config.yaml",
        )

    def test_missing_key_raises(self, mock_s3, tmp_path):
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey({}, "GetObject")

        with pytest.raises(FileNotFoundError, match="S3 key not found"):
            download_artifact(
                "run-999",
                "missing.txt",
                tmp_path,
                bucket=BUCKET,
                prefix=PREFIX,
            )

    def test_missing_key_client_error_also_raises(self, mock_s3, tmp_path):
        mock_s3.get_object.side_effect = mock_s3.exceptions.ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "GetObject",
        )

        with pytest.raises(FileNotFoundError, match="S3 key not found"):
            download_artifact(
                "run-998",
                "missing.txt",
                tmp_path,
                bucket=BUCKET,
                prefix=PREFIX,
            )
