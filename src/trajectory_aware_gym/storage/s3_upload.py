"""S3 artifact upload/download for experiment run results.

Enables collaborators to share experiment results via a shared S3 bucket
without needing access to each other's local machines.

S3 key format: ``{prefix}{experiment_run_id}/{filename}``
Example: ``experiments/orz57k-ollama-qwen3-jinyu-seed42-20260415T1430Z/config_snapshot.yaml``

Artifacts are immutable: if a key already exists, upload is skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_s3_client(client_config: dict[str, str] | None = None) -> Any:
    """Create a boto3 S3 client.

    Raises ``ImportError`` if boto3 is not installed, or
    ``ValueError`` if credentials are insufficient.
    """
    import boto3

    config = client_config or {}
    return boto3.client("s3", **config)


def upload_artifact_bundle(
    experiment_run_id: str,
    artifacts: dict[str, Path],
    *,
    bucket: str,
    prefix: str = "experiments/",
    client_config: dict[str, str] | None = None,
) -> list[str]:
    """Upload a set of artifact files to S3 under the experiment run prefix.

    Parameters
    ----------
    experiment_run_id:
        The experiment run ID used as the S3 sub-prefix.
    artifacts:
        Mapping of ``{filename: local_path}`` for files to upload.
        E.g. ``{"config_snapshot.yaml": Path("/tmp/config.yaml")}``.
    bucket:
        S3 bucket name.
    prefix:
        S3 key prefix (default: ``"experiments/"``).
    client_config:
        Optional boto3 client kwargs (region, credentials).

    Returns
    -------
    list[str]
        List of S3 keys that were actually uploaded (skipped keys not included).
    """
    result = upload_artifact_bundle_detailed(
        experiment_run_id,
        artifacts,
        bucket=bucket,
        prefix=prefix,
        client_config=client_config,
    )
    return result["uploaded_keys"]


def upload_artifact_bundle_detailed(
    experiment_run_id: str,
    artifacts: dict[str, Path],
    *,
    bucket: str,
    prefix: str = "experiments/",
    client_config: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Upload artifact files and return uploaded/skipped/failed detail."""
    result: dict[str, Any] = {
        "uploaded_keys": [],
        "skipped_existing_keys": [],
        "failed_keys": [],
    }
    try:
        client = _get_s3_client(client_config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to initialize S3 client for experiment_run_id=%s",
            experiment_run_id,
            exc_info=True,
        )
        for filename in artifacts:
            key = f"{prefix}{experiment_run_id}/{filename}"
            result["failed_keys"].append(
                {
                    "filename": filename,
                    "key": key,
                    "error": repr(exc),
                }
            )
        return result

    for filename, local_path in artifacts.items():
        key = f"{prefix}{experiment_run_id}/{filename}"

        if not local_path.exists():
            result["failed_keys"].append(
                {
                    "filename": filename,
                    "key": key,
                    "error": f"Local artifact not found: {local_path}",
                }
            )
            continue

        try:
            if _key_exists(client, bucket, key):
                logger.warning("S3 key already exists, skipping: s3://%s/%s", bucket, key)
                result["skipped_existing_keys"].append(key)
                continue

            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=local_path.read_bytes(),
            )
            result["uploaded_keys"].append(key)
            logger.info("Uploaded: s3://%s/%s", bucket, key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to upload s3://%s/%s", bucket, key, exc_info=True)
            result["failed_keys"].append(
                {
                    "filename": filename,
                    "key": key,
                    "error": repr(exc),
                }
            )

    return result


def list_remote_runs(
    *,
    bucket: str,
    prefix: str = "experiments/",
    client_config: dict[str, str] | None = None,
) -> list[str]:
    """List experiment_run_id prefixes found in S3.

    Returns a deduplicated, sorted list of run IDs extracted from
    the S3 key hierarchy.
    """
    client = _get_s3_client(client_config)
    paginator = client.get_paginator("list_objects_v2")
    run_ids: set[str] = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            # cp["Prefix"] is e.g. "experiments/run-id-here/"
            run_id = cp["Prefix"].removeprefix(prefix).rstrip("/")
            if run_id:
                run_ids.add(run_id)

    return sorted(run_ids)


def download_artifact(
    experiment_run_id: str,
    filename: str,
    dest_dir: Path,
    *,
    bucket: str,
    prefix: str = "experiments/",
    client_config: dict[str, str] | None = None,
) -> Path:
    """Download a single artifact file from S3.

    Parameters
    ----------
    experiment_run_id:
        The experiment run ID.
    filename:
        Name of the artifact file to download.
    dest_dir:
        Local directory to save the file to.
    bucket:
        S3 bucket name.
    prefix:
        S3 key prefix.
    client_config:
        Optional boto3 client kwargs.

    Returns
    -------
    Path
        Local path where the file was saved.

    Raises
    ------
    FileNotFoundError
        If the S3 key does not exist.
    """
    client = _get_s3_client(client_config)
    key = f"{prefix}{experiment_run_id}/{filename}"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    try:
        response = client.get_object(Bucket=bucket, Key=key)
        dest_path.write_bytes(response["Body"].read())
    except client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"S3 key not found: s3://{bucket}/{key}") from None

    return dest_path


def _key_exists(client: Any, bucket: str, key: str) -> bool:
    """Check if an S3 key exists (HEAD request)."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError:
        return False
