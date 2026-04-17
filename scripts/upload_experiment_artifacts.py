"""Upload completed local experiment artifacts to S3.

The experiment runner is intentionally local-first: results are written to
replication folders and the local trajectories DB during execution. This
script performs the optional post-run sync step without changing experiment
status or mutating result truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from trajectory_aware_gym.config import settings
from trajectory_aware_gym.storage.s3_upload import upload_artifact_bundle_detailed

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "upload_manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload completed experiment artifacts to S3")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--replication-dir",
        type=Path,
        help="Upload one completed replication directory",
    )
    scope.add_argument(
        "--run-dir",
        type=Path,
        help="Upload all completed replication directories under one run timestamp directory",
    )
    scope.add_argument(
        "--config-dir",
        type=Path,
        help="Upload all completed replication directories under one config directory",
    )
    parser.add_argument(
        "--bucket",
        default=settings.aws.s3_bucket,
        help="Destination S3 bucket (default: settings.aws.s3_bucket)",
    )
    parser.add_argument(
        "--prefix",
        default=settings.aws.s3_prefix,
        help="Destination S3 prefix (default: settings.aws.s3_prefix)",
    )
    return parser.parse_args()


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _iter_replication_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("replication_*")
        if path.is_dir() and (path / "run_metadata.json").exists()
    )


def _discover_replication_dirs(args: argparse.Namespace) -> list[Path]:
    if args.replication_dir is not None:
        replication_dir = args.replication_dir.resolve()
        if not replication_dir.is_dir():
            raise ValueError(f"Replication directory not found: {replication_dir}")
        return [replication_dir]
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.is_dir():
            raise ValueError(f"Run directory not found: {run_dir}")
        return _iter_replication_dirs(run_dir)
    config_dir = args.config_dir.resolve()
    if not config_dir.is_dir():
        raise ValueError(f"Config directory not found: {config_dir}")
    return _iter_replication_dirs(config_dir)


def _should_upload(metadata: dict[str, Any]) -> bool:
    return metadata.get("status") == "completed" and isinstance(
        metadata.get("experiment_run_id"), str
    )


def _artifact_map(replication_dir: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for current_dir, dirnames, filenames in replication_dir.walk(follow_symlinks=False):
        dirnames[:] = sorted(
            dirname for dirname in dirnames if not (current_dir / dirname).is_symlink()
        )
        for filename in sorted(filenames):
            path = current_dir / filename
            if not path.is_file() or path.is_symlink() or path.name == _MANIFEST_NAME:
                continue
            rel_path = path.relative_to(replication_dir)
            if ".." in rel_path.parts:
                continue
            artifacts[str(rel_path)] = path
    return artifacts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            handle.write(serialized)
            tmp_path = Path(handle.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _write_manifest(
    replication_dir: Path,
    *,
    experiment_run_id: str,
    bucket: str,
    prefix: str,
    upload_result: dict[str, Any],
) -> None:
    manifest = {
        "experiment_run_id": experiment_run_id,
        "uploaded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "bucket": bucket,
        "prefix": prefix,
        "uploaded_keys": upload_result["uploaded_keys"],
        "skipped_existing_keys": upload_result["skipped_existing_keys"],
        "failed_keys": upload_result["failed_keys"],
    }
    _write_json(replication_dir / _MANIFEST_NAME, manifest)


def _upload_replication_dir(replication_dir: Path, *, bucket: str, prefix: str) -> dict[str, Any]:
    metadata = _load_json_dict(replication_dir / "run_metadata.json")
    if metadata is None:
        raise ValueError(f"Invalid run_metadata.json in {replication_dir}")
    if not _should_upload(metadata):
        raise ValueError(f"Replication is not a completed experiment run: {replication_dir}")

    experiment_run_id = str(metadata["experiment_run_id"])
    upload_result = upload_artifact_bundle_detailed(
        experiment_run_id,
        _artifact_map(replication_dir),
        bucket=bucket,
        prefix=prefix,
        client_config=settings.aws.get_s3_client_config(),
    )
    _write_manifest(
        replication_dir,
        experiment_run_id=experiment_run_id,
        bucket=bucket,
        prefix=prefix,
        upload_result=upload_result,
    )
    return {
        "replication_dir": replication_dir,
        "experiment_run_id": experiment_run_id,
        **upload_result,
    }


def main() -> None:
    logging.basicConfig(level=getattr(logging, settings.logging.level.upper(), logging.INFO))
    args = _parse_args()

    replication_dirs = _discover_replication_dirs(args)
    if not replication_dirs:
        raise ValueError("No replication directories found for the requested scope.")

    any_failures = False
    processed = 0
    skipped = 0
    for replication_dir in replication_dirs:
        metadata = _load_json_dict(replication_dir / "run_metadata.json")
        if metadata is None or not _should_upload(metadata):
            skipped += 1
            logger.info("Skipping non-completed replication: %s", replication_dir)
            continue

        result = _upload_replication_dir(
            replication_dir,
            bucket=args.bucket,
            prefix=args.prefix,
        )
        processed += 1
        any_failures = any_failures or bool(result["failed_keys"])
        logger.info(
            "Uploaded replication=%s run_id=%s uploaded=%d skipped_existing=%d failed=%d",
            replication_dir,
            result["experiment_run_id"],
            len(result["uploaded_keys"]),
            len(result["skipped_existing_keys"]),
            len(result["failed_keys"]),
        )

    print(
        json.dumps(
            {
                "processed_replications": processed,
                "skipped_replications": skipped,
                "bucket": args.bucket,
                "prefix": args.prefix,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if any_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
