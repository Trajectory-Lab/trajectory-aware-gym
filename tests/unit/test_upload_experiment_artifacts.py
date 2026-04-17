"""Tests for the local-first S3 upload script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_upload_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "upload_experiment_artifacts.py"
    spec = importlib.util.spec_from_file_location("upload_experiment_artifacts_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_completed_replication(replication_dir: Path, experiment_run_id: str = "run-123") -> None:
    replication_dir.mkdir(parents=True)
    (replication_dir / "run_metadata.json").write_text(
        json.dumps({"status": "completed", "experiment_run_id": experiment_run_id}),
        encoding="utf-8",
    )
    (replication_dir / "config_snapshot.yaml").write_text("config: test\n", encoding="utf-8")
    (replication_dir / "cost_summary.json").write_text("{}", encoding="utf-8")
    gepa_logs = replication_dir / "gepa_logs"
    gepa_logs.mkdir()
    (gepa_logs / "trace.log").write_text("trace\n", encoding="utf-8")


def test_upload_replication_dir_writes_manifest_and_uses_relative_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    upload_script = _load_upload_script()
    replication_dir = tmp_path / "quick-test" / "20260416T120000Z" / "model" / "replication_42"
    _make_completed_replication(replication_dir)
    captured: dict[str, object] = {}

    def fake_upload(experiment_run_id, artifacts, **kwargs):
        captured["experiment_run_id"] = experiment_run_id
        captured["artifact_names"] = sorted(artifacts.keys())
        return {
            "uploaded_keys": ["experiments/run-123/config_snapshot.yaml"],
            "skipped_existing_keys": [],
            "failed_keys": [],
        }

    monkeypatch.setattr(upload_script, "upload_artifact_bundle_detailed", fake_upload)

    result = upload_script._upload_replication_dir(
        replication_dir,
        bucket="test-bucket",
        prefix="experiments/",
    )

    assert captured["experiment_run_id"] == "run-123"
    assert captured["artifact_names"] == [
        "config_snapshot.yaml",
        "cost_summary.json",
        "gepa_logs/trace.log",
        "run_metadata.json",
    ]
    assert result["failed_keys"] == []

    manifest = json.loads((replication_dir / "upload_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_run_id"] == "run-123"
    assert manifest["uploaded_keys"] == ["experiments/run-123/config_snapshot.yaml"]
    assert manifest["failed_keys"] == []


def test_upload_replication_dir_skips_symlinked_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    upload_script = _load_upload_script()
    replication_dir = tmp_path / "quick-test" / "20260416T120000Z" / "model" / "replication_42"
    _make_completed_replication(replication_dir)
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret\n", encoding="utf-8")
    (replication_dir / "leak.txt").symlink_to(outside_file)
    captured: dict[str, object] = {}

    def fake_upload(experiment_run_id, artifacts, **kwargs):
        captured["artifact_names"] = sorted(artifacts.keys())
        return {
            "uploaded_keys": [],
            "skipped_existing_keys": [],
            "failed_keys": [],
        }

    monkeypatch.setattr(upload_script, "upload_artifact_bundle_detailed", fake_upload)

    upload_script._upload_replication_dir(
        replication_dir,
        bucket="test-bucket",
        prefix="experiments/",
    )

    assert "leak.txt" not in captured["artifact_names"]


def test_upload_replication_dir_skips_symlinked_directories(
    monkeypatch,
    tmp_path: Path,
) -> None:
    upload_script = _load_upload_script()
    replication_dir = tmp_path / "quick-test" / "20260416T120000Z" / "model" / "replication_42"
    _make_completed_replication(replication_dir)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "nested.txt").write_text("secret\n", encoding="utf-8")
    (replication_dir / "linked-dir").symlink_to(outside_dir, target_is_directory=True)
    captured: dict[str, object] = {}

    def fake_upload(experiment_run_id, artifacts, **kwargs):
        captured["artifact_names"] = sorted(artifacts.keys())
        return {
            "uploaded_keys": [],
            "skipped_existing_keys": [],
            "failed_keys": [],
        }

    monkeypatch.setattr(upload_script, "upload_artifact_bundle_detailed", fake_upload)

    upload_script._upload_replication_dir(
        replication_dir,
        bucket="test-bucket",
        prefix="experiments/",
    )

    assert all(
        not str(artifact_name).startswith("linked-dir/")
        for artifact_name in captured["artifact_names"]
    )


def test_upload_replication_dir_records_skips_and_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    upload_script = _load_upload_script()
    replication_dir = tmp_path / "quick-test" / "20260416T120000Z" / "model" / "replication_42"
    _make_completed_replication(replication_dir)

    monkeypatch.setattr(
        upload_script,
        "upload_artifact_bundle_detailed",
        lambda experiment_run_id, artifacts, **kwargs: {
            "uploaded_keys": [],
            "skipped_existing_keys": ["experiments/run-123/config_snapshot.yaml"],
            "failed_keys": [
                {
                    "filename": "cost_summary.json",
                    "key": "experiments/run-123/cost_summary.json",
                    "error": "ConnectionError('boom')",
                }
            ],
        },
    )

    result = upload_script._upload_replication_dir(
        replication_dir,
        bucket="test-bucket",
        prefix="experiments/",
    )

    assert result["skipped_existing_keys"] == ["experiments/run-123/config_snapshot.yaml"]
    assert result["failed_keys"][0]["filename"] == "cost_summary.json"

    manifest = json.loads((replication_dir / "upload_manifest.json").read_text(encoding="utf-8"))
    assert manifest["skipped_existing_keys"] == ["experiments/run-123/config_snapshot.yaml"]
    assert manifest["failed_keys"][0]["error"] == "ConnectionError('boom')"


def test_upload_replication_dir_writes_manifest_for_bootstrap_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    upload_script = _load_upload_script()
    replication_dir = tmp_path / "quick-test" / "20260416T120000Z" / "model" / "replication_42"
    _make_completed_replication(replication_dir)

    def fake_upload(experiment_run_id, artifacts, **kwargs):
        return {
            "uploaded_keys": [],
            "skipped_existing_keys": [],
            "failed_keys": [
                {
                    "filename": filename,
                    "key": f"experiments/{experiment_run_id}/{filename}",
                    "error": "RuntimeError('bad creds')",
                }
                for filename in sorted(artifacts)
            ],
        }

    monkeypatch.setattr(upload_script, "upload_artifact_bundle_detailed", fake_upload)

    result = upload_script._upload_replication_dir(
        replication_dir,
        bucket="test-bucket",
        prefix="experiments/",
    )

    assert result["uploaded_keys"] == []
    assert len(result["failed_keys"]) == 4

    manifest = json.loads((replication_dir / "upload_manifest.json").read_text(encoding="utf-8"))
    assert manifest["uploaded_keys"] == []
    assert len(manifest["failed_keys"]) == 4
    assert manifest["failed_keys"][0]["error"] == "RuntimeError('bad creds')"
