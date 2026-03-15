"""Tests for scripts/collect_raw_metrics.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "collect_raw_metrics.py"


class FakeMetricRow:
    """Minimal row object compatible with script writer/summary helpers."""

    def __init__(
        self,
        *,
        run_id: str,
        success: bool,
        episode_latency_seconds: float,
        llm_cost_usd: float | None,
        total_tokens: int | None,
    ):
        self.run_id = run_id
        self.success = success
        self.episode_latency_seconds = episode_latency_seconds
        self.llm_cost_usd = llm_cost_usd
        self.total_tokens = total_tokens

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        """Provide a stable, JSON-serializable row mapping."""
        assert mode == "json"
        return {
            "run_id": self.run_id,
            "success": self.success,
            "episode_latency_seconds": self.episode_latency_seconds,
            "llm_cost_usd": self.llm_cost_usd,
            "total_tokens": self.total_tokens,
        }


@pytest.fixture
def script_module():
    """Load the collector script as a Python module for unit testing."""
    spec = importlib.util.spec_from_file_location("collect_raw_metrics_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_with_custom_values(script_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Argument parser should map custom CLI values to expected types."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_raw_metrics.py",
            "--input-dir",
            "tmp/logs",
            "--glob",
            "trajectory_custom*.json",
            "--output-dir",
            "tmp/out",
            "--output-prefix",
            "run_x",
        ],
    )

    args = script_module.parse_args()

    assert args.input_dir == Path("tmp/logs")
    assert args.glob == "trajectory_custom*.json"
    assert args.output_dir == Path("tmp/out")
    assert args.output_prefix == "run_x"


def test_write_csv_handles_empty_rows(script_module, tmp_path: Path) -> None:
    """CSV writer should emit an empty file when no rows are available."""
    output_path = tmp_path / "empty.csv"
    script_module._write_csv(output_path, [])
    assert output_path.read_text(encoding="utf-8") == ""


def test_write_csv_and_jsonl_with_rows(script_module, tmp_path: Path) -> None:
    """Writers should emit valid CSV and JSONL outputs for provided rows."""
    rows = [
        FakeMetricRow(
            run_id="r1",
            success=True,
            episode_latency_seconds=1.25,
            llm_cost_usd=0.0012,
            total_tokens=120,
        ),
        FakeMetricRow(
            run_id="r2",
            success=False,
            episode_latency_seconds=2.5,
            llm_cost_usd=None,
            total_tokens=None,
        ),
    ]

    csv_path = tmp_path / "rows.csv"
    jsonl_path = tmp_path / "rows.jsonl"

    script_module._write_csv(csv_path, rows)
    script_module._write_jsonl(jsonl_path, rows)

    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 3
    assert csv_lines[0].startswith("run_id,success,episode_latency_seconds")
    assert "r1,True,1.25,0.0012,120" in csv_lines[1]

    jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 2
    assert json.loads(jsonl_lines[0])["run_id"] == "r1"
    assert json.loads(jsonl_lines[1])["run_id"] == "r2"


def test_summary_for_empty_and_populated_rows(script_module) -> None:
    """Summary helper should handle both empty and partially instrumented row sets."""
    assert script_module._summary([]) == {
        "episodes": 0,
        "successes": 0,
        "mean_episode_latency_seconds": None,
        "total_llm_cost_usd": None,
        "total_tokens": None,
    }

    rows = [
        FakeMetricRow(
            run_id="r1",
            success=True,
            episode_latency_seconds=1.0,
            llm_cost_usd=0.001,
            total_tokens=100,
        ),
        FakeMetricRow(
            run_id="r2",
            success=False,
            episode_latency_seconds=3.0,
            llm_cost_usd=None,
            total_tokens=None,
        ),
        FakeMetricRow(
            run_id="r3",
            success=True,
            episode_latency_seconds=2.0,
            llm_cost_usd=0.002,
            total_tokens=200,
        ),
    ]
    summary = script_module._summary(rows)

    assert summary == {
        "episodes": 3,
        "successes": 2,
        "mean_episode_latency_seconds": 2.0,
        "total_llm_cost_usd": 0.003,
        "total_tokens": 300,
    }


def test_main_collects_writes_and_prints(
    script_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Main should collect logs, write artifacts, and print output locations."""
    input_dir = tmp_path / "logs"
    output_dir = tmp_path / "results"
    input_dir.mkdir()

    (input_dir / "trajectory_b.json").write_text("{}", encoding="utf-8")
    (input_dir / "trajectory_a.json").write_text("{}", encoding="utf-8")

    captured_paths: list[Path] = []
    fake_rows = [
        FakeMetricRow(
            run_id="r1",
            success=True,
            episode_latency_seconds=1.0,
            llm_cost_usd=0.001,
            total_tokens=100,
        )
    ]

    def fake_collect(paths: list[Path]) -> list[FakeMetricRow]:
        captured_paths.extend(paths)
        return fake_rows

    monkeypatch.setattr(script_module, "collect_raw_metrics", fake_collect)
    monkeypatch.setattr(
        script_module,
        "parse_args",
        lambda: argparse.Namespace(
            input_dir=input_dir,
            glob="trajectory_*.json",
            output_dir=output_dir,
            output_prefix="phase3_raw_metrics",
        ),
    )

    script_module.main()

    assert [path.name for path in captured_paths] == ["trajectory_a.json", "trajectory_b.json"]

    csv_path = output_dir / "phase3_raw_metrics.csv"
    jsonl_path = output_dir / "phase3_raw_metrics.jsonl"
    summary_path = output_dir / "phase3_raw_metrics_summary.json"

    assert csv_path.exists()
    assert jsonl_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["episodes"] == 1
    assert summary["successes"] == 1

    stdout = capsys.readouterr().out
    assert "Found logs: 2" in stdout
    assert "Metric rows: 1" in stdout
    assert str(csv_path) in stdout
