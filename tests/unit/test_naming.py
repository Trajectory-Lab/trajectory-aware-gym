"""Tests for the experiment run naming protocol."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trajectory_aware_gym.storage.naming import (
    _sanitize,
    _shorten_model_id,
    generate_experiment_run_id,
    get_git_info,
    get_operator,
)

# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("orz57k-gepa-light", "orz57k-gepa-light"),
        ("HotpotQA", "hotpotqa"),
        ("some/path.name", "some-path-name"),
        ("  spaced  ", "spaced"),
        ("UPPER_CASE_123", "upper-case-123"),
        ("already-clean", "already-clean"),
        ("a..b//c", "a-b-c"),
    ],
)
def test_sanitize(raw, expected):
    assert _sanitize(raw) == expected


# ---------------------------------------------------------------------------
# _shorten_model_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("ollama/qwen3-1.7b-base", "qwen3-1-7b-base"),
        ("bedrock/llama-8b", "llama-8b"),
        ("sagemaker/qwen3-4b", "qwen3-4b"),
        ("some-model-no-slash", "some-model-no-slash"),
        ("ollama/Qwen3.4B", "qwen3-4b"),
    ],
)
def test_shorten_model_id(model_id, expected):
    assert _shorten_model_id(model_id) == expected


# ---------------------------------------------------------------------------
# generate_experiment_run_id
# ---------------------------------------------------------------------------


class TestGenerateExperimentRunId:
    def test_basic_format(self):
        ts = datetime(2026, 4, 15, 14, 30, tzinfo=UTC)
        result = generate_experiment_run_id(
            config_name="orz57k-gepa-light",
            provider="ollama",
            model_id="ollama/qwen3-1.7b-base",
            operator="jinyu",
            seed=42,
            timestamp=ts,
        )
        assert result == "orz57k-gepa-light-ollama-qwen3-1-7b-base-jinyu-seed42-20260415T1430Z"

    def test_bedrock_provider(self):
        ts = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        result = generate_experiment_run_id(
            config_name="hotpotqa-medium",
            provider="bedrock",
            model_id="bedrock/llama-8b",
            operator="alice",
            seed=0,
            timestamp=ts,
        )
        assert result == "hotpotqa-medium-bedrock-llama-8b-alice-seed0-20260301T0900Z"

    def test_special_chars_sanitized(self):
        ts = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        result = generate_experiment_run_id(
            config_name="My Config!!",
            provider="sagemaker",
            model_id="sagemaker/Model.V2",
            operator="Bob Smith",
            seed=7,
            timestamp=ts,
        )
        # All special chars become hyphens; lowercase
        assert "!!" not in result
        assert " " not in result
        assert result == "my-config-sagemaker-model-v2-bob-smith-seed7-20260101T0000Z"

    def test_defaults_to_now(self):
        """Without timestamp, uses current UTC time."""
        result = generate_experiment_run_id(
            config_name="test",
            provider="ollama",
            model_id="ollama/test",
            operator="ci",
            seed=1,
        )
        # Just verify it ends with a timestamp-like pattern
        assert result.startswith("test-ollama-test-ci-seed1-")
        # Last segment should be YYYYMMDDTHHMMZ
        ts_part = result.split("-")[-1]
        assert ts_part.endswith("Z")
        assert "T" in ts_part

    def test_deterministic_with_same_timestamp(self):
        ts = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
        kwargs = {
            "config_name": "quick-test",
            "provider": "ollama",
            "model_id": "ollama/qwen3-1.7b",
            "operator": "dev",
            "seed": 99,
            "timestamp": ts,
        }
        assert generate_experiment_run_id(**kwargs) == generate_experiment_run_id(**kwargs)


# ---------------------------------------------------------------------------
# get_operator
# ---------------------------------------------------------------------------


class TestGetOperator:
    def test_reads_repo_git_config(self, tmp_path, monkeypatch):
        """When repo git config has a user name, return it."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[user]\nname = Jinyu Han\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert get_operator() == "Jinyu Han"

    def test_reads_home_git_config_when_repo_name_missing(self, tmp_path, monkeypatch):
        """Falls back to the user's global git config."""
        workspace = tmp_path / "workspace"
        home_dir = tmp_path / "home"
        workspace.mkdir()
        home_dir.mkdir()
        (home_dir / ".gitconfig").write_text("[user]\nname = Global User\n", encoding="utf-8")
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HOME", str(home_dir))
        assert get_operator() == "Global User"

    def test_falls_back_to_user_env(self, tmp_path, monkeypatch):
        """When git config is unavailable, falls back to $USER."""
        workspace = tmp_path / "workspace"
        home_dir = tmp_path / "home"
        workspace.mkdir()
        home_dir.mkdir()
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HOME", str(home_dir))
        monkeypatch.setenv("USER", "testuser")
        assert get_operator() == "testuser"

    def test_falls_back_to_unknown(self, tmp_path, monkeypatch):
        """When git config and $USER are unavailable, returns 'unknown'."""
        workspace = tmp_path / "workspace"
        home_dir = tmp_path / "home"
        workspace.mkdir()
        home_dir.mkdir()
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HOME", str(home_dir))
        monkeypatch.delenv("USER", raising=False)
        assert get_operator() == "unknown"


# ---------------------------------------------------------------------------
# get_git_info
# ---------------------------------------------------------------------------


class TestGetGitInfo:
    def test_reads_symbolic_ref(self, tmp_path, monkeypatch):
        """Reads branch name and commit from a symbolic HEAD."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/feat/logging-v2\n")
        refs_dir = git_dir / "refs" / "heads" / "feat"
        refs_dir.mkdir(parents=True)
        (refs_dir / "logging-v2").write_text("abc1234deadbeef\n")

        monkeypatch.chdir(tmp_path)
        commit, branch = get_git_info()
        assert branch == "feat/logging-v2"
        assert commit == "abc1234deadbeef"

    def test_detached_head(self, tmp_path, monkeypatch):
        """Detached HEAD returns commit hash, no branch."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("deadbeef12345678\n")

        monkeypatch.chdir(tmp_path)
        commit, branch = get_git_info()
        assert commit == "deadbeef12345678"
        assert branch is None

    def test_no_git_dir(self, tmp_path, monkeypatch):
        """No .git directory → (None, None)."""
        monkeypatch.chdir(tmp_path)
        commit, branch = get_git_info()
        assert commit is None
        assert branch is None

    def test_packed_refs_fallback(self, tmp_path, monkeypatch):
        """Falls back to packed-refs when loose ref file doesn't exist."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
        # No loose ref file — use packed-refs instead
        (git_dir / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted\nabc123 refs/heads/main\n"
        )

        monkeypatch.chdir(tmp_path)
        commit, branch = get_git_info()
        assert branch == "main"
        assert commit == "abc123"

    def test_rejects_path_traversal_symbolic_ref(self, tmp_path, monkeypatch):
        """A symbolic ref must not resolve outside .git."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        (outside_dir / "secret-ref").write_text("deadbeef1234\n")
        (git_dir / "HEAD").write_text("ref: ../outside/secret-ref\n")

        monkeypatch.chdir(tmp_path)
        commit, branch = get_git_info()
        assert commit is None
        assert branch is None

    def test_rejects_absolute_symbolic_ref(self, tmp_path, monkeypatch):
        """A symbolic ref must not use an absolute path."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        outside_file = tmp_path / "absolute-ref"
        outside_file.write_text("abc123def456\n")
        (git_dir / "HEAD").write_text(f"ref: {outside_file}\n")

        monkeypatch.chdir(tmp_path)
        commit, branch = get_git_info()
        assert commit is None
        assert branch is None
