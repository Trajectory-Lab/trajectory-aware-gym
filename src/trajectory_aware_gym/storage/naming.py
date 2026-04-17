"""Unified naming protocol for experiment runs.

Generates human-readable, sortable IDs used as:
- ``experiment_runs`` DB primary key
- S3 artifact path prefix

Format: ``{config}-{provider}-{model_short}-{operator}-seed{seed}-{YYYYMMDD}T{HHMM}Z``
Example: ``orz57k-gepa-light-ollama-qwen3-1.7b-jinyu-seed42-20260415T1430Z``
"""

from __future__ import annotations

import configparser
import os
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def _sanitize(segment: str) -> str:
    """Lowercase, replace non-alphanumeric chars with hyphens, collapse runs."""
    s = segment.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _shorten_model_id(model_id: str) -> str:
    """Strip provider prefix and sanitize.

    ``"ollama/qwen3-1.7b-base"`` → ``"qwen3-1-7b-base"``
    ``"bedrock/llama-8b"`` → ``"llama-8b"``
    """
    # Remove provider prefix (everything before the first slash)
    if "/" in model_id:
        model_id = model_id.split("/", 1)[1]
    return _sanitize(model_id)


def generate_experiment_run_id(
    config_name: str,
    provider: str,
    model_id: str,
    operator: str,
    seed: int,
    timestamp: datetime | None = None,
) -> str:
    """Build a deterministic, human-readable experiment run ID.

    Parameters
    ----------
    config_name:
        E.g. ``"orz57k-gepa-light"`` — the experiment config name.
    provider:
        E.g. ``"ollama"``, ``"bedrock"``, ``"sagemaker"``.
    model_id:
        Full model identifier, e.g. ``"ollama/qwen3-1.7b-base"``.
        Provider prefix is stripped; remaining name is sanitized.
    operator:
        User name (e.g. from ``get_operator()``).
    seed:
        Replication seed.
    timestamp:
        Override for deterministic tests. Defaults to ``datetime.now(UTC)``.
    """
    ts = timestamp or datetime.now(UTC)
    ts_str = ts.strftime("%Y%m%dT%H%MZ")

    parts = [
        _sanitize(config_name),
        _sanitize(provider),
        _shorten_model_id(model_id),
        _sanitize(operator),
        f"seed{seed}",
        ts_str,
    ]
    return "-".join(parts)


def get_operator() -> str:
    """Return the current operator name.

    Tries git config files first, falls back to ``$USER`` env var,
    then ``"unknown"``.
    """
    git_dir = _find_git_dir()
    config_paths = [git_dir / "config"] if git_dir is not None else []
    config_paths.append(Path.home() / ".gitconfig")
    for config_path in config_paths:
        if (name := _read_git_user_name(config_path)) is not None:
            return name
    return os.getenv("USER", "unknown")


def _find_git_dir(start: Path | None = None) -> Path | None:
    """Walk upward from ``start`` and return the repo's git metadata directory."""
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / ".git"
        if candidate.is_dir():
            return candidate
        if not candidate.is_file():
            continue
        git_pointer = candidate.read_text(encoding="utf-8").strip()
        if not git_pointer.startswith("gitdir: "):
            continue
        resolved = (candidate.parent / git_pointer.removeprefix("gitdir: ").strip()).resolve()
        if resolved.is_dir():
            return resolved
    return None


def _read_git_user_name(config_path: Path) -> str | None:
    """Read ``[user] name`` from a git config file."""
    if not config_path.is_file():
        return None

    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(config_path, encoding="utf-8")
    except (configparser.Error, OSError):
        return None

    user_name = parser.get("user", "name", fallback="").strip()
    return user_name or None


def _resolve_git_ref_path(git_dir: Path, ref: str) -> Path | None:
    """Resolve a symbolic ref path only if it stays under ``git_dir``."""
    if not ref or "\\" in ref:
        return None

    ref_path = PurePosixPath(ref)
    if ref_path.is_absolute() or any(part == ".." for part in ref_path.parts):
        return None

    resolved = (git_dir / Path(*ref_path.parts)).resolve()
    try:
        resolved.relative_to(git_dir.resolve())
    except ValueError:
        return None
    return resolved


def get_git_info() -> tuple[str | None, str | None]:
    """Return ``(commit_hash, branch_name)`` by reading ``.git/HEAD``.

    Uses file I/O only — no subprocess calls. Returns ``(None, None)``
    if the git directory cannot be found or read.
    """
    git_dir = _find_git_dir()
    if git_dir is None:
        return None, None

    # Read HEAD
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None, None

    head_content = head_path.read_text(encoding="utf-8").strip()

    branch: str | None = None
    commit: str | None = None

    if head_content.startswith("ref: "):
        # Symbolic reference, e.g. "ref: refs/heads/feat/logging-v2"
        ref = head_content[5:]
        ref_path = _resolve_git_ref_path(git_dir, ref)
        if ref_path is None:
            return None, None
        branch = ref.removeprefix("refs/heads/")
        if ref_path.is_file():
            commit = ref_path.read_text(encoding="utf-8").strip()
        else:
            # Packed refs fallback
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(ref):
                        commit = line.split()[0]
                        break
    else:
        # Detached HEAD — content is the commit hash itself
        commit = head_content

    return commit, branch
