"""Sandboxed ``python_exec`` tool.

Spawns a fresh Python subprocess per call (via
``trajectory_aware_gym.mcp.tools._sandbox_runner``).  This isolates the
host process from runaway code: a ``while True`` or memory-bomb in one
call cannot wedge the eval thread pool or eat all RAM.

Limits enforced:
  - Wall-clock timeout (parent, ``subprocess.run(timeout=...)``)
  - CPU time + virtual memory (child, ``resource.setrlimit``)
  - Output truncation (child, in ``_sandbox_runner.execute``)

Trade-off: ~50-100ms subprocess startup per call.  Acceptable for tool
use where each call already involves an LLM round-trip.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from typing import Any

from trajectory_aware_gym.mcp.server import mcp

logger = logging.getLogger(__name__)

_SANDBOX_MODULE = "trajectory_aware_gym.mcp.tools._sandbox_runner"
_WALL_TIMEOUT_SECONDS = 15  # parent-side wall clock cap


@mcp.tool()
def python_exec(code: str) -> dict[str, Any]:
    """Execute Python code in a sandboxed subprocess and return stdout.

    The sandbox has standard math/computation libraries pre-imported
    (math, collections, itertools, numpy, sympy, etc.) and an import
    allowlist.  Hard caps on CPU time, memory, wall clock, and output
    size keep one bad call from wedging the host.
    """
    if not code:
        return {"status": "error", "error": "Missing 'code' argument"}

    try:
        completed = subprocess.run(  # noqa: S603  # trusted argv
            [sys.executable, "-m", _SANDBOX_MODULE],
            input=code,
            capture_output=True,
            text=True,
            timeout=_WALL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": f"Execution exceeded wall-clock timeout of {_WALL_TIMEOUT_SECONDS}s",
        }
    except OSError as exc:
        logger.exception("Failed to spawn python_exec sandbox")
        return {"status": "error", "error": f"Sandbox spawn failed: {exc}"}

    # The child uses ``os._exit(0)`` after flushing stdout, so under normal
    # completion the returncode is 0 and stdout holds a JSON result. When
    # the process is killed post-output (rare race with resource limits on
    # cleanup) stdout may still be valid — prefer it over the exit code.
    parsed_stdout: dict[str, Any] | None = None
    if completed.stdout:
        try:
            candidate = json.loads(completed.stdout)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and "status" in candidate:
            parsed_stdout = candidate

    if parsed_stdout is not None:
        return parsed_stdout

    if completed.returncode != 0:
        # Common causes: SIGXCPU (CPU limit), SIGKILL from OOM-killer, or an
        # import error inside the runner before the sandbox executed user
        # code.  Map well-known signal return codes to a human-readable hint
        # so the model can steer its next attempt.
        signal_hints = {
            -9: "killed by SIGKILL (likely OOM or exceeded memory/CPU limits)",
            -11: "segfault (SIGSEGV)",
            -24: "CPU time limit exceeded (SIGXCPU)",
        }
        hint = signal_hints.get(completed.returncode, "")
        stderr = completed.stderr.strip() or "no stderr"
        error_msg = f"Sandbox process exited {completed.returncode}"
        if hint:
            error_msg += f" — {hint}"
        if stderr != "no stderr":
            error_msg += f"\nstderr: {stderr[:500]}"
        return {"status": "error", "error": error_msg}

    preview = completed.stdout[:200] if completed.stdout else "(empty)"
    return {
        "status": "error",
        "error": f"Sandbox returned invalid JSON: {preview}",
    }
