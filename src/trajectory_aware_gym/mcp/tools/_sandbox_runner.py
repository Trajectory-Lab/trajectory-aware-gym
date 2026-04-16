"""Sandbox child process for python_exec.

Invoked by ``python_exec.py`` as ``python -m
trajectory_aware_gym.mcp.tools._sandbox_runner``.  Reads Python code from
stdin, executes it with restricted builtins, an import allowlist, and
RLIMIT_AS / RLIMIT_CPU caps, then writes a JSON result to stdout.

Kept deliberately small and free of project imports so subprocess startup
stays fast (no FastMCP / LiteLLM init in the child).
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import resource
import sys
import traceback
from typing import Any

# ── Limits ──────────────────────────────────────────────────────
# Hard caps applied via ``resource.setrlimit`` in the child.  Wall-clock
# timeout is enforced by the parent via ``subprocess.run(timeout=...)``.
# Virtual address space cap.  Must be generous enough for numpy/sympy
# imports — OpenBLAS reserves large VM regions even when not actively used.
_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB virtual address space
# CPU time (not wall) — defense in depth. Kept higher than the parent's wall
# timeout so Python's shutdown / GC / module-cleanup CPU cost after a
# successful run cannot tip the process over RLIMIT_CPU and SIGKILL a valid
# result. ``main()`` uses ``os._exit(0)`` to skip most shutdown work, but
# this gives headroom for cases where _exit is reached slightly late.
_CPU_TIME_LIMIT_SECONDS = 30
_MAX_OUTPUT_BYTES = 64 * 1024  # truncate stdout to keep IPC payload small

# ── Import allowlist ────────────────────────────────────────────
_ALLOWED_MODULES = frozenset(
    {
        # Math and numeric
        "math",
        "cmath",
        "decimal",
        "fractions",
        "statistics",
        "numbers",
        # Data structures and algorithms
        "collections",
        "itertools",
        "functools",
        "operator",
        "bisect",
        "heapq",
        "array",
        # String and regex
        "re",
        "string",
        "textwrap",
        # Utilities
        "copy",
        "pprint",
        "json",
        "random",
        "datetime",
        "time",
        # Typing
        "typing",
        "abc",
        "enum",
        # Scientific (optional — no error if not installed)
        "numpy",
        "pandas",
        "sympy",
    }
)

_builtin_import = builtins.__import__


def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Import hook that only allows whitelisted modules."""
    top_level = name.split(".")[0]
    if top_level not in _ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed")
    return _builtin_import(name, *args, **kwargs)


# ── Pre-populated namespace (cached) ───────────────────────────
_GEM_PRE_IMPORTS = """\
import math
from math import (
    floor, log2, log10, sqrt, comb, gcd, ceil, inf,
    isqrt, factorial, atan2, pi, log, prod,
)
from collections import defaultdict, deque, Counter, OrderedDict
from itertools import (
    accumulate, chain, combinations, count, permutations, product,
    groupby, islice, repeat, zip_longest, cycle, pairwise,
)
from functools import reduce, cache, lru_cache, cmp_to_key, partial
from operator import itemgetter, sub, xor, or_, iand
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import (
    heappush, heappop, heapify, merge, nlargest, nsmallest, heapreplace,
)
import re
from re import search as re_search
import string
from string import ascii_lowercase, ascii_uppercase
import copy
import random
from random import randrange, shuffle
import fractions
import decimal
import json
"""

_BASE_NAMESPACE: dict[str, Any] | None = None


def _get_base_namespace() -> dict[str, Any]:
    """Build (and cache) the base execution namespace with GEM pre-imports."""
    global _BASE_NAMESPACE  # noqa: PLW0603
    if _BASE_NAMESPACE is not None:
        return _BASE_NAMESPACE

    restricted_builtins = {**vars(builtins), "__import__": _restricted_import}
    # Remove builtins that have no place in a computation sandbox.
    for name in ("open", "breakpoint", "exec", "eval", "compile", "input", "exit", "quit"):
        restricted_builtins.pop(name, None)

    ns: dict[str, Any] = {"__builtins__": restricted_builtins}

    # Execute GEM-compatible pre-imports into the namespace.
    exec(_GEM_PRE_IMPORTS, ns)  # nosec B102  # noqa: S102

    # Optional heavy libraries — skip silently if not installed.
    for module_name, alias in [("numpy", "np"), ("pandas", "pd"), ("sympy", "sympy")]:
        try:
            mod = _builtin_import(module_name)
            ns[alias] = mod
            ns[module_name] = mod
        except ImportError:
            pass

    _BASE_NAMESPACE = ns
    return _BASE_NAMESPACE


def reset_base_namespace() -> None:
    """Clear the cached namespace (for test isolation)."""
    global _BASE_NAMESPACE  # noqa: PLW0603
    _BASE_NAMESPACE = None


def _apply_resource_limits() -> None:
    """Cap virtual memory and CPU time in this process."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES))
    except (ValueError, OSError):
        # macOS / unprivileged contexts may reject; parent still has wall timeout.
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_TIME_LIMIT_SECONDS, _CPU_TIME_LIMIT_SECONDS))
    except (ValueError, OSError):
        pass


def _format_error(error: BaseException) -> str:
    """Build a concise error message with the exception class and location.

    The model consumes this string to decide how to fix its next tool call,
    so it must include:
      - the exception class name (``MemoryError`` is useless as just ``""``)
      - a line number and snippet from the offending source when available
      - the exception's message
    """
    message = str(error) or type(error).__name__

    # Find the deepest frame that came from the <string> source, i.e. the
    # user's code. The first frames are inside this runner.
    line_hint = ""
    tb = error.__traceback__
    user_tb = None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<string>":
            user_tb = tb
        tb = tb.tb_next
    if user_tb is not None:
        line_hint = f" at line {user_tb.tb_lineno}"
    elif isinstance(error, SyntaxError) and error.lineno is not None:
        line_hint = f" at line {error.lineno}"

    return f"{type(error).__name__}{line_hint}: {message}"


def execute(code: str) -> dict[str, Any]:
    """Execute Python code in the sandbox namespace and return the result dict."""
    if not code:
        return {"status": "error", "error": "Missing 'code' argument"}

    buffer = io.StringIO()
    ns = dict(_get_base_namespace())

    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "<string>", "exec"), ns)  # nosec B102  # noqa: S102
        output = buffer.getvalue()
        if len(output) > _MAX_OUTPUT_BYTES:
            output = output[:_MAX_OUTPUT_BYTES] + "\n[output truncated]"
        return {"status": "success", "output": output}
    except SyntaxError as error:
        return {"status": "error", "error": _format_error(error)}
    except MemoryError:
        # MemoryError often has an empty message — make sure the model sees
        # something useful and can steer away from huge allocations.
        return {
            "status": "error",
            "error": (
                f"MemoryError: allocation exceeded the sandbox's "
                f"{_MEMORY_LIMIT_BYTES // (1024**3)} GB memory limit"
            ),
        }
    except Exception as error:  # noqa: BLE001
        # Partial stdout is preserved so the model can see what the code
        # printed before crashing.
        partial_output = buffer.getvalue()
        result: dict[str, Any] = {"status": "error", "error": _format_error(error)}
        if partial_output:
            result["partial_output"] = partial_output[:_MAX_OUTPUT_BYTES]
        # Include a short traceback for runtime errors so the model can
        # locate the failure inside a multi-line snippet. Filter to frames
        # that originated in the user-supplied code ("<string>" source) so
        # the noise from this runner's exec(...) call doesn't leak in.
        user_tb_text = "".join(
            traceback.format_list(
                [
                    frame
                    for frame in traceback.extract_tb(error.__traceback__)
                    if frame.filename == "<string>"
                ]
            )
        )
        if user_tb_text:
            result["traceback"] = user_tb_text.strip()
        return result


def main() -> None:
    """Read code from stdin, execute, write JSON result to stdout.

    Uses ``os._exit(0)`` after flushing stdout so the interpreter's shutdown
    phase (GC, atexit handlers, module cleanup) does not consume additional
    CPU time. Without this, Python's cleanup cost after a successful long
    computation can trip ``RLIMIT_CPU`` and SIGKILL the process *after* the
    result has been written — causing the parent to discard a valid answer.
    """
    _apply_resource_limits()
    code = sys.stdin.read()
    result = execute(code)
    json.dump(result, sys.stdout)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
