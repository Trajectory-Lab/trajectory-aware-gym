"""GEM-compatible Python execution sandbox.

Provides a sandboxed ``python_exec`` tool with:
- Full Python builtins (minus file I/O)
- Pre-imported standard math/computation libraries matching GEM's BASE_IMPORTS
- Allowlist-based import restriction (blocks os, subprocess, socket, etc.)
"""

import builtins
import contextlib
import io
from typing import Any

from trajectory_aware_gym.mcp.server import mcp

# ── Import allowlist ────────────────────────────────────────────
# Only these top-level modules (and their submodules) may be imported
# by sandboxed code.  Everything else is blocked.

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
# Matches GEM's BASE_IMPORTS so models don't need explicit imports
# for common math/algorithm operations.

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


@mcp.tool()
def python_exec(code: str) -> dict[str, Any]:
    """Execute Python code and return stdout.

    A GEM-compatible sandbox with standard math/computation libraries
    pre-imported (math, collections, itertools, numpy, etc.).
    Imports are restricted to a safe allowlist.
    """
    if not code:
        return {"status": "error", "error": "Missing 'code' argument"}

    buffer = io.StringIO()
    ns = dict(_get_base_namespace())

    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, ns)  # nosec B102  # noqa: S102

        return {"status": "success", "output": buffer.getvalue()}

    except Exception as error:  # noqa: BLE001
        return {"status": "error", "error": str(error)}
