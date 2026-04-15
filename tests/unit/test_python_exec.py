"""Tests for the python_exec sandbox tool."""

from __future__ import annotations

import pytest

from trajectory_aware_gym.mcp.tools.python_exec import (
    _ALLOWED_MODULES,
    _get_base_namespace,
    _restricted_import,
    reset_base_namespace,
)
from trajectory_aware_gym.mcp.tools.python_exec import (
    python_exec as _python_exec_tool,
)

# The @mcp.tool() decorator wraps the function in a FunctionTool object.
# Access the underlying callable via .fn for direct unit testing.
_exec = _python_exec_tool.fn


@pytest.fixture(autouse=True)
def _clean_namespace():
    """Reset cached namespace before each test."""
    reset_base_namespace()
    yield
    reset_base_namespace()


# ---------------------------------------------------------------------------
# _restricted_import
# ---------------------------------------------------------------------------


class TestRestrictedImport:
    @pytest.mark.parametrize("module", ["math", "collections", "itertools", "json", "re"])
    def test_allows_whitelisted_modules(self, module):
        mod = _restricted_import(module)
        assert mod is not None

    @pytest.mark.parametrize("module", ["os", "subprocess", "socket", "sys", "shutil"])
    def test_blocks_disallowed_modules(self, module):
        with pytest.raises(ImportError, match="not allowed"):
            _restricted_import(module)

    def test_allows_submodule_of_whitelisted(self):
        mod = _restricted_import("collections.abc")
        assert mod is not None

    def test_blocks_submodule_of_disallowed(self):
        with pytest.raises(ImportError, match="not allowed"):
            _restricted_import("os.path")


# ---------------------------------------------------------------------------
# _get_base_namespace
# ---------------------------------------------------------------------------


class TestBaseNamespace:
    def test_has_pre_imported_math_functions(self):
        ns = _get_base_namespace()
        for name in ("sqrt", "gcd", "factorial", "ceil", "floor", "log2", "pi"):
            assert name in ns, f"Missing pre-import: {name}"

    def test_has_pre_imported_collections(self):
        ns = _get_base_namespace()
        for name in ("defaultdict", "deque", "Counter"):
            assert name in ns

    def test_has_restricted_builtins(self):
        ns = _get_base_namespace()
        builtins = ns["__builtins__"]
        assert "print" in builtins
        assert "len" in builtins
        assert "range" in builtins

    @pytest.mark.parametrize(
        "removed", ["open", "breakpoint", "exec", "eval", "compile", "input", "exit", "quit"]
    )
    def test_dangerous_builtins_removed(self, removed):
        ns = _get_base_namespace()
        builtins = ns["__builtins__"]
        assert removed not in builtins

    def test_import_hook_installed(self):
        ns = _get_base_namespace()
        builtins = ns["__builtins__"]
        assert builtins["__import__"] is _restricted_import

    def test_namespace_is_cached(self):
        ns1 = _get_base_namespace()
        ns2 = _get_base_namespace()
        assert ns1 is ns2

    def test_reset_clears_cache(self):
        ns1 = _get_base_namespace()
        reset_base_namespace()
        ns2 = _get_base_namespace()
        assert ns1 is not ns2


# ---------------------------------------------------------------------------
# python_exec tool
# ---------------------------------------------------------------------------


class TestPythonExec:
    def test_simple_print(self):
        result = _exec(code="print(42)")
        assert result["status"] == "success"
        assert result["output"].strip() == "42"

    def test_empty_code_returns_error(self):
        result = _exec(code="")
        assert result["status"] == "error"
        assert "Missing" in result["error"]

    def test_computation_with_pre_imports(self):
        result = _exec(code="print(factorial(5))")
        assert result["status"] == "success"
        assert result["output"].strip() == "120"

    def test_math_functions_available(self):
        result = _exec(code="print(sqrt(16))")
        assert result["status"] == "success"
        assert result["output"].strip() == "4.0"

    def test_collections_available(self):
        result = _exec(code="c = Counter('aabbc'); print(c.most_common(1))")
        assert result["status"] == "success"
        assert "a" in result["output"] or "b" in result["output"]

    def test_itertools_available(self):
        result = _exec(code="print(list(combinations([1,2,3], 2)))")
        assert result["status"] == "success"
        assert "(1, 2)" in result["output"]

    def test_def_and_for_loops(self):
        code = """\
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
print(fib(10))
"""
        result = _exec(code=code)
        assert result["status"] == "success"
        assert result["output"].strip() == "55"

    def test_import_allowed_module(self):
        result = _exec(code="import random; print(type(random.randint(1,10)))")
        assert result["status"] == "success"
        assert "int" in result["output"]

    def test_import_blocked_module(self):
        result = _exec(code="import os")
        assert result["status"] == "error"
        assert "not allowed" in result["error"]

    def test_import_subprocess_blocked(self):
        result = _exec(code="import subprocess")
        assert result["status"] == "error"
        assert "not allowed" in result["error"]

    def test_runtime_error_captured(self):
        result = _exec(code="1 / 0")
        assert result["status"] == "error"
        assert "division by zero" in result["error"]

    def test_name_error_captured(self):
        result = _exec(code="print(undefined_variable)")
        assert result["status"] == "error"
        assert "undefined_variable" in result["error"]

    def test_no_stdout_returns_empty_output(self):
        result = _exec(code="x = 42")
        assert result["status"] == "success"
        assert result["output"] == ""

    def test_multiline_output(self):
        result = _exec(code="for i in range(3): print(i)")
        assert result["status"] == "success"
        assert result["output"].strip() == "0\n1\n2"

    def test_exec_builtin_blocked(self):
        result = _exec(code="exec('print(1)')")
        assert result["status"] == "error"

    def test_eval_builtin_blocked(self):
        result = _exec(code="eval('1+1')")
        assert result["status"] == "error"

    def test_open_builtin_blocked(self):
        result = _exec(code="open('/etc/passwd')")
        assert result["status"] == "error"

    def test_namespace_isolation_between_calls(self):
        _exec(code="my_secret = 999")
        result = _exec(code="print(my_secret)")
        assert result["status"] == "error"
        assert "my_secret" in result["error"]
