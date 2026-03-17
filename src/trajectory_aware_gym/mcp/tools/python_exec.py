import contextlib
import io
from typing import Any

from trajectory_aware_gym.mcp.server import mcp


@mcp.tool()
def python_exec(code: str) -> dict[str, Any]:
    """Execute short Python code and return stdout."""
    if not code:
        return {
            "status": "error",
            "error": "Missing 'code' argument",
        }

    buffer = io.StringIO()

    try:
        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
        }

        with contextlib.redirect_stdout(buffer):
            # Intentionally allow exec for agent Python tool execution.
            # Sandbox limits builtins to a safe subset.
            exec(code, {"__builtins__": safe_builtins})  # nosec B102

        return {
            "status": "success",
            "output": buffer.getvalue(),
        }

    except Exception as error:
        return {
            "status": "error",
            "error": str(error),
        }
