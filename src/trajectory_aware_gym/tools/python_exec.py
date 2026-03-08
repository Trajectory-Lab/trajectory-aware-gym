import contextlib
import io
from typing import Any

from trajectory_aware_gym.tools.base import MCPTool


class PythonExecutionTool(MCPTool):
    name = "python_exec"

    description = "Execute short Python code and return stdout."

    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            }
        },
        "required": ["code"],
    }

    def run(self, **kwargs) -> dict[str, Any]:
        code = kwargs.get("code")

        # Validate required argument
        if not code:
            return {
                "status": "error",
                "error": "Missing 'code' argument",
            }

        # Validate argument type
        if not isinstance(code, str):
            return {
                "status": "error",
                "error": "'code' must be a string",
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

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
