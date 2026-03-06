from trajectory_aware_gym.adapters.tool_runtime import ToolRuntime
from trajectory_aware_gym.utils.tool_setup import build_tool_registry


def test_tool_runtime():
    registry = build_tool_registry()

    runtime = ToolRuntime(registry)

    call = {
        "tool": "python_exec",
        "arguments": {"code": "print(10*10)"},
    }

    result = runtime.execute(call)

    assert result["status"] == "success"
