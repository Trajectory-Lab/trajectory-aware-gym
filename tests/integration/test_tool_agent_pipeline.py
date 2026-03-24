from trajectory_aware_gym.adapters.tool_runtime import ToolRuntime


def test_tool_runtime():
    runtime = ToolRuntime()

    call = {
        "tool": "python_exec",
        "arguments": {"code": "print(10*10)"},
    }

    result = runtime.execute(call)

    assert result["status"] == "success"
