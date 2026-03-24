from trajectory_aware_gym.adapters.tool_runtime import ToolRuntime


async def test_tool_runtime():
    runtime = ToolRuntime()

    call = {
        "tool": "python_exec",
        "arguments": {"code": "print(10*10)"},
    }

    result = await runtime.execute(call)

    assert result["status"] == "success"
