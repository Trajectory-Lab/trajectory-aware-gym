from trajectory_aware_gym.tools.python_exec import PythonExecutionTool


def test_python_execution():
    tool = PythonExecutionTool()

    result = tool.run(code="print(2+2)")

    assert result["status"] == "success"
    assert "4" in result["output"]
