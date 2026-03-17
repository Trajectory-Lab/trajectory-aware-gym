from trajectory_aware_gym.mcp.tools.python_exec import python_exec


def test_python_execution():
    result = python_exec.fn(code="print(2+2)")

    assert result["status"] == "success"
    assert "4" in result["output"]


def test_python_exec_missing_code():
    result = python_exec.fn(code="")

    assert result["status"] == "error"
    assert result["error"] == "Missing 'code' argument"


def test_python_exec_runtime_error():
    result = python_exec.fn(code="1/0")

    assert result["status"] == "error"
    assert "division by zero" in result["error"]
