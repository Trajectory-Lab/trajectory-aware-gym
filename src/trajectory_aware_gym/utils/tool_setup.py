from trajectory_aware_gym.tools.python_exec import PythonExecutionTool
from trajectory_aware_gym.tools.registry import ToolRegistry
from trajectory_aware_gym.tools.search import SearchTool


def build_tool_registry():
    registry = ToolRegistry()

    registry.register(PythonExecutionTool())
    registry.register(SearchTool())

    return registry
