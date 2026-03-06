from abc import ABC, abstractmethod
from typing import Any


class MCPTool(ABC):
    """
    Base interface for tools callable by an agent.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """
        Execute tool logic.

        Returns
        -------
        Dict[str, Any]
            JSON serializable result.
        """
        raise NotImplementedError
