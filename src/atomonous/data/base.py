from abc import ABC, abstractmethod
from typing import Any
from .types import AIFormat, Filepath


class HeuristicMismatchError(Exception):
    """Raised when a converter supports a data type but the content doesn't match its heuristics."""
    pass

class DataConverter[T](ABC):
    """
    Abstract Base Class for all scientific data converters.
    Handles mapping arbitrary inputs (T_in) to str or PIL.Image.Image.
    """

    @property
    @abstractmethod
    def input_type(self) -> T:
        """The specific Python type this converter handles."""
        pass

    @abstractmethod
    def convert(self, data: T) -> AIFormat:
        """
        Converts the input data into either a str or a PIL.Image.Image.
        
        Args:
            data: The input object (Path, Array, Custom class, Dict, etc.)
            
        Returns:
            AIFormat: The data format natively ingestible by the Agent.
        """
        pass

    def can_handle(self, data: Any) -> bool:
        """
        Checks if this converter can handle the given input.
        Default implementation uses isinstance against self.input_type.
        """
        return isinstance(data, self.input_type)
