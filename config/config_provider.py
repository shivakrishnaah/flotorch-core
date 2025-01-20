from abc import ABC, abstractmethod
from typing import Any

class ConfigProvider(ABC):
    """
    Abstract base class for configuration providers.
    """
    @abstractmethod
    def get(self, key: str) -> Any:
        """
        Retrieves the configuration value for the given key.
        Args:
            key (str): The configuration key to retrieve.
        Returns:
            Any: The value of the configuration key.
        """
        pass
