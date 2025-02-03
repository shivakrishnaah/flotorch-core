from abc import ABC, abstractmethod
from typing import List, Dict, Any

"""
This class is responsible for storing the data in the database.
"""
class DBStorage(ABC):

    @abstractmethod
    def read(self, key) -> dict:
        pass

    @abstractmethod
    def write(self, item: dict):
        pass

    def bulk_write(self, items: List[dict]):
        for item in items:
            self.write(item)

    @abstractmethod
    def update(self, key: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Update method accepts:
        - `key`: Unique identifier to find the record (e.g., {'id': 123})
        - `data`: Fields to be updated with new values (e.g., {'status': 'completed'})
        """