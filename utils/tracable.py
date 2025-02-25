from abc import ABC, abstractmethod

class Traceable(ABC):
    """
    Interface that requires classes to return a dictionary with metadata for tracing.
    """
    @abstractmethod
    def get_trace_data(self) -> dict:
        pass