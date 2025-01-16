from abc import ABC, abstractmethod
from langchain.text_splitter import CharacterTextSplitter
from typing import List
import uuid

class Chunk:
    def __init__(self, data):
        self.id = str(uuid.uuid4())
        self.data = data
        self.child_data = None
    
    def add_child(self, child_data):
        if not self.child_data:
            self.child_data = []
        self.child_data.append(child_data)

    def __str__(self):
        return f"Parent Chunk: {self.data}, Chunk: {self.child_data}"


class BaseChunker(ABC):
    def __init__(self):
        super().__init__()
        self.space = ' '
        self.separators = [' ', '\t', '\n', '\r', '\f', '\v']
        self.tokens_per_charecter = 4
    
    @abstractmethod
    def chunk(self, data: str) -> List[Chunk]:
        pass
    
    def chunk_list(self, data: List[str]) -> List[Chunk]:
        result = []
        for d in data:
            result.extend(self.chunk(d))
        return result

    def _clean_data(self, data: str) -> str:
        for sep in self.separators:
            data = data.replace(sep, self.space)
        return data