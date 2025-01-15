from abc import ABC, abstractmethod
from storage.storage import StorageProvider, LocalStorage
import re
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


class BaseChunk(ABC):
    def __init__(self):
        super().__init__()
        self.space = ' '
        self.seperators = [' ', '\t', '\n', '\r', '\f', '\v']
        self.tokens_per_charecter = 4
    
    @abstractmethod
    def chunk(self, data) -> List[Chunk]:
        pass

    def save(self, data: List[Chunk]) -> None:
        self.storage_provider.write(data)

    def _clean_data(self, data: str) -> str:
        for sep in self.separators:
            data = data.replace(sep, self.space)
        return data
    

class FixedSizeChunk(BaseChunk):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = self.tokens_per_charecter * chunk_size
        self.chunk_overlap = int (chunk_overlap * self.chunk_size / 100)

    def chunk(self, data: str) -> List[Chunk]:
        if not data:
            raise ValueError("Input text cannot be empty or None")

        data = self._clean_data(data)
        self.text_splitter = CharacterTextSplitter(
            separator=self.space,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        chunks = self.text_splitter.split_text(data)
        return [Chunk(chunk) for chunk in chunks]

class HieraricalChunk(FixedSizeChunk):
    def __init__(self, chunk_size: int, chunk_overlap: int, parent_chunk_size: int):
        super().__init__(chunk_size, chunk_overlap)
        if parent_chunk_size <= 0:
            raise ValueError("child_chunk_size must be positive")
        if parent_chunk_size > chunk_size:
            raise ValueError("child_chunk_size must be less than parent chunk size")
        self.parent_chunk_size = self.tokens_per_charecter * parent_chunk_size   

    def chunk(self, data: str) -> List[Chunk]:
        if not data:
            raise ValueError("Input text cannot be empty or None")

        data = self.clean_data(data)
        self.parent_text_splitter = CharacterTextSplitter(
            separator=self.space,
            chunk_size=self.parent_chunk_size,
            chunk_overlap=0, # Can change this at a later point of time
            length_function=len,
            is_separator_regex=False
        )
        self.child_text_splitter = CharacterTextSplitter(
            separator=self.space,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        parent_chunks = self.parent_text_splitter.split_text(data)
        overall_chunks = []
        for parent_chunk in parent_chunks:
            chunk_object = Chunk(parent_chunk)
            child_chunks = self.child_text_splitter.split_text(parent_chunk)
            for child_chunk in child_chunks:
                chunk_object.add_child(child_chunk)
                overall_chunks.append(chunk_object)
        return overall_chunks