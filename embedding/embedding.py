from abc import ABC, abstractmethod
from typing import List, Dict

from chunking.chunking import Chunk


class EmbeddingMetadata:
    def __init__(self, input_tokens: int, latency_ms: int):
        self.input_tokens = input_tokens
        self.latency_ms = latency_ms


class Embeddings:
    def __init__(self, embeddings: List[List[float]], metadata: EmbeddingMetadata):
        self.embeddings = embeddings
        self.metadata = metadata


class BaseEmbedding(ABC):
    def __init__(self, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__()
        self.dimension = dimensions
        self.normalize = normalize

    @abstractmethod
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        pass

    @abstractmethod
    def embed(self, chunk: Chunk) -> Embeddings:
        pass

    def embed_list(self, chunks: List[Chunk]) -> List[Embeddings]:
        if not isinstance(chunks, list):
            return [self.embed(chunks)]
        return [self.embed(chunk) for chunk in chunks]
