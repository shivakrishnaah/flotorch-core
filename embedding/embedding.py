from abc import ABC, abstractmethod
from typing import List, Dict

from chunking.chunking import Chunk

"""
This class is responsible for embedding the text using the Llama model.
"""
class EmbeddingMetadata:
    """
    Initializes the EmbeddingMetadata class.
    :param input_tokens: The number of input tokens.
    :param latency_ms: The latency in milliseconds.
    """
    def __init__(self, input_tokens: int, latency_ms: int):
        self.input_tokens = input_tokens
        self.latency_ms = latency_ms


class Embeddings:
    """
    Initializes the Embeddings class.
    :param embeddings: The embeddings.
    :param metadata: The metadata.
    """
    def __init__(self, embeddings: List[List[float]], metadata: EmbeddingMetadata):
        self.embeddings = embeddings
        self.metadata = metadata

"""
This class is responsible for embedding the text."""
class BaseEmbedding(ABC):
    """
    Initializes the BaseEmbedding class.
    :param dimensions: The dimensions of the embedding.
    :param normalize: Normalize the embedding.
    """

    def __init__(self, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__()
        self.dimension = dimensions
        self.normalize = normalize

    """
    Prepares the chunk for embedding.
    :param chunk: The chunk to be embedded.
    :return: The prepared chunk.
    """
    @abstractmethod
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        pass

    """
    Embeds the chunk.
    :param chunk: The chunk to be embedded.
    :return: The embeddings.
    """
    @abstractmethod
    def embed(self, chunk: Chunk) -> Embeddings:
        pass

    """
    Embeds the list of chunks.
    :param chunks: The list of chunks to be embedded.
    :return: The list of embeddings.
    """
    def embed_list(self, chunks: List[Chunk]) -> List[Embeddings]:
        if not isinstance(chunks, list):
            return [self.embed(chunks)]
        return [self.embed(chunk) for chunk in chunks]
