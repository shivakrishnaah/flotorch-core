import ollama
from .embedding import BaseEmbedding, Embeddings, EmbeddingMetadata
from chunking.chunking import Chunk
"""

This class is responsible for embedding the text using the Llama model.
If Ollama server is running remotely set the environment variable OLLAMA_HOST to the server URL."""
class LlamaEmbedding(BaseEmbedding):
    """
    Initializes the LlamaEmbedding class.
    :param model_id: The model id of the Llama model.
    """

    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id

    """
    Prepares the chunk for embedding.
    :param chunk: The chunk to be embedded.
    :return: The prepared chunk.
    """
    def embed(self, chunk: Chunk) -> Embeddings:
        response = ollama.embeddings(model=self.model_id, prompt=chunk.data)
        embedding = Embeddings(embeddings=response['embedding'], metadata=None)
        return embedding