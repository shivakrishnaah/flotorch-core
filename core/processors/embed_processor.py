from core.embedding import EmbedderFactory
from typing import Dict, List, Tuple
from config.experimental_config import ExperimentalConfig
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class EmbedProcessor:
    """Processor for embedding text chunks."""

    def __init__(self, experimentalConfig : ExperimentalConfig) -> None:
        self.experimentalConfig = experimentalConfig
        self.embedder = EmbedderFactory.create_embedder(experimentalConfig)

    def embed(self, chunks: List[str]) -> List[Tuple[List[float], str]]:
        """Embed each chunk one by one."""
        embeddings = []
        try:
            dimensions = self.experimentalConfig.vector_dimension 
            normalize = True  # Always normalize

            logger.info(f"Embedding {len(chunks)} chunks with dimensions: {dimensions}.")
            for idx, chunk in enumerate(chunks):
                logger.debug(f"Embedding chunk {idx + 1}/{len(chunks)}: {chunk[:50]}...")
                embedding = self.embedder.embed(chunk, dimensions=dimensions, normalize=normalize)
                embeddings.append((embedding, chunk))  # Append as tuple

            logger.info("Embedding process completed successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding process: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Embed each chunk one by one."""
        try:
            dimensions = self.experimentalConfig.vector_dimension 
            normalize = True  # Always normalize
            embedding = self.embedder.embed(text, dimensions=dimensions, normalize=normalize)
            logger.info("Embedding text process completed successfully.")
            return embedding
        except Exception as e:
            logger.error(f"Error during embedding process: {e}")
            raise
