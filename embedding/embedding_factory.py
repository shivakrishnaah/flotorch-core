from embedding.titanv2_embedding import TitanV2Embedding
from embedding.titanv1_embedding import TitanV1Embedding
from embedding.embedding import BaseEmbedding


class EmbeddingFactory:
    """
    Factory to create embedding instances based on the embedding model ID.
    """

    @staticmethod
    def create_embedding(model_id: str, region: str, dimensions: int) -> BaseEmbedding:
        """
        Creates and returns an embedding object based on the model ID.

        Args:
            model_id (str): Identifier for the embedding model.
            region (str): AWS region where the model is deployed.
            dimensions (int): Dimension of the embedding vectors.

        Returns:
            BaseEmbedding: The appropriate embedding object.
        """
        if "amazon.titan-embed-text-v2:0" in model_id:
            return TitanV2Embedding(model_id, region, dimensions)
        elif "amazon.titan-text-express-v1" in model_id:
            return TitanV1Embedding(model_id, region, dimensions)
        else:
            raise ValueError(f"Unsupported embedding model: {model_id}")
