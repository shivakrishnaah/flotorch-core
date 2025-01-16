from typing import List, Dict

from chunking.chunking import Chunk
from .bedrock_embedding import BedRockEmbedding


class TitanV1Embedding(BedRockEmbedding):

    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__(model_id, region, dimensions, normalize)

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"inputText": chunk.data, "embeddingConfig": {"outputEmbeddingLength": self.dimension}}

    def extract_embedding(self, response: Dict) -> List[float]:
        return response["embedding"]
