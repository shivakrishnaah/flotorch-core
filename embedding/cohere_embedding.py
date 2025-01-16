from typing import Dict

from chunking.chunking import Chunk
from .bedrock_embedding import BedRockEmbedding


class CohereEmbedding(BedRockEmbedding):
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__(model_id, region, dimensions, normalize)

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"texts": [chunk.data], "input_type": "search_document"}
