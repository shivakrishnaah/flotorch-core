from .bedrock_embedding import BedRockEmbedding
from chunking.chunking import Chunk
from typing import List, Dict


class CohereEmbedding(BedRockEmbedding):
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__(model_id, region, dimensions, normalize)

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"texts": [chunk.data], "input_type": "search_document"}