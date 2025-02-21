from embedding.embedding import BaseEmbedding
from typing import List, Dict
from chunking.chunking import Chunk
from embedding.embedding import Embeddings, EmbeddingList
from guardrails.guardrails import BaseGuardRail


class GuardrailsEmbedding(BaseEmbedding):

    def __init__(self, base_embedding: BaseEmbedding, 
                 base_guardrail: BaseGuardRail):
        super().__init__(base_embedding.dimension, base_embedding.normalize)
        self.base_embedding = base_embedding
        self.base_guardrail = base_guardrail

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return self.base_embedding._prepare_chunk(chunk)

    """
    Embeds the chunk.
    :param chunk: The chunk to be embedded.
    :return: The embeddings.
    """
    def embed(self, chunk: Chunk) -> Embeddings | None:
        guardrail_response = self.base_guardrail.apply_guardrail(content=chunk.data)
        # Exception what if we got ClientError from the above method
        if guardrail_response['action'] == 'GUARDRAIL_INTERVENED':
            # assessment = guardrail_response.get('assessments', [])
            modified_text = ' '.join(output['text'] for output in guardrail_response['outputs'])
            chunk.data = modified_text
            return self.base_embedding.embed(chunk)
        return None

    """
    Embeds the list of chunks.
    :param chunks: The list of chunks to be embedded.
    :return: The list of embeddings.
    """
    def embed_list(self, chunks: List[Chunk]) -> EmbeddingList:
        embedding_list = EmbeddingList()
        if not isinstance(chunks, list):
            return embedding_list.append(self.embed(chunks))
        for chunk in chunks:
            embedding = self.embed(chunk)
            if not embedding is None:
                embedding_list.append(embedding)
        return embedding_list
