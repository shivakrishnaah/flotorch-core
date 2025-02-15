from vector_storage import VectorStorage
from guardrails.guardrails import BaseGuardRail

class GuardRailsVectorStorage(VectorStorage):

    def __init__(self, vectorStorage: VectorStorage, base_guardrail: BaseGuardRail, 
                 apply_prompt=True, apply_response=True):
        self.vectorStorage = vectorStorage
        self.base_guardrail = base_guardrail
        self.apply_prompt = apply_prompt
        self.apply_response = apply_response
    
    def search(self, query: str, knn: int, hierarchical=False):
        if self.apply_prompt:
            query = self.base_guardrail.apply_guardrail(query, 'INPUT')
            results = self.vectorStorage.search(query, knn, hierarchical)
        if self.apply_response:
            results = self.base_guardrail.apply_guardrail(results, 'OUTPUT')
        return results
    
    def embed_query(self, embedding, knn, hierarical=False):
        self.vectorStorage.embed_query(embedding, knn, hierarical)