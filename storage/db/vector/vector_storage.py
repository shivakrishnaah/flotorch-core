from storage.db.db_storage import DBStorage
from abc import ABC, abstractmethod
from typing import List, Optional
from embedding.embedding import BaseEmbedding

class VectorStorage(DBStorage, ABC):
    def __init__(self, embedder: Optional[BaseEmbedding] = None):
        self.embedder = embedder
        
    
    @abstractmethod
    def search(self, query: str, knn, hierarchical=False):
        pass

    @abstractmethod
    def embed_query(self, embedding, knn, hierarical=False):
        pass
    
