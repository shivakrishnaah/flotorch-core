from dataclasses import dataclass, field
import json
from storage.db.db_storage import DBStorage
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from embedding.embedding import BaseEmbedding

@dataclass
class VectorStorageSearchItem:
    text: str
    execution_id: str = field(default_factory=None)
    chunk_id: str = field(default_factory=None)
    vectors: List[float] = field(default_factory=[])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self):
        return json.dumps({
            "text": self.text,
            "execution_id": self.execution_id,
            "chunk_id": self.chunk_id,
            "vectors": self.vectors,
            "metadata": self.metadata
        })
@dataclass
class VectorStorageSearchResponse:
    status: bool
    result: VectorStorageSearchItem = field(default_factory=None)
    metadata: Dict[str, Any] = field(default_factory=dict)

    
class VectorStorage(DBStorage, ABC):
    def __init__(self, embedder: Optional[BaseEmbedding] = None):
        self.embedder = embedder
        
    
    @abstractmethod
    def search(self, query: str, knn, hierarchical=False):
        pass

    @abstractmethod
    def embed_query(self, embedding, knn, hierarical=False):
        pass
    
