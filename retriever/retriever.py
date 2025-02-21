from reader.json_reader import JSONReader
from embedding.embedding import BaseEmbedding
from chunking.chunking import Chunk
from storage.db.vector.vector_storage import VectorStorage
from pydantic import BaseModel
from inferencer.inferencer import BaseInferencer

class Question(BaseModel):
    question: str
    answer: str

    def get_chunk(self) -> Chunk:
        return Chunk(data=self.question)

class Retriever:
    def __init__(self):
        pass

    def retrieve(self, question: Question, path: str, query: str, knn: int, hierarchical=False):
        pass
    # def __init__(self, json_reader: JSONReader, embedding: BaseEmbedding, vector_storage: VectorStorage, inferencer: BaseInferencer) -> None:
    #     self.json_reader = json_reader
    #     self.embedding = embedding
    #     self.vector_storage = vector_storage
    #     self.inferencer = inferencer

    # def retrieve(self, path: str, query: str, knn: int, hierarchical: False):
    #     questions_list = self.json_reader.read_as_model(path, Question)
    #     for question in questions_list:
    #         question_chunk = question.get_chunk()
    #         question_embedding = self.embedding.embed(question_chunk)
    #         query = self.vector_storage.embed_query(question_embedding.embeddings, knn, hierarchical)
    #         context = self.vector_storage.search(query)
    #         metadata, answer = self.inferencer.generate_text(question.question, context)