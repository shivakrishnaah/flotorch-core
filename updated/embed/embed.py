from abc import ABC, abstractmethod
from typing import List, Dict
from chunk.chunk import Chunk
import boto3
import json

class EmbeddingMetadata:
    def __init__(self, input_tokens: int, latency_ms: int):
        self.input_tokens = input_tokens
        self.latency_ms = latency_ms

class Embeddings:
    def __init__(self, embeddings: List[List[float]], metadata: EmbeddingMetadata):
        self.embeddings = embeddings
        self.metadata = metadata

class BaseEmbed(ABC):
    def __init__(self, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__()
        self.dimension = dimensions
        self.normalize = normalize
    
    @abstractmethod
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        pass
    
    @abstractmethod
    def embed(self, chunk: Chunk) -> Embeddings:
        pass

    def embed(self, chunks: List[Chunk]) -> List[Embeddings]:
        return [self.embed(chunk) for chunk in chunks]

class BedRockEmbed(BaseEmbed):
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__(dimensions, normalize)
        self.model_id = model_id
        self.region = region
        self._application_json = "application/json"
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
    
    def embed(self, chunk: Chunk) -> Embeddings:
        payload = self._prepare_chunk(chunk)
        response = self._invoke_model(payload)
        metadata = self._extract_metadata(response)
        model_response = self._parse_model_response(response)
        return Embeddings(embeddings=self.extract_embedding(model_response), 
                            metadata=metadata)

    def _invoke_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.invoke_model(
            modelId=self.model_id,
            contentType=self._application_json,
            accept=self._application_json,
            body=json.dumps(payload)
        )

    def _extract_metadata(self, response: Dict[str, Any]) -> EmbeddingMetadata:
        if not response or 'ResponseMetadata' not in response:
            return EmbeddingMetadata(input_tokens=0, latency_ms=0)

        headers = response['ResponseMetadata'].get('HTTPHeaders', {})
        return EmbeddingMetadata(
            input_tokens=headers.get('x-amzn-bedrock-input-token-count', ''),
            latency_ms=headers.get('x-amzn-bedrock-invocation-latency', '')
        )

    def _parse_model_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if 'body' not in response:
            raise ValueError("Invalid response format: 'body' not found.")
        return json.loads(response['body'].read())
    
    def extract_embedding(self, response: Dict[str, Any]) -> List[float]:
        return response["embeddings"][0]

class CohereEmbed(BedRockEmbed):
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"texts": [chunk.data], "input_type": "search_document"}

class TitanV1Embed(BedRockEmbed):
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"inputText": chunk.data, "embeddingConfig" : {"outputEmbeddingLength" : self.dimension}}

    def extract_embedding(self, response: Dict) -> List[float]:
        return response["embedding"]
    
class TitanV2Embed(BedRockEmbed):
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"inputText": chunk.data, "dimensions": self.dimension, "normalize": self.normalize}

    def extract_embedding(self, response: Dict) -> List[float]:
        return response["embedding"]

        