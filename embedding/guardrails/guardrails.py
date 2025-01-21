from embedding.embedding import BaseEmbedding
from typing import List, Dict, Optional, Any
import boto3
from chunking.chunking import Chunk
from botocore.exceptions import ClientError
from .guardrail_config import GuardrailCreateConfig
from embedding.embedding import Embeddings, EmbeddingList


class GuardrailsEmbedding:

    def __init__(self, base_embedding: BaseEmbedding, 
                 guardrail_id: str, 
                 guardrail_version: str):
        self.base_embedding = base_embedding
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.bedrock_client = boto3.client('bedrock')
        self.runtime_client = boto3.client('bedrock-runtime')
    
    def apply_guardrail(self, content: str,
        source: str = 'INPUT'):
        """
        Apply a guardrail to content using Amazon Bedrock ApplyGuardrails API
        
        Args:
            content (str): The content to validate against the guardrail
            source (str): The source of the content ('INPUT' or 'OUTPUT')            
        Returns:
            Dict: Response from the ApplyGuardrails API
            
        Example response structure:
        {
            'results': [{
                'status': 'ALLOWED'|'FILTERED'|'DENIED',
                'statusMessage': 'string',
                'violations': [{
                    'policyName': 'string',
                    'violationType': 'string',
                    'violationMessage': 'string'
                }]
            }],
            'responseMetadata': {
                'requestId': 'string',
                'attempts': 123,
                'totalRetryDelay': 123.0
            }
        }
        """
        try:
            request_params = {
                'guardrailIdentifier': self.guardrail_id,
                'guardrailVersion': self.guardrail_version,
                'source': source,
                'content': content
            }
            response = self.runtime_client.apply_guardrail(**request_params)
            return response

        except ClientError as e:
            print(f"Error applying guardrail: {str(e)}")
            raise


    def create_guardrail(
            self,
            guardrail_config: GuardrailCreateConfig
    ) -> Dict:
        """
        Create a comprehensive guardrail using Amazon Bedrock
        
        Args:
            name: Name of the guardrail
            description: Description of the guardrail
            content_policy: Content policy configuration
            topic_policy: Topic policy configuration
            word_policy: Word policy configuration
            sensitive_info_policy: Sensitive information policy configuration
            contextual_grounding_policy: Contextual grounding policy configuration
            input_filter: Input filtering configuration
            output_filter: Output filtering configuration
        """
        try:

            response = self.bedrock_client.create_guardrail(**guardrail_config)
            return response

        except ClientError as e:
            print(f"Error creating guardrail: {str(e)}")
            raise

    """
    Embeds the chunk.
    :param chunk: The chunk to be embedded.
    :return: The embeddings.
    """
    def embed(self, chunk: Chunk) -> Embeddings:
        guardrail_response = self.apply_guardrail(content=[{'text': chunk.data}])
        if guardrail_response['action'] == 'GUARDRAIL_INTERVENED':
            # assessment = guardrail_response.get('assessments', [])
            modified_text = ' '.join(output['text'] for output in guardrail_response['outputs'])
            self.base_embedding.embed(modified_text)

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
            embedding_list.append(embedding)
        return embedding_list
