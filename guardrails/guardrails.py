from abc import ABC, abstractmethod
import boto3
from utils.otel_util import trace_and_log_metrics

class BaseGuardRail(ABC):

    def __init__(self, prompt=True, response=True):
        self.prompt = prompt
        self.response = response
        
    @abstractmethod
    def apply_guardrail(self, text: str, source: str = 'INPUT'):
        pass

class BedrockGuardrail(BaseGuardRail):
    def __init__(self, guardrail_id: str, guardrail_version: str, runtime_client = None):
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.runtime_client = runtime_client or boto3.client('bedrock-runtime')

    @trace_and_log_metrics    
    def apply_guardrail(self, text: str,
        source: str = 'INPUT'):
        try:
            request_params = {
                'guardrailIdentifier': self.guardrail_id,
                'guardrailVersion': self.guardrail_version,
                'source': source,
                'content': [{"text": {"text": text}}]
            }
            response = self.runtime_client.apply_guardrail(**request_params)
            return response
        except Exception as e:
            print(f"Error applying guardrail: {str(e)}")
            raise