from abc import ABC, abstractmethod
import boto3

class BaseGuardRail(ABC):

    def __init__(self, prompt=True, response=True):
        self.prompt = prompt
        self.response = response
        
    @abstractmethod
    def apply_guardrail(self, content: str,
        source: str = 'INPUT'):
        pass

class BedrockGuardrail(BaseGuardRail):
    def __init__(self, guardrail_id: str, guardrail_version: str, runtime_client = None):
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.runtime_client = runtime_client or boto3.client('bedrock-runtime')
        
    def apply_guardrail(self, content: str,
        source: str = 'INPUT'):
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
        # response = self.runtime_client.apply_guardrail(
        #     guardrail_id=self.guardrail_id,
        #     guardrail_version=self.guardrail_version,
        #     content=[{'text': content}],
        #     source=source
        # )

        # if response['action'] == 'GUARDRAIL_INTERVENED':
        #     assessment