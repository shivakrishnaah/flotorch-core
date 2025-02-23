from typing import Any, Dict, List, Tuple
from guardrails.guardrails import BaseGuardRail
from inferencer.inferencer import BaseInferencer


class GuardRailsInferencer(BaseInferencer):
    def __init__(self, base_inferencer: BaseInferencer, base_guardrail: BaseGuardRail):
        self.base_inferencer = base_inferencer
        self.base_guardrail = base_guardrail

    def generate_text(self, user_query: str, context: List[Dict]) -> Tuple[Dict[Any, Any], str]:
        metadata, answer = self.base_inferencer.generate_text(user_query, context)

        guardrail_response = self.base_guardrail.apply_guardrail(answer, 'OUTPUT')
        if guardrail_response['action'] == 'GUARDRAIL_INTERVENED':
            return metadata, guardrail_response['outputs'][0]['text']
        
        return metadata, answer