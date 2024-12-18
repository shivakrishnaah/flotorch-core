from core.inference import InferencerFactory
from typing import Dict, List, Tuple
from config.experimental_config import ExperimentalConfig
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class InferenceProcessor:
    """Processor for embedding text chunks."""

    def __init__(self, experimentalConfig : ExperimentalConfig) -> None:
        self.experimentalConfig = experimentalConfig
        self.inferencer = InferencerFactory.create_inferencer(experimentalConfig)

    def generate_text(self, user_query: str, context: List[Dict], default_prompt: str, **kwargs) -> str:
        try:
            answer = self.inferencer.generate_text(
                user_query=user_query,
                context = context,
                default_prompt = default_prompt,
                experiment_config = self.experimentalConfig
            )
            return answer
        except Exception as e:
            logger.error(f"Error generating text with Inferencer: {str(e)}")
            raise
