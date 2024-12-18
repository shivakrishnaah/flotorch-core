from pydantic import BaseModel
from typing import Dict, List, Optional

class ExperimentRequest(BaseModel):
    config: Dict
    gt_data: str
    kb_data: str
    region: str
    name: str

class ExperimentResponse(BaseModel):
    status: str
    experiment_ids: List[str]

class ExecutionResponse(BaseModel):
    status: str
    execution_id: str