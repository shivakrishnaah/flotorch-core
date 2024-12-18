from config.config import get_config
from core.dynamodb import DynamoDBOperations
from ..orchestrator import StepFunctionOrchestrator

class Database:
    def __init__(self):
        self.config = get_config()
        self.execution_db = None
        self.experiment_db = None
        self.question_metrics_db = None
        self.execution_model_invocations_db = None
        self.step_function_orchestrator = None

    def initialize(self):
        """Initialize all database connections and services"""
        if not self.execution_db:
            self.execution_db = DynamoDBOperations(
                self.config.execution_table,
                region=self.config.aws_region
            )
        if not self.experiment_db:
            self.experiment_db = DynamoDBOperations(
                self.config.experiment_table,
                region=self.config.aws_region
            )
        if not self.question_metrics_db:
            self.question_metrics_db = DynamoDBOperations(
                self.config.experiment_question_metrics_table,
                region=self.config.aws_region
            )
        if not self.execution_model_invocations_db:
            self.execution_model_invocations_db = DynamoDBOperations(
                self.config.execution_model_invocations_table,
                region=self.config.aws_region
            )
        if not self.step_function_orchestrator:
            self.step_function_orchestrator = StepFunctionOrchestrator()

# Create a global instance
db = Database()

# Dependency functions
def get_execution_db() -> DynamoDBOperations:
    return db.execution_db

def get_experiment_db() -> DynamoDBOperations:
    return db.experiment_db

def get_question_metrics_db() -> DynamoDBOperations:
    return db.question_metrics_db

def get_execution_model_invocations_db() -> DynamoDBOperations:
    return db.execution_model_invocations_db

def get_step_function_orchestrator() -> StepFunctionOrchestrator:
    return db.step_function_orchestrator
