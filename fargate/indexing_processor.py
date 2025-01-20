from fargate.base_task_processor import BaseFargateTaskProcessor
from logger.global_logger import get_logger

logger = get_logger()

class IndexingProcessor(BaseFargateTaskProcessor):
    """
    Processor for indexing tasks in Fargate.
    """

    def process(self):
        logger.info("Starting indexing process.")
        try:
            exp_config_data = self.input_data.get("experimentConfig", {})
            logger.info(f"Experiment config data: {exp_config_data}")

            # TODO: logic

            output = {"status": "success", "message": "Indexing completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during indexing process: {str(e)}")
            self.send_task_failure(str(e))
