from fargate.base_task_processor import BaseFargateTaskProcessor
from logger.global_logger import get_logger
from config.config import Config
from config.env_config_provider import EnvConfigProvider
from reader.json_reader import JSONReader
from storage.s3_storage import S3StorageProvider
from retriever.retriever import Retriever
from storage.db.vector.open_search import OpenSearchClient
from embedding.titanv2_embedding import TitanV2Embedding


logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)

class RetrieverProcessor(BaseFargateTaskProcessor):
    """
    Processor for retriever tasks in Fargate.
    """

    def process(self):
        logger.info("Starting retriever process.")
        try:
            #logic
            exp_config_data = {
                "kb_data": "s3://flotorch-data-paimon/0f5062ad-7dff-4daa-b924-b5a75a88ffa6/kb_data/medical_abstracts_100_169kb.pdf",
                "chunk_size": 128,
                "chunk_overlap": 5,
                "parent_chunk_size": 512,
                "embedding_model": "amazon.titan-embed-text-v2:0",
                "aws_region": "us-east-1",
                "gt_data": "s3://flotorch-data-paimon/0f5062ad-7dff-4daa-b924-b5a75a88ffa6/gt_data/gt.json",
                "knn_num": 5,
                "vector_dimension": 1024
            }

            index_id = "local-index-1024"

            gt_data = exp_config_data.get("gt_data")
            gt_data_bucket, gt_data_path = self._get_s3_bucket_and_path(gt_data)
            s3_storage = S3StorageProvider(gt_data_bucket)
            json_reader = JSONReader(s3_storage)
            embedding = TitanV2Embedding(exp_config_data.get("embedding_model"), exp_config_data.get("aws_region"), exp_config_data.get("vector_dimension"))

            open_search_client = OpenSearchClient(config.get_opensearch_host(), config.get_opensearch_port(),
                                           config.get_opensearch_username(), config.get_opensearch_password(),
                                           index_id)

            retriever = Retriever(json_reader, embedding, open_search_client)
            retriever.retrieve(gt_data_path, "What is the patient's name?", exp_config_data.get("knn_num"))
            
            output = {"status": "success", "message": "Retriever completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during retriever process: {str(e)}")
            self.send_task_failure(str(e))
