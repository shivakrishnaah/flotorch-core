from chunking.fixedsize_chunking import FixedSizeChunker
from embedding.titanv2_embedding import TitanV2Embedding
from fargate.base_task_processor import BaseFargateTaskProcessor
from logger.global_logger import get_logger
from reader.pdf_reader import PDFReader
from storage.db.vector import OpenSearchClient
from storage.s3_storage import S3StorageProvider
from indexing.indexing import Index
from config.config import Config
from config.env_config_provider import EnvConfigProvider


logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)

class IndexingProcessor(BaseFargateTaskProcessor):
    """
    Processor for indexing tasks in Fargate.
    """

    def process(self):
        logger.info("Starting indexing process.")
        try:
            exp_config_data = self.input_data.get("experimentConfig", {})
            logger.info(f"Experiment config data: {exp_config_data}")

            kb_data = exp_config_data.get("kb_data")
            kb_data_bucket, kb_data_path = self._get_s3_bucket_and_path(kb_data)
            s3_storage = S3StorageProvider(kb_data_bucket)
            pdf_reader = PDFReader(s3_storage)
            chunking = FixedSizeChunker(exp_config_data.get("chunk_size"), exp_config_data.get("chunk_overlap"))
            embedding = TitanV2Embedding(exp_config_data.get("embedding_model"), exp_config_data.get("aws_region"))
            indexing = Index(pdf_reader, chunking, embedding)
            embeddings_list = indexing.index(kb_data_path)
            open_search_client = OpenSearchClient(config.get_opensearch_host(), config.get_opensearch_port(),
                                           config.get_opensearch_username(), config.get_opensearch_password(),
                                           config.get_opensearch_index())
    
            bulk_data = []
            for embedding in embeddings_list.embeddings:
                # TODO See this index also can be included in the to_dict method
                bulk_data.append({"index": {"_index": config.get_opensearch_index()}})
                bulk_data.append(embedding.to_dict())
            open_search_client.write_bulk(body=bulk_data)
            output = {"status": "success", "message": "Indexing completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during indexing process: {str(e)}")
            self.send_task_failure(str(e))
