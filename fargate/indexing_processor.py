from chunking.fixedsize_chunking import FixedSizeChunker
from chunking.hierarical_chunking import HieraricalChunker
from embedding.titanv2_embedding import TitanV2Embedding
from embedding.titanv1_embedding import TitanV1Embedding
from fargate.base_task_processor import BaseFargateTaskProcessor
from logger.global_logger import get_logger
from reader.pdf_reader import PDFReader
from storage.db.vector import OpenSearchClient
from storage.s3_storage import S3StorageProvider
from indexing.indexing import Index
from config.config import Config
from config.env_config_provider import EnvConfigProvider
from storage.storage_provider_factory import StorageProviderFactory
from embedding.embedding_registry import embedding_registry
from chunking.chunking_provider_factory import ChunkingFactory


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

            # exp_config_data = {
            #     "kb_data": "file://C:/Projects/refactor/medical_abstracts_100_169kb.pdf",
            #     "chunk_size": 128,
            #     "chunk_overlap": 5,
            #     "parent_chunk_size": 512,
            #     "embedding_model": "amazon.titan-embed-image-v1",
            #     "aws_region": "us-east-1",
            #     "chunking_strategy": "Fixed",
            # }

            index_id = exp_config_data.get("index_id") # "local-index-1024"

            logger.info(f"Experiment config data: {exp_config_data}")

            kb_data = exp_config_data.get("kb_data")
            storage = StorageProviderFactory.create_storage_provider(kb_data)
            kb_data_path = storage.get_path(kb_data)
            pdf_reader = PDFReader(storage)

            chunking = ChunkingFactory.create_chunker(
                exp_config_data.get("chunking_strategy"), 
                exp_config_data.get("chunk_size"), 
                exp_config_data.get("chunk_overlap"),
                exp_config_data.get("parent_chunk_size", None)
            )
            
            embedding_class = embedding_registry.get_model(exp_config_data.get("embedding_model"))
            embedding = embedding_class(exp_config_data.get("embedding_model"), exp_config_data.get("aws_region"))
            indexing = Index(pdf_reader, chunking, embedding)
            embeddings_list = indexing.index(kb_data_path)

            open_search_client = OpenSearchClient(
                config.get_opensearch_host(), 
                config.get_opensearch_port(),
                config.get_opensearch_username(), 
                config.get_opensearch_password(),
                index_id
            )
    
            bulk_data = []
            for embedding in embeddings_list.embeddings:
                # TODO See this index also can be included in the to_dict method
                bulk_data.append({"index": {"_index": index_id}})
                bulk_data.append(embedding.to_json())
            open_search_client.write_bulk(body=bulk_data)
            output = {"status": "success", "message": "Indexing completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during indexing process: {str(e)}")
            self.send_task_failure(str(e))
