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
            # exp_config_data = self.input_data.get("experimentConfig", {})

            exp_config_data = {
                "kb_data": "s3://flotorch-data-paimon/0f5062ad-7dff-4daa-b924-b5a75a88ffa6/kb_data/medical_abstracts_100_169kb.pdf",
                "chunk_size": 128,
                "chunk_overlap": 5,
                "parent_chunk_size": 512,
                "embedding_model": "amazon.titan-embed-image-v1",
                "aws_region": "us-east-1"
            }

            index_id = "6w6qd_hi_none_none_b_amazontitanimagev1_256_hnsw"

            logger.info(f"Experiment config data: {exp_config_data}")

            kb_data = exp_config_data.get("kb_data")
            kb_data_bucket, kb_data_path = self._get_s3_bucket_and_path(kb_data)
            s3_storage = S3StorageProvider(kb_data_bucket)
            pdf_reader = PDFReader(s3_storage)
            chunking = HieraricalChunker(exp_config_data.get("chunk_size"), exp_config_data.get("chunk_overlap"), exp_config_data.get("parent_chunk_size"))
            embedding = TitanV1Embedding(exp_config_data.get("embedding_model"), exp_config_data.get("aws_region"))
            indexing = Index(pdf_reader, chunking, embedding)
            embeddings_list = indexing.index(kb_data_path)
            open_search_client = OpenSearchClient(config.get_opensearch_host(), config.get_opensearch_port(),
                                           config.get_opensearch_username(), config.get_opensearch_password(),
                                           index_id)
    
            bulk_data = []
            for embedding in embeddings_list.embeddings:
                # TODO See this index also can be included in the to_dict method
                bulk_data.append({"index": {"_index": config.get_opensearch_index()}})
                # below code wont work as Embedding class is not inside the list, embedding array and metadata is extracted earlier from Embeddings class
                # bulk_data.append(embedding.to_json())

                bulk_data.append(embedding.to_json())
            open_search_client.write_bulk(body=bulk_data)
            output = {"status": "success", "message": "Indexing completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during indexing process: {str(e)}")
            self.send_task_failure(str(e))
