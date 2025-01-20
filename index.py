from chunking.fixedsize_chunking import FixedSizeChunker
from embedding.bedrock_embedding import TitanV2Embedding
# from embedding.llama_embedding import LlamaEmbedding
from indexing.indexing import Index
from reader.pdf_reader import PDFReader
from storage.db.vector import OpenSearchClient
from storage.local_storage import LocalStorageProvider
from config.env_config_provider import EnvConfigProvider
from config.config import Config


def main():
    env_config_provider = EnvConfigProvider()
    config = Config(env_config_provider)
    # storage_provider = S3StorageProvider('flotorch-data-677276078734-us-east-1-qgp1f5')
    storage_provider = LocalStorageProvider()
    pdf_reader = PDFReader(storage_provider)
    chunker = FixedSizeChunker(128, 5)
    #embedder = LlamaEmbedding('llama3.3')
    embedder = TitanV2Embedding('amazon.titan-embed-text-v2:0', config.get_region(), 256, True)
    index = Index(pdf_reader, chunker, embedder)
    # embeddings = index.index(path="0b48bc48-8a1a-42bc-9ee4-aa53380bb58d/kb_data/kb.pdf")
    embeddings = index.index(path="/Users/shivakrishna/Downloads/medical_abstracts_100.pdf")

    open_search_client = OpenSearchClient(
        host=config.get_opensearch_host(),
        port=config.get_opensearch_port(),
        username=config.get_opensearch_username(),
        password=config.get_opensearch_password(),
        index=config.get_opensearch_index()
    )

    # should this be included in Utils?
    bulk_data = []
    for embedding in embeddings.embeddings:
        bulk_data.append({"index": {"_index": config.get_opensearch_index()}})
        bulk_data.append(embedding.to_dict())

    open_search_client.write_bulk(body=bulk_data)


if __name__ == "__main__":
    main()
