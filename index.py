from chunking.fixedsize_chunking import FixedSizeChunker
from embedding.bedrock_embedding import TitanV2Embedding
from embedding.llama_embedding import LlamaEmbedding
from indexing.indexing import Index
from reader.pdf_reader import PDFReader
from storage.db.vector import OpenSearchClient
from storage.local_storage import LocalStorageProvider


def main():
    # storage_provider = S3StorageProvider('flotorch-data-677276078734-us-east-1-qgp1f5')
    storage_provider = LocalStorageProvider()
    pdf_reader = PDFReader(storage_provider)
    chunker = FixedSizeChunker(128, 5)
    #embedder = LlamaEmbedding('llama3.3')
    embedder = TitanV2Embedding('amazon.titan-embed-text-v2:0', 'us-east-1', 256, True)
    index = Index(pdf_reader, chunker, embedder)
    # embeddings = index.index(path="0b48bc48-8a1a-42bc-9ee4-aa53380bb58d/kb_data/kb.pdf")
    embeddings = index.index(path="/Users/shivakrishna/Downloads/medical_abstracts_100.pdf")



if __name__ == "__main__":
    main()
