from indexing.indexing import Index
from storage.s3_storage import S3StorageProvider
from storage.local_storage import LocalStorageProvider
from reader.pdf_reader import PDFReader
from chunking.fixedsize_chunking import FixedSizeChunker
from embedding.bedrock_embedding import TitanV2Embedding

def main():
    #storage_provider = S3StorageProvider('flotorch-data-677276078734-us-east-1-qgp1f5')
    storage_provider = LocalStorageProvider()
    pdf_reader = PDFReader(storage_provider)
    chunker = FixedSizeChunker(128, 5)
    embedder = TitanV2Embedding('amazon.titan-embed-text-v2:0', 'us-east-1', 256, True)
    index = Index(pdf_reader, chunker, embedder)
    #embeddings = index.index(path="0b48bc48-8a1a-42bc-9ee4-aa53380bb58d/kb_data/kb.pdf")
    embeddings = index.index(path="/Users/shivakrishna/Downloads/kb.pdf")
    print(embeddings)

if __name__ == "__main__":
    main()
