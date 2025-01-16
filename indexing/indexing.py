from reader.pdf_reader import  PDFReader
from chunking.chunking import BaseChunker
from embedding.embedding import BaseEmbedding


class Index:
    def __init__(self, pdf_reader: PDFReader, chunker: BaseChunker, embedder: BaseEmbedding):
        self.pdf_reader = pdf_reader
        self.chunker = chunker
        self.embedder = embedder

    def index(self, path):
        text = self.pdf_reader.read_pdf(path)
        chunks = self.chunker.chunk_list(text)
        embeddings = self.embedder.embed_list(chunks)
        return embeddings