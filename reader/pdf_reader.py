from storage.storage import StorageProvider
import io
from PyPDF2 import PdfReader

class PDFReader:
    def __init__(self, storage_provider: StorageProvider):
        self.storage_provider = storage_provider

    def read_pdf(self, path):
        text = []
        for data in self.storage_provider.read(path):
            if data is not None:
                text.append(self._read_pdf(data))
        return text

    def _read_pdf(self, data: bytes):
        stream = io.BytesIO(data)
        reader = PdfReader(stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text