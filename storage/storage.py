from abc import ABC, abstractmethod
from typing import Generator
import logging
import boto3
import os
import io
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageProvider(ABC):

    @abstractmethod
    def write(self, path, data) -> None:
        pass

    @abstractmethod
    def read(self, path) -> Generator[bytes, None, None]:
        pass

    def read_as_string(self, path) -> Generator[str, None, None]:
        yield from (data.decode('utf-8') for data in self.read(path))