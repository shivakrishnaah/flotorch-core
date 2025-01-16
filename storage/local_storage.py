from requests.compat import chardet

from .storage import StorageProvider
from typing import Generator
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalStorageProvider(StorageProvider):
    def __init__(self):
        super().__init__()

    def write(self, path, data) -> None:
        logger.info(f'Writing data to local storage: {data}')
        if os.path.isdir(path):
            path = os.path.join(path, 'tmp.data')
        with open(path, 'w') as file:
            file.write(data)

    def read(self, path) -> Generator[bytes, None, None]:
        logger.info('Reading data from local storage')
        if os.path.isdir(path):
            yield from self._read_directory(path)
        else:
            with open(path, "rb") as file:
                yield file.read()

    def _read_directory(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                with open(path, "rb") as file:
                    return file.read()
