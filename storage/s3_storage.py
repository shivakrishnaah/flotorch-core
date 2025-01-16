from typing import Generator
import logging
import boto3
import os
from .storage import StorageProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3StorageProvider(StorageProvider):
    def __init__(self, bucket):
        super().__init__()
        self.bucket = bucket
        self.s3_client = boto3.client('s3')

    def write(self, path, data) -> None:
        logger.info(f'Writing data to S3 storage: {data}')
        if not path.endswith("/"):
            key = path
        else:
            key = path + "tmp.data"
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def read(self, path) -> Generator[bytes, None, None]:
        logger.info('Reading data from S3 storage')
        if os.path.isdir(path):
            yield from self._read_directory(path)
        else:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=path)
            yield response['Body'].read()

    def _read_directory(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                response = self.s3_client.get_object(Bucket=self.bucket, Key=file_path)
                yield response['Body'].read().decode('utf-8')
