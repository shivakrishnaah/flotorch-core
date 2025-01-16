import logging
import os
from typing import Generator

import boto3

from .storage import StorageProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3StorageProvider(StorageProvider):
    """
    S3 storage provider
    """

    def __init__(self, bucket):

        """
        Initializes the S3Storage class with the specified S3 bucket.
        Args:
            bucket (str): The name of the S3 bucket to interact with.
        Attributes:
            bucket (str): The name of the S3 bucket.
            s3_client (boto3.client): The boto3 client for interacting with S3.
        """

        super().__init__()
        self.bucket = bucket
        self.s3_client = boto3.client('s3')

    def write(self, path, data) -> None:
        """
        Writes data to the specified path in the S3 bucket.
        Args:
            path (str): The path to write the data to in the S3 bucket.
            data (bytes): The data to write to the S3 bucket.
        """
        logger.info(f'Writing data to S3 storage: {data}')
        if not path.endswith("/"):
            key = path
        else:
            key = path + "tmp.data"
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def read(self, path) -> Generator[bytes, None, None]:
        """
        Reads data from the specified path in the S3 bucket.
        Args:
            path (str): The path to read the data from in the S3 bucket.
        Returns:
            Generator[bytes, None, None]: A generator that yields the data read from the S3 bucket.
        """
        logger.info('Reading data from S3 storage')
        if os.path.isdir(path):
            yield from self._read_directory(path)
        else:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=path)
            yield response['Body'].read()

    def _read_directory(self, path):
        """
        Reads all files in the specified directory in the S3 bucket.
        Args:
            path (str): The path to the directory in the S3 bucket.
        Returns:
            Generator[bytes, None, None]: A generator that yields the data read from the S3 bucket.
        """

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                response = self.s3_client.get_object(Bucket=self.bucket, Key=file_path)
                yield response['Body'].read()
