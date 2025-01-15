from abc import ABC, abstractmethod
import logging
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageProvider(ABC):

    @abstractmethod
    def write(self, data):
        pass

    @abstractmethod
    def read(self):
        pass

class LocalStorageProvider(StorageProvider):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def write(self, data) -> None:
        logger.info(f'Writing data to local storage: {data}')
        with open(self.path, 'w') as file:
            file.write(data)

    def read(self) -> str:
        logger.info('Reading data from local storage')
        with open(self.path, 'r') as file:
            data = file.read()
        return data
    
    @staticmethod
    def with_path(path):
        return LocalStorageProvider(path)

class S3StorageProvider(StorageProvider):
    def __init__(self, bucket, key):
        super().__init__()
        self.bucket = bucket
        self.key = key
        self.s3_client = boto3.client('s3')

    def write(self, data) -> None:
        logger.info(f'Writing data to S3 storage: {data}')
        self.s3_client.put_object(Bucket=self.bucket, Key=self.key, Body=data)

    def read(self) -> str:
        logger.info('Reading data from S3 storage')
        response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
        data = response['Body'].read().decode('utf-8')
        return data

    @staticmethod
    def with_bucket_and_key(bucket, key):
        return S3StorageProvider(bucket, key)