import boto3

from botocore.client import Config

from config.config import get_config


config = get_config()

# Create global S3 client
S3_BUCKET = config.s3_bucket
s3 = boto3.client('s3', config=Config(signature_version='s3v4'))

def get_s3_client() -> boto3.client:
    return s3
