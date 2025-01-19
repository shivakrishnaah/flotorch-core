import boto3
from storage.db import DBStorage
from botocore.exceptions import ClientError

class DynamoDB(DBStorage):
    def __init__(self, table_name, region_name='us-east-1'):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)

    def write(self, item: dict):
        try:
            self.table.put_item(Item=item)
            return True
        except ClientError as e:
            print(f"Error writing to DynamoDB: {e}")
            return False

    def read(self, key) -> dict:
        try:
            response = self.table.get_item(Key=key)
            return response.get('Item', None)
        except ClientError as e:
            print(f"Error reading from DynamoDB: {e}")
            return None
    
    def bulk_write(self, items: list):
        with self.table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)
        return True
    