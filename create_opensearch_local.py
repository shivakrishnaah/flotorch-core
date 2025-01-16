import os

import boto3
from config.config import get_config
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth


def create_index(client, index_name):
    vector_field = 'vectors'

    # Delete indexing if it exists
    if client.indices.exists(index=index_name):
        print(f"Deleting existing indexing {index_name}")
        client.indices.delete(index=index_name)

    index_body = {
        "settings": {
            "indexing": {
                "knn": True,
                "knn.algo_param.ef_search": 512
            }
        },
        "mappings": {
            "properties": {
                vector_field: {  # This will use the field name from your .env file
                    "type": "knn_vector",
                    "dimension": 3584,  # Titan embedding model dimension
                    "method": {
                        "name": 'hnsw',
                        "engine": "nmslib",
                        "space_type": "l2",
                        "parameters": {}
                    }
                },
                "text": {
                    "type": "text"
                }
            }
        }
    }

    # Create the indexing
    print(f"Creating indexing {index_name}")
    response = client.indices.create(index=index_name, body=index_body)
    print("Index creation response:", response)

    # Verify the mapping
    mapping = client.indices.get_mapping(index=index_name)
    print("\nIndex mapping:", mapping)


config = get_config()

# OpenSearch client setup
profile_name = os.getenv('profile_name')
host = os.getenv('opensearch_host')
region = os.getenv('aws_region')
service = 'es'
username = os.getenv('opensearch_username')
password = os.getenv('opensearch_password')
credentials = boto3.Session(profile_name=profile_name).get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

if config.opensearch_serverless:
    try:
        # Get credentials from the Lambda role
        credentials = boto3.Session().get_credentials()
        # Create AWS V4 Signer Auth for OpenSearch Serverless
        auth = AWSV4SignerAuth(credentials, region, 'aoss')

        # Initialize OpenSearch client for serverless
        client = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_auth=('admin', 'Test@password42'),
            use_ssl=True,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
            # Add required headers for OpenSearch Serverless
            headers={
                'host': host
            }
        )

    except Exception as e:
        # logger.error(f"Failed to initialize OpenSearch Serverless client: {str(e)}", exc_info=True)
        raise
else:
    client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        http_auth=('admin', 'Test@password42'),
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True
    )

# Index configuration
indices = [
    'local-indexing-3584'
]

for index in indices:
    create_index(client, index)
