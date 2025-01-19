import os
from opensearchpy import OpenSearch
from storage.db.db_storage import DbStorage
from typing import List

"""
This class is responsible for storing the data in the OpenSearch.
"""

class OpenSearchClient(DbStorage):
    def __init__(self, host, port, username, password, index, use_ssl=True, verify_certs=True, ssl_assert_hostname=False, ssl_show_warn=False):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.index = index
        
        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_auth=(self.username, self.password),
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_assert_hostname=ssl_assert_hostname,
            ssl_show_warn=ssl_show_warn,
        )
    
    def write(self, body):
        return self.client.index(index=self.index, body=body)
    
    def read(self, body):
        return self._search(self.index, body)
    
    def write_bulk(self, body: List[dict]):
        return self.client.bulk(body=body)

    def _search(self, index, body):
        return self.client.search(index=index, body=body)