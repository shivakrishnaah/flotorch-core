from .config_provider import ConfigProvider

class Config:
    """
    Main configuration class that uses a configuration provider.
    """

    def __init__(self, provider: ConfigProvider):
        """
        Initializes the Config class with the specified provider.
        Args:
            provider (ConfigProvider): The configuration provider.
        """
        self.provider = provider

    def get_region(self) -> str:
        """
        Retrieves the AWS region from the configuration provider.
        Returns:
            str: The AWS region.
        Raises:
            ValueError: If the region is not set.
        """
        region = self.provider.get("AWS_REGION", "us-east-1")
        if not region:
            raise ValueError("AWS region is not set. Value not presnt in configuration")
        return region
    
    def get_opensearch_host(self) -> str:
        """
        Retrieves the OpenSearch host from the configuration provider.
        """
        open_search_host = self.provider.get("OPENSEARCH_HOST", "localhost")
        if not open_search_host:
            raise ValueError("OpenSearch host is not set. Value not presnt in configuration")
        return open_search_host

    def get_opensearch_port(self) -> int:
        """
        Retrieves the OpenSearch port from the configuration provider.
        """
        open_search_port = int(self.provider.get("OPENSEARCH_PORT", 5000))
        if not open_search_port:
            raise ValueError("OpenSearch port is not set. Value not presnt in configuration")
        return open_search_port
    
    def get_opensearch_username(self) -> str:
        """
        Retrieves the OpenSearch username from the configuration provider.
        """
        open_search_username = self.provider.get("OPENSEARCH_USERNAME")
        if not open_search_username:
            raise ValueError("OpenSearch username is not set. Value not presnt in configuration")
        return open_search_username

    def get_opensearch_password(self) -> str:
        """
        Retrieves the OpenSearch password from the configuration provider.
        """
        open_search_password = self.provider.get("OPENSEARCH_PASSWORD")
        if not open_search_password:
            raise ValueError("OpenSearch password is not set. Value not presnt in configuration")
        return open_search_password

    def get_opensearch_index(self) -> str:
        """
        Retrieves the OpenSearch index name from the configuration provider.
        """
        open_search_index = self.provider.get("OPENSEARCH_INDEX")
        if not open_search_index:
            raise ValueError("OpenSearch index is not set. Value not presnt in configuration")
        return open_search_index

    def get_task_token(self) -> str:
        """
        Retrieves the task token from the configuration provider.
        """
        task_token = self.provider.get("TASK_TOKEN")
        if not task_token:
            raise ValueError("task token is not set. Value not presnt in configuration")
        return task_token

    def get_fargate_input_data(self) -> str:
        """
        Retrieves the input data for fargate handlers from the configuration provider.
        """
        input_data = self.provider.get("INPUT_DATA", {})
        if not input_data:
            raise ValueError("input data is not set. Value not present in configuration")
        return input_data