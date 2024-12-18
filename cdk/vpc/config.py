from pydantic import BaseModel

class InfraConfig(BaseModel):
    project_name: str = "flotorch"
    vpc_cidr: str = "10.0.0.0/16"
    aws_region: str = "us-east-1"
    opensearch_admin_password: str = "Flotorch@123"
    opensearch_admin_user: str = "admin"

infraConfig = InfraConfig()
