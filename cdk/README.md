# Flotorch Infrastructure

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Infrastructure Components](#infrastructure-components)
   - [VPC Stack](#vpc-stack)
   - [State Machine Stack](#state-machine-stack)
   - [App Stack](#app-stack)
6. [Deployment](#deployment)
   - [Deploy New Infrastructure](#deploy-new-infrastructure)
   - [Destroy Infrastructure](#destroy-infrastructure)
7. [Resource Management](#resource-management)
8. [Environment Variables](#environment-variables)
9. [Dependencies](#dependencies)
10. [Prefrred Deployment OS](#preferred-deployment-os)
11. [Troubleshooting](#troubleshooting)

## Overview
This CDK project defines the AWS infrastructure for the Flotorch application using AWS CDK in Python. The infrastructure is designed with a suffix-based deployment strategy, allowing multiple isolated deployments to coexist in the same AWS account.

## Prerequisites

### Required Tools
- Docker 20.10.x or later
- AWS CLI v2
- Python 3.8+
- Node.js 16.x
- AWS CDK CLI 2.x
- jq (JSON processor)

### AWS Account Requirements
- AWS Account with administrative access
- AWS credentials configured locally
- Sufficient permissions to create:
  - VPC and networking resources
  - DynamoDB tables
  - ECR repositories
  - App Runner services
  - IAM roles and policies
  - S3 buckets
  - Step Functions
  - OpenSearch domains

### Automatic Setup
Install all prerequisites automatically:
```bash
./deploy.sh
```
The `deploy.sh` script now supports MacOS, Ubuntu, and Amazon Linux 2, automatically detecting the OS and using the appropriate package manager (Homebrew for MacOS, apt for Ubuntu, yum for Amazon Linux).

## Project Structure
```
cdk/
├── app/
│   └── app_stack.py         # App Runner service definition
├── vpc/
│   ├── vpc_stack.py         # VPC and core infrastructure
│   └── config.py           # VPC configuration
├── state_machine/
│   └── state_machine_stack.py  # Step Functions and Lambda definitions
├── app.py                   # CDK app entry point
├── deploy.sh               # Deployment automation script
├── requirements.txt        # Python dependencies
├── cdk.json               # CDK configuration
└── README.md              # Documentation
```

## Infrastructure Components

### VPC Stack
The VPC stack (`vpc_stack.py`) creates the following resources:

#### Networking
- VPC with public and private subnets
  ```python
  vpc = ec2.Vpc(
      self, "FlotorchVPC",
      max_azs=2,
      nat_gateways=1,
      subnet_configuration=[
          ec2.SubnetConfiguration(
              name="Public",
              subnet_type=ec2.SubnetType.PUBLIC,
              cidr_mask=24
          ),
          ec2.SubnetConfiguration(
              name="Private",
              subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
              cidr_mask=24
          )
      ]
  )
  ```
- NAT Gateways for private subnet internet access
- Security Groups for various services

#### OpenSearch Domain
- Version: OpenSearch 2.15
- Instance Type: r7g.large.search
- Data Nodes: 3
- Storage: 100GB GP3 EBS volumes
- Security:
  - Node-to-node encryption
  - Encryption at rest
  - HTTPS enforcement
  - Fine-grained access control
  - VPC deployment
  - Custom domain name with table suffix

Example Configuration:
```python
opensearch_domain = opensearch.Domain(
    self,
    "FlotorchOpenSearch",
    version=opensearch.EngineVersion.OPENSEARCH_2_15,
    domain_name=f"flotorch-{table_suffix}",
    vpc=vpc,
    vpc_subnets=[ec2.SubnetSelection(
        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
    )],
    capacity=opensearch.CapacityConfig(
        data_node_instance_type="r7g.large.search",
        data_nodes=3
    ),
    ebs=opensearch.EbsOptions(
        volume_size=100,
        volume_type=ec2.EbsDeviceVolumeType.GP3
    ),
    fine_grained_access_control=opensearch.AdvancedSecurityOptions(
        master_user_name="admin",
        master_user_password=master_user_password
    )
)
```

#### DynamoDB Tables
- Execution Table
  ```python
  execution_table = dynamodb.Table(
      self,
      "ExecutionTable",
      partition_key=dynamodb.Attribute(
          name="id",
          type=dynamodb.AttributeType.STRING
      ),
      billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
  )
  ```
- Experiment Table
- Metrics Table
- Model Invocations Table

#### S3 Buckets
- Data bucket for storing application data
  ```python
  data_bucket = s3.Bucket(
      self,
      "DataBucket",
      bucket_name=f"flotorch-data-{account_id}-{region}-{table_suffix}",
      versioned=True,
      encryption=s3.BucketEncryption.S3_MANAGED,
      removal_policy=RemovalPolicy.DESTROY
  )
  ```
- Versioning enabled
- Encryption enabled

#### ECS Configuration
- Fargate Task Definitions
  ```python
  task_definition = ecs.FargateTaskDefinition(
      self,
      f"FlotorchTask-{table_suffix}",
      memory_limit_mib=32768,
      cpu=8192,
      task_role=task_role
  )
  ```
- Memory: 32GB
- CPU: 8 vCPUs
- Task Role with necessary permissions

### State Machine Stack
The State Machine stack (`state_machine_stack.py`) defines:

#### Step Functions State Machine
- Parallel execution of experiments
- Map state with failure tolerance:
  ```python
  experiments_map = sfn.Map(
      self,
      "ExperimentsMap",
      max_concurrency=10,
      items_path=sfn.JsonPath.string_at("$.Items"),
      parameters={
          "execution_id.$": "$.execution_id",
          "experiment.$": "$$.Map.Item.Value"
      },
      result_path="$.experiments_results",
      tolerated_failures=10000
  )
  ```
- Maximum concurrency: 10
- Continues processing even if individual experiments fail
- Lambda function configurations:
  - Memory: 1024MB
  - Timeout: 300 seconds
  - VPC integration
  - Security group access

### App Stack
The App stack (`app_stack.py`) provisions:

#### App Runner Service
```python
webapp_service = apprunner.CfnService(
    self,
    f"FlotorchWebApp-{table_suffix}",
    service_name=f"flotorch-webapp-{table_suffix}",
    source_configuration=apprunner.CfnService.SourceConfigurationProperty(
        authentication_configuration=apprunner.CfnService.AuthenticationConfigurationProperty(
            access_role_arn=access_role.role_arn
        ),
        auto_deployments_enabled=True,
        image_repository=apprunner.CfnService.ImageRepositoryProperty(
            image_identifier=f"{ecr_repo.repository_uri}:latest",
            image_configuration=apprunner.CfnService.ImageConfigurationProperty(
                port="80",
                runtime_environment_variables=env_vars
            )
        )
    )
)
```

Features:
- ECR-based deployment
- VPC integration
- Environment variable configuration
- Health check configuration:
  ```python
  health_check_configuration=apprunner.CfnService.HealthCheckConfigurationProperty(
      protocol="HTTP",
      path="/health",
      interval=10,
      timeout=10,
      healthy_threshold=2,
      unhealthy_threshold=3
  )
  ```
- Auto-scaling settings
- Instance Configuration:
  ```python
  instance_configuration=apprunner.CfnService.InstanceConfigurationProperty(
      cpu="4 vCPU",
      memory="12 GB",
      instance_role_arn=instance_role.role_arn
  )
  ```

## Deployment

### Deploy New Infrastructure

1. Set up environment:
```bash
# Clone repository and navigate to cdk directory
git clone <repository-url>
cd flotorch/cdk

# Install dependencies
pip install -r requirements.txt
```

2. Deploy infrastructure:
```bash
# Deploy all stacks
./deploy.sh

# Deploy specific stack
cdk deploy "FlotorchVPCStack-${TABLE_SUFFIX}" --require-approval never
```

The deployment process:
1. Generates unique suffix
2. Bootstraps CDK environment
3. Deploys VPC stack
4. Creates ECR repositories
5. Builds and pushes Docker images
6. Deploys State Machine stack
7. Updates environment variables

### Destroy Infrastructure

1. Destroy specific deployment:
```bash
# Set the suffix of the deployment to destroy
export TABLE_SUFFIX="your-suffix"

# Destroy App Runner service first
aws apprunner delete-service --service-arn <service-arn>

# Empty S3 bucket
aws s3 rm s3://flotorch-data-${AWS_ACCOUNT_ID}-${AWS_REGION}-${TABLE_SUFFIX} --recursive

# Delete ECR repositories
aws ecr delete-repository --repository-name flotorch-runtime-${TABLE_SUFFIX} --force
aws ecr delete-repository --repository-name flotorch-app-${TABLE_SUFFIX} --force

# Destroy CDK stacks in reverse order
cdk destroy "FlotorchStateMachineStack-${TABLE_SUFFIX}" --force
cdk destroy "FlotorchVPCStack-${TABLE_SUFFIX}" --force
```

2. Cleanup environment:
```bash
# Remove environment files
rm .env
rm cdk-outputs.json
```

## Environment Variables
Key environment variables are automatically configured in `.env`:
```bash
# AWS Configuration
aws_region=us-east-1

# DynamoDB Tables
experiment_question_metrics_table=ExperimentQuestionMetrics_${TABLE_SUFFIX}
execution_table=Execution_${TABLE_SUFFIX}
experiment_table=Experiment_${TABLE_SUFFIX}
execution_model_invocations_table=ExecutionModelInvocations_${TABLE_SUFFIX}

# OpenSearch Configuration
opensearch_host=<domain-endpoint>
opensearch_username=admin
opensearch_password=<password>
opensearch_serverless=false
vector_field_name=vectors

# Other Resources
step_function_arn=<state-machine-arn>
s3_bucket=flotorch-data-${AWS_ACCOUNT_ID}-${AWS_REGION}-${TABLE_SUFFIX}
bedrock_role_arn=<role-arn>
sagemaker_role_arn=<role-arn>
```

## Dependencies
Core dependencies are managed in `requirements.txt`:
```
aws-cdk-lib
constructs>=10.0.0
pydantic>=2.0.0
boto3>=1.26.0
sagemaker==2.175.0
```

## Preferred Deployment OS
- Ubuntu
- Amazon Linux 2

## Troubleshooting

### Common Issues

1. OpenSearch Domain Creation
   - Ensure sufficient instance capacity in the selected AZ
   - Check VPC subnet configurations:
     ```bash
     aws ec2 describe-subnets --subnet-ids <subnet-id>
     ```
   - Verify security group rules:
     ```bash
     aws ec2 describe-security-groups --group-ids <security-group-id>
     ```

2. State Machine Execution
   - Check IAM roles and permissions
   - Verify DynamoDB table access:
     ```bash
     aws dynamodb describe-table --table-name <table-name>
     ```
   - Monitor CloudWatch logs:
     ```bash
     aws logs get-log-events --log-group-name /aws/lambda/<function-name>
     ```

3. VPC Resources
   - Ensure sufficient IP address space
   - Check NAT Gateway configurations:
     ```bash
     aws ec2 describe-nat-gateways
     ```
   - Verify security group rules

### Logs and Monitoring
- CloudWatch Logs for Lambda functions:
  ```bash
  aws logs tail /aws/lambda/<function-name> --follow
  ```
- OpenSearch dashboard access:
  ```bash
  https://<domain-endpoint>/_dashboards/
  ```
- Step Functions execution history:
  ```bash
  aws stepfunctions get-execution-history --execution-arn <execution-arn>
  ```
- App Runner service logs:
  ```bash
  aws apprunner list-operations --service-arn <service-arn>
  ```

## Resource Management
- All resources are tagged with the table suffix
- RemovalPolicy.DESTROY set for development environments
- Automatic cleanup of resources on stack deletion

For more detailed information, refer to the AWS CDK documentation and individual service documentation.
