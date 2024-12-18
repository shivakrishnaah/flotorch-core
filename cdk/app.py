#!/usr/bin/env python3
import os
import aws_cdk as cdk
from vpc.vpc_stack import VPCStack
from app.app_stack import AppStack
from state_machine.state_machine_stack import StateMachineStack

app = cdk.App()

# Environment configuration
env = cdk.Environment(
    account=os.getenv("AWS_ACCOUNT_ID"),
    region=os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
)

# Get table suffix from environment variable
table_suffix = os.getenv("TABLE_SUFFIX", "default")

# Create VPC Stack with table suffix
vpc_stack = VPCStack(app, f"FlotorchVPCStack-{table_suffix}", table_suffix=table_suffix, env=env)

# Create State Machine Stack with explicit dependency on VPC Stack
state_machine_stack = StateMachineStack(app, f"FlotorchStateMachineStack-{table_suffix}", vpc_stack=vpc_stack, env=env)

# Create App Stack with explicit dependency on VPC Stack
app_stack = AppStack(app, f"FlotorchAppStack-{table_suffix}", vpc_stack=vpc_stack, env=env)

# Add dependencies to ensure proper order of deployment
state_machine_stack.add_dependency(vpc_stack)
app_stack.add_dependency(vpc_stack)
app_stack.add_dependency(state_machine_stack)

app.synth()