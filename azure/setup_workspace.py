"""
Azure ML Training Setup for DrishT OCR
=========================================
Creates Azure ML Workspace, uploads data, and submits training jobs
for both detection (SSDLite-MobileNetV3) and recognition (CRNN-Light).

Azure for Students: $100 credit, use low-priority GPU instances.

Usage:
    python azure/setup_workspace.py          # Create workspace
    python azure/submit_training.py detection # Submit detection job
    python azure/submit_training.py recognition # Submit recognition job
"""

import os
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Workspace,
    AmlCompute,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBSCRIPTION_ID = "9d35b44b-44e8-4c6f-a536-234a87e59983"
RESOURCE_GROUP = "drisht-rg"
WORKSPACE_NAME = "drisht-ml"
LOCATION = "centralindia"  # Close to India, good for Students sub

# Compute cluster — low-priority T4 GPU for cost savings
COMPUTE_NAME = "gpu-t4-lowpri"
COMPUTE_VM_SIZE = "Standard_NC4as_T4_v3"  # 4 vCPU, 28 GB RAM, T4 GPU
COMPUTE_MIN_INSTANCES = 0  # Scale to zero when idle (no cost!)
COMPUTE_MAX_INSTANCES = 1  # Single node (budget-friendly)


def get_ml_client():
    """Get authenticated ML client."""
    try:
        credential = AzureCliCredential()
        client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )
        # Test the connection
        client.workspaces.get(WORKSPACE_NAME)
        return client
    except Exception:
        credential = DefaultAzureCredential()
        return MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )


def create_resource_group():
    """Create the resource group if it doesn't exist."""
    print(f"  Creating resource group: {RESOURCE_GROUP} in {LOCATION}")
    os.system(f'az group create -n {RESOURCE_GROUP} -l {LOCATION} -o none')
    print(f"  Done.")


def create_workspace():
    """Create Azure ML workspace."""
    print(f"\n{'='*60}")
    print(f"  Creating Azure ML Workspace: {WORKSPACE_NAME}")
    print(f"{'='*60}")

    credential = AzureCliCredential()
    client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
    )

    # Check if workspace already exists
    try:
        ws = client.workspaces.get(WORKSPACE_NAME)
        print(f"  Workspace already exists: {ws.name}")
        return client
    except Exception:
        pass

    # Create workspace
    ws = Workspace(
        name=WORKSPACE_NAME,
        location=LOCATION,
        display_name="DrishT OCR Training",
        description="Indian scene text detection and recognition model training",
    )
    print(f"  Creating workspace (this may take 2-3 minutes)...")
    ws = client.workspaces.begin_create(ws).result()
    print(f"  Workspace created: {ws.name} ({ws.location})")

    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def create_compute(client):
    """Create low-priority GPU compute cluster."""
    print(f"\n{'='*60}")
    print(f"  Setting up compute: {COMPUTE_NAME}")
    print(f"{'='*60}")

    try:
        compute = client.compute.get(COMPUTE_NAME)
        print(f"  Compute already exists: {compute.name} ({compute.size})")
        return
    except Exception:
        pass

    compute = AmlCompute(
        name=COMPUTE_NAME,
        type="amlcompute",
        size=COMPUTE_VM_SIZE,
        min_instances=COMPUTE_MIN_INSTANCES,
        max_instances=COMPUTE_MAX_INSTANCES,
        tier="low_priority",  # ~70% cheaper than dedicated!
    )
    print(f"  Creating compute cluster: {COMPUTE_VM_SIZE}")
    print(f"  Tier: low_priority (saves ~70% cost)")
    print(f"  Estimated cost: ~$0.13/hr (low-priority T4)")
    client.compute.begin_create_or_update(compute).result()
    print(f"  Compute cluster created: {COMPUTE_NAME}")


def main():
    print(f"\n{'='*60}")
    print(f"  DrishT OCR — Azure ML Setup")
    print(f"{'='*60}")
    print(f"  Subscription: Azure for Students")
    print(f"  Resource Group: {RESOURCE_GROUP}")
    print(f"  Workspace: {WORKSPACE_NAME}")
    print(f"  Location: {LOCATION}")
    print(f"  Compute: {COMPUTE_NAME} ({COMPUTE_VM_SIZE}, low-priority)")

    # Step 1: Resource group
    create_resource_group()

    # Step 2: ML Workspace
    client = create_workspace()

    # Step 3: Compute cluster
    create_compute(client)

    print(f"\n{'='*60}")
    print(f"  Setup Complete!")
    print(f"{'='*60}")
    print(f"  Next: Run 'python azure/submit_training.py detection' or")
    print(f"        'python azure/submit_training.py recognition'")


if __name__ == "__main__":
    main()
