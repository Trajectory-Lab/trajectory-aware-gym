"""SageMaker deployment manager for Qwen base models (boto3 only).

Creates HuggingFace TGI endpoints for models not natively hosted on Bedrock.

Usage::

    python -m trajectory_aware_gym.sagemaker.deploy deploy all
    python -m trajectory_aware_gym.sagemaker.deploy deploy 1.7b
    python -m trajectory_aware_gym.sagemaker.deploy deploy 4b

    python -m trajectory_aware_gym.sagemaker.deploy delete all
    python -m trajectory_aware_gym.sagemaker.deploy delete 1.7b

    python -m trajectory_aware_gym.sagemaker.deploy status

Prerequisites:
    pip install boto3
    aws configure
"""

from __future__ import annotations

import sys
import time

import boto3

from trajectory_aware_gym.config import settings

_sm_cfg = settings.sagemaker


def _build_models() -> dict[str, dict[str, str | dict[str, str]]]:
    """Build the MODELS registry from centralised YAML/env config."""
    return {
        "1.7b": {
            "endpoint_name": _sm_cfg.endpoint_1_7b,
            "model_id": _sm_cfg.model_id_1_7b,
            "env": {
                "HF_MODEL_ID": _sm_cfg.model_id_1_7b,
                "HF_TASK": "text-generation",
                "SM_NUM_GPUS": "1",
                "MAX_INPUT_LENGTH": "2048",
                "MAX_TOTAL_TOKENS": "4096",
            },
        },
        "4b": {
            "endpoint_name": _sm_cfg.endpoint_4b,
            "model_id": _sm_cfg.model_id_4b,
            "env": {
                "HF_MODEL_ID": _sm_cfg.model_id_4b,
                "HF_TASK": "text-generation",
                "SM_NUM_GPUS": "1",
                "MAX_INPUT_LENGTH": "2048",
                "MAX_TOTAL_TOKENS": "4096",
            },
        },
    }


MODELS = _build_models()

COST_PER_HOUR = 1.41  # approximate for ml.g5.xlarge


def deploy_model(model_key: str) -> None:
    """Deploy a single model to a SageMaker real-time endpoint."""
    info = MODELS[model_key]
    name = str(info["endpoint_name"])
    model_id = str(info["model_id"])

    print(f"Deploying {model_id} to endpoint '{name}'...")
    print(f"Instance: {_sm_cfg.instance_type} (~${COST_PER_HOUR}/hr)")
    print()

    sm = boto3.client("sagemaker", region_name=_sm_cfg.region)
    timestamp = str(int(time.time()))
    model_name = f"{name}-model-{timestamp}"
    config_name = f"{name}-config-{timestamp}"

    print("  [1/3] Creating model...")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": _sm_cfg.tgi_image_uri,
            "Environment": info["env"],
        },
        ExecutionRoleArn=_sm_cfg.role_arn,
    )

    print("  [2/3] Creating endpoint config...")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "main",
                "ModelName": model_name,
                "InstanceType": _sm_cfg.instance_type,
                "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600,
            }
        ],
    )

    print("  [3/3] Creating endpoint (this takes 5-10 minutes)...")
    try:
        sm.create_endpoint(
            EndpointName=name,
            EndpointConfigName=config_name,
        )
    except sm.exceptions.ClientError as e:
        if "Cannot create already existing endpoint" in str(e):
            print(f"  Endpoint '{name}' already exists. Updating...")
            sm.update_endpoint(
                EndpointName=name,
                EndpointConfigName=config_name,
            )
        else:
            raise

    print("  Waiting for endpoint to be ready...")
    poll_interval = 30
    deadline = time.monotonic() + _sm_cfg.deploy_timeout
    while time.monotonic() < deadline:
        resp = sm.describe_endpoint(EndpointName=name)
        ep_status = resp["EndpointStatus"]
        if ep_status == "InService":
            print(f"\n  [OK] {model_id} is LIVE at endpoint '{name}'")
            return
        if ep_status == "Failed":
            reason = resp.get("FailureReason", "Unknown")
            print(f"\n  [FAILED] {reason}")
            return
        remaining = int(deadline - time.monotonic())
        print(f"  Status: {ep_status}... (waiting {poll_interval}s, {remaining}s left)")
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Endpoint '{name}' did not become InService within "
        f"{_sm_cfg.deploy_timeout}s. Check the SageMaker console."
    )


def delete_model(model_key: str) -> None:
    """Delete a model endpoint and associated resources."""
    info = MODELS[model_key]
    name = str(info["endpoint_name"])
    model_id = str(info["model_id"])

    print(f"Deleting {model_id} endpoint '{name}'...")
    sm = boto3.client("sagemaker", region_name=_sm_cfg.region)

    config_name = None
    try:
        resp = sm.describe_endpoint(EndpointName=name)
        config_name = resp.get("EndpointConfigName")
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not describe endpoint (may not exist): {exc}")

    try:
        sm.delete_endpoint(EndpointName=name)
        print("  Endpoint deleted.")
    except sm.exceptions.ClientError:
        print("  Endpoint not found (already deleted).")

    if config_name:
        try:
            sm.delete_endpoint_config(EndpointConfigName=config_name)
            print("  Endpoint config deleted.")
        except Exception as exc:  # noqa: BLE001
            print(f"  Could not delete endpoint config: {exc}")

    try:
        models = sm.list_models(NameContains=name, MaxResults=10)["Models"]
        for m in models:
            sm.delete_model(ModelName=m["ModelName"])
            print(f"  Model '{m['ModelName']}' deleted.")
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not clean up models: {exc}")

    print(f"  [OK] {model_id} shut down. Billing stopped.")


def status() -> None:
    """Print status and cost estimate for all configured endpoints."""
    sm = boto3.client("sagemaker", region_name=_sm_cfg.region)
    running_cost = 0.0

    for key, info in MODELS.items():
        name = str(info["endpoint_name"])
        model_id = str(info["model_id"])

        try:
            resp = sm.describe_endpoint(EndpointName=name)
            s = resp["EndpointStatus"]
            if s == "InService":
                print(f"  [{key}] {model_id}: RUNNING  (~${COST_PER_HOUR}/hr)")
                running_cost += COST_PER_HOUR
            elif s == "Creating":
                print(f"  [{key}] {model_id}: DEPLOYING...")
            elif s == "Failed":
                reason = resp.get("FailureReason", "Unknown")
                print(f"  [{key}] {model_id}: FAILED - {reason}")
            else:
                print(f"  [{key}] {model_id}: {s}")
        except sm.exceptions.ClientError:
            print(f"  [{key}] {model_id}: OFF")

    print()
    if running_cost > 0:
        print(f"  Total: ~${running_cost:.2f}/hr (${running_cost * 24:.2f}/day)")
        print("  Shut down: python -m trajectory_aware_gym.sagemaker.deploy delete all")
    else:
        print("  No endpoints running. No billing.")


def _resolve_targets(target: str) -> list[str]:
    """Resolve 'all' to list of model keys."""
    if target == "all":
        return list(MODELS.keys())
    if target in MODELS:
        return [target]
    print(f"Unknown target: '{target}'. Use: 1.7b, 4b, or all")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m trajectory_aware_gym.sagemaker.deploy deploy [1.7b|4b|all]")
        print("  python -m trajectory_aware_gym.sagemaker.deploy delete [1.7b|4b|all]")
        print("  python -m trajectory_aware_gym.sagemaker.deploy status")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "status":
        status()
    elif command in ("deploy", "delete"):
        if len(sys.argv) < 3:
            print(f"Specify target: ... {command} [1.7b|4b|all]")
            sys.exit(1)
        targets = _resolve_targets(sys.argv[2].lower())

        for t in targets:
            if command == "deploy":
                deploy_model(t)
            else:
                delete_model(t)
            print()

        if command == "deploy":
            print("=" * 60)
            print("Test with:")
            print('  python -m trajectory_aware_gym.sagemaker.client 1.7b "Hello"')
            print('  python -m trajectory_aware_gym.sagemaker.client 4b "Hello"')
            print()
            print("SHUT DOWN WHEN DONE:")
            print("  python -m trajectory_aware_gym.sagemaker.deploy delete all")
            print("=" * 60)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
