#!/usr/bin/env python3
"""List available models in AWS Bedrock."""

import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def list_bedrock_models():
    """List available models using Bedrock API with bearer token."""

    region = os.getenv("AWS_REGION", "us-east-1")
    bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

    if not bearer_token:
        print("❌ AWS_BEARER_TOKEN_BEDROCK not set in .env file")
        return

    # Bedrock API endpoint for listing foundation models
    url = f"https://bedrock.{region}.amazonaws.com/foundation-models"

    headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}

    print(f"🔍 Querying Bedrock in region: {region}")
    print(f"   Endpoint: {url}\n")

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        models = data.get("modelSummaries", [])

        print(f"📊 Found {len(models)} models\n")
        print("=" * 80)

        # Filter for Qwen models
        qwen_models = [m for m in models if "qwen" in m.get("modelId", "").lower()]

        if qwen_models:
            print("\n🎯 Qwen Models:")
            print("-" * 80)
            for model in qwen_models:
                model_id = model.get("modelId", "N/A")
                model_name = model.get("modelName", "N/A")
                provider = model.get("providerName", "N/A")
                print(f"\nModel ID: {model_id}")
                print(f"  Name: {model_name}")
                print(f"  Provider: {provider}")
        else:
            print("\n⚠️  No Qwen models found")

        # Show all models for reference
        print("\n\n📋 All Available Models:")
        print("-" * 80)
        for model in models:
            model_id = model.get("modelId", "N/A")
            model_name = model.get("modelName", "N/A")
            provider = model.get("providerName", "N/A")
            print(f"{provider:20} | {model_name:40} | {model_id}")

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Status Code: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    list_bedrock_models()
