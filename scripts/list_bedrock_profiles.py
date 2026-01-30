#!/usr/bin/env python3
"""List available inference profiles in AWS Bedrock."""

import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def list_inference_profiles():
    """List available inference profiles using Bedrock API with bearer token."""

    region = os.getenv("AWS_REGION", "us-east-1")
    bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

    if not bearer_token:
        print("❌ AWS_BEARER_TOKEN_BEDROCK not set in .env file")
        return

    # Bedrock API endpoint for listing inference profiles
    url = f"https://bedrock.{region}.amazonaws.com/inference-profiles"

    headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}

    print(f"🔍 Querying Bedrock Inference Profiles in region: {region}")
    print(f"   Endpoint: {url}\n")

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        profiles = data.get("inferenceProfileSummaries", [])

        print(f"📊 Found {len(profiles)} inference profiles\n")
        print("=" * 120)

        # Filter for Llama models
        llama_profiles = [p for p in profiles if "llama" in p.get("inferenceProfileId", "").lower()]

        if llama_profiles:
            print("\n🎯 Llama Inference Profiles:")
            print("-" * 120)
            for profile in llama_profiles:
                profile_id = profile.get("inferenceProfileId", "N/A")
                profile_name = profile.get("inferenceProfileName", "N/A")
                model_id = (
                    profile.get("models", [{}])[0].get("modelId", "N/A")
                    if profile.get("models")
                    else "N/A"
                )
                print(f"\nProfile ID: {profile_id}")
                print(f"  Name: {profile_name}")
                print(f"  Underlying Model: {model_id}")
        else:
            print("\n⚠️  No Llama inference profiles found")

        # Filter for Qwen models
        qwen_profiles = [p for p in profiles if "qwen" in p.get("inferenceProfileId", "").lower()]

        if qwen_profiles:
            print("\n\n🎯 Qwen Inference Profiles:")
            print("-" * 120)
            for profile in qwen_profiles:
                profile_id = profile.get("inferenceProfileId", "N/A")
                profile_name = profile.get("inferenceProfileName", "N/A")
                model_id = (
                    profile.get("models", [{}])[0].get("modelId", "N/A")
                    if profile.get("models")
                    else "N/A"
                )
                print(f"\nProfile ID: {profile_id}")
                print(f"  Name: {profile_name}")
                print(f"  Underlying Model: {model_id}")

        # Show all profiles for reference
        print("\n\n📋 All Available Inference Profiles:")
        print("-" * 120)
        print(f"{'Profile ID':<60} | {'Name':<40} | {'Model ID'}")
        print("-" * 120)
        for profile in profiles:
            profile_id = profile.get("inferenceProfileId", "N/A")
            profile_name = profile.get("inferenceProfileName", "N/A")
            models = profile.get("models", [])
            model_id = models[0].get("modelId", "N/A") if models else "N/A"
            print(f"{profile_id:<60} | {profile_name:<40} | {model_id}")

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Status Code: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    list_inference_profiles()
