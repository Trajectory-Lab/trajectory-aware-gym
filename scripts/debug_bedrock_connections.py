"""Debug script to isolate Bedrock 503 "Too many connections" errors.

Tests Bedrock access directly via boto3 (bypassing litellm), via litellm,
and with alternate model IDs. Quickly narrows down whether this is an
account/model access issue, a litellm client issue, or a rate limit.
"""

from __future__ import annotations

import json
import os
import time

# ── Config ──────────────────────────────────────────────────────
BEDROCK_MODEL_LITELLM = "bedrock/us.anthropic.claude-sonnet-4-6"
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
BEDROCK_HAIKU_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def test_boto3_direct():
    """Test 1: Call Bedrock directly via boto3 — bypasses litellm entirely."""
    print("\n--- Test 1: Direct boto3 call to Bedrock ---")
    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        print(f"  Region: {AWS_REGION}")
        print(f"  Model:  {BEDROCK_MODEL_ID}")

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 32,
                "temperature": 0.0,
            }
        )
        start = time.monotonic()
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        elapsed = time.monotonic() - start
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]
        print(f"  OK: {text!r} ({elapsed:.1f}s)")
        client.close()
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")


def test_boto3_converse():
    """Test 2: Call Bedrock via converse API (what litellm actually uses)."""
    print("\n--- Test 2: Direct boto3 converse API ---")
    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        print(f"  Model: {BEDROCK_MODEL_ID}")

        start = time.monotonic()
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": "Say hello in one word."}]}],
            inferenceConfig={"maxTokens": 32, "temperature": 0.0},
        )
        elapsed = time.monotonic() - start
        text = response["output"]["message"]["content"][0]["text"]
        tokens = response.get("usage", {})
        print(f"  OK: {text!r} ({elapsed:.1f}s) tokens={tokens}")
        client.close()
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")


def test_boto3_haiku():
    """Test 3: Try Haiku via converse to check if it's model-specific."""
    print("\n--- Test 3: Direct boto3 converse → Haiku (alternate model) ---")
    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        print(f"  Model: {BEDROCK_HAIKU_ID}")

        start = time.monotonic()
        response = client.converse(
            modelId=BEDROCK_HAIKU_ID,
            messages=[{"role": "user", "content": [{"text": "Say hello in one word."}]}],
            inferenceConfig={"maxTokens": 32, "temperature": 0.0},
        )
        elapsed = time.monotonic() - start
        text = response["output"]["message"]["content"][0]["text"]
        print(f"  OK: {text!r} ({elapsed:.1f}s)")
        client.close()
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")


def test_litellm_sonnet():
    """Test 4: Call via litellm (how the runner actually calls it)."""
    print("\n--- Test 4: litellm → Bedrock Sonnet ---")
    from litellm import completion

    start = time.monotonic()
    try:
        resp = completion(
            model=BEDROCK_MODEL_LITELLM,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            temperature=0.0,
            max_tokens=32,
            aws_region_name=AWS_REGION,
            num_retries=0,
        )
        elapsed = time.monotonic() - start
        text = resp.choices[0].message.content
        print(f"  OK: {text!r} ({elapsed:.1f}s)")
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"  FAIL ({elapsed:.1f}s): {type(exc).__name__}: {str(exc)[:300]}")


def test_litellm_haiku():
    """Test 5: Call Haiku via litellm."""
    print("\n--- Test 5: litellm → Bedrock Haiku ---")
    from litellm import completion

    start = time.monotonic()
    try:
        resp = completion(
            model=f"bedrock/{BEDROCK_HAIKU_ID}",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            temperature=0.0,
            max_tokens=32,
            aws_region_name=AWS_REGION,
            num_retries=0,
        )
        elapsed = time.monotonic() - start
        text = resp.choices[0].message.content
        print(f"  OK: {text!r} ({elapsed:.1f}s)")
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"  FAIL ({elapsed:.1f}s): {type(exc).__name__}: {str(exc)[:300]}")


def test_litellm_sonnet_httpx_url():
    """Test 6: Check what URL litellm is actually hitting."""
    print("\n--- Test 6: litellm debug — what endpoint is it using? ---")
    import litellm

    litellm._turn_on_debug()
    from litellm import completion

    try:
        resp = completion(
            model=BEDROCK_MODEL_LITELLM,
            messages=[{"role": "user", "content": "Say hi."}],
            temperature=0.0,
            max_tokens=16,
            aws_region_name=AWS_REGION,
            num_retries=0,
        )
        print(f"  OK: {resp.choices[0].message.content!r}")
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {str(exc)[:200]}")
    finally:
        litellm._turn_off_debug()


def test_list_bedrock_models():
    """Test 7: List available Bedrock models to check access."""
    print("\n--- Test 7: List Bedrock model access ---")
    try:
        import boto3

        client = boto3.client("bedrock", region_name=AWS_REGION)
        response = client.list_foundation_models(byProvider="Anthropic", byOutputModality="TEXT")
        models = response.get("modelSummaries", [])
        sonnet_models = [
            m["modelId"]
            for m in models
            if "sonnet" in m["modelId"].lower() or "claude" in m["modelId"].lower()
        ]
        print(f"  Available Claude models in {AWS_REGION}:")
        for m in sorted(sonnet_models):
            print(f"    {m}")

        # Check inference profiles
        try:
            profiles = client.list_inference_profiles()
            profile_list = profiles.get("inferenceProfileSummaries", [])
            claude_profiles = [
                p
                for p in profile_list
                if "claude" in p.get("inferenceProfileId", "").lower()
                or "claude" in p.get("modelArn", "").lower()
            ]
            if claude_profiles:
                print("\n  Claude inference profiles:")
                for p in claude_profiles:
                    print(f"    {p.get('inferenceProfileId')} — {p.get('status', '?')}")
        except Exception as exc:
            print(f"  (Could not list inference profiles: {exc})")

        client.close()
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")


def run_test():
    print("=" * 70)
    print("Bedrock Connection Debug")
    print(f"  Region:         {AWS_REGION}")
    print(f"  Sonnet model:   {BEDROCK_MODEL_ID}")
    print(f"  Haiku model:    {BEDROCK_HAIKU_ID}")
    print("=" * 70)

    test_list_bedrock_models()
    test_boto3_direct()
    test_boto3_converse()
    test_boto3_haiku()
    test_litellm_sonnet()
    test_litellm_haiku()
    # Only run debug test if prior tests failed
    test_litellm_sonnet_httpx_url()

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    from pathlib import Path

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    run_test()
