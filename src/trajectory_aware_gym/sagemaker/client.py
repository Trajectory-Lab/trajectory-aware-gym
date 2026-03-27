"""Lightweight client for Qwen models hosted on SageMaker TGI endpoints.

Useful for quick endpoint health checks and standalone inference
outside the DSPy pipeline.  For DSPy-integrated inference, use
``get_task_lm("qwen3-base:1.7b")`` from ``config.llm_provider`` instead.

Usage::

    from trajectory_aware_gym.sagemaker import QwenClient

    client = QwenClient("1.7b")
    print(client.generate("Hello!"))

CLI::

    python -m trajectory_aware_gym.sagemaker.client 1.7b "Hello!"
"""

from __future__ import annotations

import json
import sys

import boto3

from trajectory_aware_gym.config import settings

ENDPOINTS: dict[str, str] = {
    "1.7b": settings.sagemaker.endpoint_1_7b,
    "4b": settings.sagemaker.endpoint_4b,
}


class QwenClient:
    """Simple client to call Qwen models on SageMaker."""

    def __init__(self, model: str = "1.7b", region: str | None = None) -> None:
        if model not in ENDPOINTS:
            raise ValueError(f"Unknown model '{model}'. Choose: {list(ENDPOINTS.keys())}")

        self.model = model
        self.endpoint_name = ENDPOINTS[model]
        self.client = boto3.client(
            "sagemaker-runtime",
            region_name=region or settings.sagemaker.region,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The input text/question.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeating tokens.

        Returns:
            Generated text string.
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
            },
        }

        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read().decode())

        if isinstance(result, list):
            return result[0].get("generated_text", "")
        return result.get("generated_text", "")

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate text for multiple prompts sequentially.

        Args:
            prompts: List of input prompts.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.

        Returns:
            List of generated texts.
        """
        results: list[str] = []
        for i, prompt in enumerate(prompts):
            if (i + 1) % 50 == 0:
                print(f"  [{self.model}] Processing {i + 1}/{len(prompts)}...")
            result = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)
        return results

    def is_available(self) -> bool:
        """Check if the endpoint is running and responding."""
        try:
            self.generate("test", max_tokens=1)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.model}] Endpoint not available: {e}")
            return False
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m trajectory_aware_gym.sagemaker.client [1.7b|4b] "prompt"')
        sys.exit(1)

    _model = sys.argv[1]
    _prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello! Who are you?"

    print(f"Model: {_model} | Endpoint: {ENDPOINTS.get(_model, '???')}")
    print(f"Prompt: {_prompt}")
    print("-" * 40)

    qwen = QwenClient(_model)

    if not qwen.is_available():
        print("\nEndpoint is not running.")
        print(f"Start it with: python -m trajectory_aware_gym.sagemaker.deploy deploy {_model}")
        sys.exit(1)

    print(f"Response: {qwen.generate(_prompt)}")
