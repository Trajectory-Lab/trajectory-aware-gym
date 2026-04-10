"""Lightweight client for Qwen models hosted on SageMaker TGI endpoints.

Useful for quick endpoint health checks and standalone inference
outside the DSPy pipeline.  For DSPy-integrated inference, use
``get_task_lm("qwen3-sagemaker:1.7b")`` from ``config.llm_provider`` instead.

Usage::

    from trajectory_aware_gym.sagemaker import QwenClient

    client = QwenClient("1.7b")
    print(client.generate("Hello!"))

CLI::

    python -m trajectory_aware_gym.sagemaker.client 1.7b "Hello!"
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from trajectory_aware_gym.config import settings

logger = logging.getLogger(__name__)

ENDPOINTS: dict[str, str] = {
    "1.7b": settings.sagemaker.endpoint_1_7b,
    "4b": settings.sagemaker.endpoint_4b,
}

_RETRYABLE_SAGEMAKER_STATUS_CODES = (424, 429, 503)
_RETRYABLE_SAGEMAKER_ERROR_CODES = ("ModelError", "ThrottlingException")


def _is_retryable_sagemaker_error(exc: BaseException) -> bool:
    """Check if a botocore ``ClientError`` is retryable for SageMaker endpoints."""
    if not isinstance(exc, ClientError):
        return False
    error_info = exc.response.get("Error", {})
    code = error_info.get("Code", "")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
    return status in _RETRYABLE_SAGEMAKER_STATUS_CODES or code in _RETRYABLE_SAGEMAKER_ERROR_CODES


class QwenClient:
    """Simple client to call Qwen models on SageMaker."""

    def __init__(self, model: str = "1.7b", region: str | None = None) -> None:
        if model not in ENDPOINTS:
            raise ValueError(f"Unknown model '{model}'. Choose: {list(ENDPOINTS.keys())}")

        self.model = model
        self.endpoint_name = ENDPOINTS[model]

        retry_cfg = settings.retry
        boto_config = BotocoreConfig(
            retries={
                "mode": retry_cfg.boto3_retry_mode,
                "total_max_attempts": retry_cfg.boto3_max_attempts,
            },
            max_pool_connections=retry_cfg.inference_semaphore_size + 2,
            read_timeout=90,
        )
        self.client = boto3.client(
            "sagemaker-runtime",
            region_name=region or settings.sagemaker.region,
            config=boto_config,
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

        retry_cfg = settings.retry
        retryer = Retrying(
            stop=stop_after_attempt(retry_cfg.max_attempts),
            wait=wait_exponential_jitter(
                initial=retry_cfg.initial_wait_seconds,
                max=retry_cfg.max_wait_seconds,
                exp_base=retry_cfg.exponential_base,
            ),
            retry=retry_if_exception(_is_retryable_sagemaker_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                response = self.client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=json.dumps(payload),
                )

        result = json.loads(response["Body"].read().decode())  # type: ignore[possibly-undefined]

        if isinstance(result, list):
            return result[0].get("generated_text", "")
        return result.get("generated_text", "")

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        max_workers: int = 4,
    ) -> list[str]:
        """Generate text for multiple prompts in parallel.

        Args:
            prompts: List of input prompts.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            max_workers: Maximum concurrent requests to SageMaker.

        Returns:
            List of generated texts, in the same order as *prompts*.
        """
        results: list[str | None] = [None] * len(prompts)

        def _invoke(idx: int, prompt: str) -> tuple[int, str]:
            return idx, self.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_invoke, i, p): i for i, p in enumerate(prompts)}
            done_count = 0
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
                done_count += 1
                if done_count % 50 == 0:
                    print(f"  [{self.model}] Completed {done_count}/{len(prompts)}...")

        return [r if r is not None else "" for r in results]

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
