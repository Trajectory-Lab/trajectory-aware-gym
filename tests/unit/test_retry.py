"""Tests for exponential backoff retry and concurrency semaphore."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from litellm.exceptions import (  # type: ignore[import-untyped]
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)

from trajectory_aware_gym.adapters.gem_episode_runner import (
    _completion_with_retry,
    _reset_inference_semaphore,
)
from trajectory_aware_gym.config.core import Settings


def _make_response(text: str = "answer") -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    message = SimpleNamespace(content=text, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _fast_retry_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """Load settings with fast retry waits for tests."""
    Settings.reset()
    _reset_inference_semaphore()
    s = Settings()
    monkeypatch.setattr(
        s.retry, "initial_wait_seconds", overrides.get("initial_wait_seconds", 0.01)
    )
    monkeypatch.setattr(s.retry, "max_wait_seconds", overrides.get("max_wait_seconds", 0.05))
    for key, value in overrides.items():
        if key not in ("initial_wait_seconds", "max_wait_seconds"):
            monkeypatch.setattr(s.retry, key, value)


# ===========================================================================
# Retry on transient errors
# ===========================================================================


class TestCompletionWithRetry:
    """Tests for _completion_with_retry() tenacity wrapper."""

    def test_succeeds_after_transient_error(self, monkeypatch):
        _fast_retry_settings(monkeypatch, max_attempts=3)
        calls: list[dict[str, Any]] = []

        def mock_completion(messages: Any, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            if len(calls) == 1:
                raise ServiceUnavailableError(
                    message="503 Service Unavailable",
                    llm_provider="bedrock",
                    model="test-model",
                )
            return _make_response()

        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            mock_completion,
        )

        result = _completion_with_retry(
            messages=[{"role": "user", "content": "test"}],
            completion_kwargs={"model": "bedrock/test"},
        )
        assert len(calls) == 2
        assert result.choices[0].message.content == "answer"

    def test_exhausted_retries_reraise_original(self, monkeypatch):
        _fast_retry_settings(monkeypatch, max_attempts=3)

        def mock_completion(messages: Any, **kwargs: Any) -> SimpleNamespace:
            raise RateLimitError(
                message="429 Rate Limit",
                llm_provider="bedrock",
                model="test-model",
            )

        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            mock_completion,
        )

        with pytest.raises(RateLimitError, match="429"):
            _completion_with_retry(
                messages=[{"role": "user", "content": "test"}],
                completion_kwargs={"model": "bedrock/test"},
            )

    def test_non_retryable_exception_propagates_immediately(self, monkeypatch):
        _fast_retry_settings(monkeypatch, max_attempts=3)
        calls: list[int] = []

        def mock_completion(messages: Any, **kwargs: Any) -> SimpleNamespace:
            calls.append(1)
            raise AuthenticationError(
                message="403 Forbidden",
                llm_provider="bedrock",
                model="test-model",
            )

        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            mock_completion,
        )

        with pytest.raises(AuthenticationError, match="403"):
            _completion_with_retry(
                messages=[{"role": "user", "content": "test"}],
                completion_kwargs={"model": "bedrock/test"},
            )

        assert len(calls) == 1

    def test_injects_litellm_num_retries_zero(self, monkeypatch):
        _fast_retry_settings(monkeypatch, litellm_num_retries=0)
        captured_kwargs: dict[str, Any] = {}

        def mock_completion(messages: Any, **kwargs: Any) -> SimpleNamespace:
            captured_kwargs.update(kwargs)
            return _make_response()

        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            mock_completion,
        )

        _completion_with_retry(
            messages=[{"role": "user", "content": "test"}],
            completion_kwargs={"model": "bedrock/test"},
        )
        assert captured_kwargs["num_retries"] == 0

    def test_no_retry_when_max_attempts_is_one(self, monkeypatch):
        _fast_retry_settings(monkeypatch, max_attempts=1)
        calls: list[int] = []

        def mock_completion(messages: Any, **kwargs: Any) -> SimpleNamespace:
            calls.append(1)
            raise ServiceUnavailableError(
                message="503",
                llm_provider="bedrock",
                model="test-model",
            )

        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            mock_completion,
        )

        with pytest.raises(ServiceUnavailableError):
            _completion_with_retry(
                messages=[{"role": "user", "content": "test"}],
                completion_kwargs={"model": "bedrock/test"},
            )
        assert len(calls) == 1

    def test_uses_exponential_backoff_without_jitter(self, monkeypatch):
        _fast_retry_settings(monkeypatch, max_attempts=2, jitter=False)

        def mock_completion(messages: Any, **kwargs: Any) -> SimpleNamespace:
            if not hasattr(mock_completion, "_called"):
                mock_completion._called = True  # type: ignore[attr-defined]
                raise ServiceUnavailableError(
                    message="503",
                    llm_provider="bedrock",
                    model="test-model",
                )
            return _make_response()

        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            mock_completion,
        )

        result = _completion_with_retry(
            messages=[{"role": "user", "content": "test"}],
            completion_kwargs={"model": "bedrock/test"},
        )
        assert result.choices[0].message.content == "answer"


# ===========================================================================
# Semaphore concurrency tests
# ===========================================================================


class TestInferenceSemaphore:
    """Tests that the semaphore caps concurrent inflight LLM calls."""

    def test_semaphore_limits_concurrency(self, monkeypatch):
        _fast_retry_settings(monkeypatch, inference_semaphore_size=2)
        _reset_inference_semaphore()

        max_concurrent = 0
        current_concurrent = 0
        lock = threading.Lock()

        def mock_completion(messages: Any, **kwargs: Any) -> SimpleNamespace:
            nonlocal max_concurrent, current_concurrent
            with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent

            # Give other threads a chance to enter
            time.sleep(0.05)

            with lock:
                current_concurrent -= 1

            return _make_response()

        monkeypatch.setattr(
            "trajectory_aware_gym.adapters.gem_episode_runner.completion",
            mock_completion,
        )

        results: list[Any] = [None] * 4
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                results[idx] = _completion_with_retry(
                    messages=[{"role": "user", "content": f"test-{idx}"}],
                    completion_kwargs={"model": "bedrock/test"},
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Worker threads raised: {errors}"
        assert all(r is not None for r in results)
        # With semaphore_size=2, at most 2 should be concurrent
        assert max_concurrent <= 2
