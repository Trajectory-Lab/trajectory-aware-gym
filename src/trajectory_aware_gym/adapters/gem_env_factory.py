"""Factory for GEM environments that caches HF datasets across instantiations.

``gem.make(env_id)`` calls ``datasets.load_dataset(dataset_name)`` inside the
env constructor, which sends HEAD requests to Hugging Face even when the
dataset is already cached locally. With hundreds of eval/GEPA rollouts, that
floods the HF endpoint and triggers HTTP 429 rate limits.

This module resolves the ``dataset_name`` from GEM's ``ENV_REGISTRY`` entry,
loads the dataset exactly once per env_id (thread-safe), and passes the cached
object to ``gem.make()`` via the ``dataset=`` kwarg that ``MathEnv`` / ``QaEnv``
/ ``CodeEnv`` / ``MathVisualEnv`` all accept.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

_dataset_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


def _load_env_dataset(env_id: str) -> Any | None:
    """Return a cached dataset for ``env_id`` or ``None`` if the env has none.

    Looks up ``env_id`` in GEM's registry, pulls the ``dataset_name`` kwarg,
    and calls ``datasets.load_dataset`` once per env_id. Envs without a
    ``dataset_name`` (game envs, reasoning_gym) return ``None``.
    """
    if env_id in _dataset_cache:
        return _dataset_cache[env_id]

    with _cache_lock:
        if env_id in _dataset_cache:
            return _dataset_cache[env_id]

        importlib.import_module("gem.envs")
        registration = importlib.import_module("gem.envs.registration")
        registry: dict[str, Any] = registration.ENV_REGISTRY
        spec = registry.get(env_id)
        if spec is None:
            _dataset_cache[env_id] = None
            return None

        dataset_name = spec.kwargs.get("dataset_name")
        if not dataset_name:
            _dataset_cache[env_id] = None
            return None

        from datasets import load_dataset  # pyright: ignore[reportMissingImports]

        # GEM itself registers datasets by name only and loads them without
        # revision pinning. Pinning here would be asymmetric with the
        # underlying library and does not reduce the attack surface.
        _dataset_cache[env_id] = load_dataset(dataset_name)  # nosec B615
        return _dataset_cache[env_id]


def make_env(env_id: str, **kwargs: Any) -> Any:
    """Construct a GEM env, reusing a process-wide cached HF dataset.

    Falls back to plain ``gem.make(env_id, **kwargs)`` for envs that don't
    use a Hugging Face dataset or when the caller has already supplied a
    ``dataset=`` override.
    """
    gem = importlib.import_module("gem")

    if "dataset" not in kwargs:
        cached = _load_env_dataset(env_id)
        if cached is not None:
            kwargs["dataset"] = cached

    return gem.make(env_id, **kwargs)


def reset_dataset_cache() -> None:
    """Clear the cache (for test isolation)."""
    with _cache_lock:
        _dataset_cache.clear()
