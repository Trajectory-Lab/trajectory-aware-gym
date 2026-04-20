"""Enable per-example progress for GEPA's inner valset evaluations.

Upstream `gepa.core.engine` ticks its "GEPA Optimization: X/N rollouts" bar
only once per iteration, after the full valset eval finishes — the user sees
no progress for minutes, then a 300-rollout jump. `gepa` exposes no
per-rollout callback, but DSPy's `Evaluate` (used inside
``dspy.teleprompt.gepa.gepa_utils.DspyAdapter.evaluate``) does accept
``display_progress=True``; GEPA just doesn't pass it.

This module replaces the ``Evaluate`` symbol in ``gepa_utils`` with a
subclass that defaults ``display_progress=True``. Result: an inner per-example
bar redraws during each valset eval, giving live feedback while the outer
GEPA bar still ticks in iteration-sized chunks. Idempotent and thread-safe.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

_patch_applied = False
_patch_lock = threading.Lock()


def enable_gepa_eval_progress() -> None:
    """Patch DSPy's GEPA adapter so inner valset evals show live progress."""
    global _patch_applied
    if _patch_applied:
        return

    with _patch_lock:
        if _patch_applied:
            return

        gepa_utils = importlib.import_module("dspy.teleprompt.gepa.gepa_utils")
        original_evaluate = gepa_utils.Evaluate

        class _EvaluateWithProgress(original_evaluate):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs.setdefault("display_progress", True)
                super().__init__(*args, **kwargs)

        gepa_utils.Evaluate = _EvaluateWithProgress  # pyright: ignore[reportAttributeAccessIssue]
        _patch_applied = True


def reset_gepa_eval_progress_patch() -> None:
    """Undo the patch (for test isolation). Restores the original `Evaluate`."""
    global _patch_applied
    with _patch_lock:
        if not _patch_applied:
            return
        gepa_utils = importlib.import_module("dspy.teleprompt.gepa.gepa_utils")
        wrapper = gepa_utils.Evaluate
        gepa_utils.Evaluate = wrapper.__bases__[0]  # pyright: ignore[reportAttributeAccessIssue]
        _patch_applied = False
