"""DSPy Module that wraps GEMEpisodeRunner for GEPA prompt optimization.

GEPA evolves the instructions of the internal Predict module. On each forward
call the current instructions are extracted and used as the system prompt for
a full GEM episode via GEMEpisodeRunner. The resulting trajectory is attached
to the returned Prediction so the GEPA metric can score it.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, cast

import dspy  # type: ignore[import-untyped]
from dspy.utils.dummies import DummyLM  # type: ignore[import-untyped]

from trajectory_aware_gym.adapters.gem_episode_runner import (
    GEMEpisodeRunner,
    run_episode_with_retry,
)
from trajectory_aware_gym.adapters.trajectory_logger import TrajectoryLog

logger = logging.getLogger(__name__)


class GEMSolverSignature(dspy.Signature):
    """Solve a GEM environment task given a problem statement."""

    problem: str = dspy.InputField(desc="The task/problem from the GEM environment")
    seed: int = dspy.InputField(desc="Deterministic GEM reset seed for this example")
    answer: str = dspy.OutputField(desc="The final answer produced by the agent")


class GEMSolverModule(dspy.Module):
    """DSPy module bridging GEPA prompt evolution with GEM episode execution.

    GEPA modifies ``self.predict.signature.instructions``. Each forward call
    extracts those instructions and passes them as the system prompt to
    ``GEMEpisodeRunner``, which runs a full multi-turn GEM episode.

    The returned ``dspy.Prediction`` carries:
    - ``answer``: the agent's final action text
    - ``trajectory``: the full ``TrajectoryLog`` for fitness scoring
    """

    def __init__(
        self,
        runner: GEMEpisodeRunner,
        *,
        default_instructions: str = "",
        val_seeds: frozenset[int] | None = None,
        val_temperature: float | None = None,
    ) -> None:
        """Wire a runner for GEPA compile.

        ``val_seeds`` and ``val_temperature`` together let GEPA score its
        valset examples greedily while trainset rollouts stay stochastic.
        When the forward call's ``seed`` is in ``val_seeds``, the runner
        is invoked with ``temperature_override=val_temperature`` — matching
        the GEM paper's "sampling temperature, evaluation = 0.0" column
        (Table 3) without needing a second runner instance.
        """
        super().__init__()
        self._runner = runner
        self._val_seeds = val_seeds
        self._val_temperature = val_temperature
        sig = GEMSolverSignature
        if default_instructions:
            sig = sig.with_instructions(default_instructions)
        self.predict = dspy.Predict(sig)

    def __deepcopy__(self, memo: dict) -> GEMSolverModule:
        """Deep-copy DSPy state (Predict/instructions) but share the runner."""
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "_runner":
                object.__setattr__(new, k, v)
            else:
                object.__setattr__(new, k, copy.deepcopy(v, memo))
        return new

    @property
    def instructions(self) -> str:
        return cast(str, getattr(self.predict.signature, "instructions", ""))

    def forward(self, *, problem: str, seed: int | None = None, **kwargs: Any) -> dspy.Prediction:
        system_prompt = self.instructions or "You are a helpful assistant."
        temperature_override: float | None = None
        if self._val_seeds is not None and seed is not None and seed in self._val_seeds:
            temperature_override = self._val_temperature

        # Catch runner failures so the metric still gets called and can record
        # the outcome. DSPy's ParallelExecutor swallows exceptions raised here
        # and skips the metric entirely, which would hide infra failures from
        # the validation-scorable count. ``run_episode_with_retry`` adds a
        # bounded retry on transient infra errors (timeouts, connection blips)
        # before we give up and mark the episode runner_error.
        status = "ok"
        trajectory: TrajectoryLog | None = None
        try:
            trajectory = run_episode_with_retry(
                lambda: self._runner.run(
                    system_prompt,
                    seed_override=seed,
                    expected_observation=problem,
                    temperature_override=temperature_override,
                ),
                context_label=f"GEMSolverModule.forward seed={seed}",
            )
        except Exception as exc:  # noqa: BLE001 — GEM/tool layers raise bare Exception
            logger.warning("GEMSolverModule.forward failed for seed=%s: %r", seed, exc)
            status = "runner_error"

        answer = _extract_final_answer(trajectory) if trajectory is not None else ""

        # GEPA needs a Predict invocation in the DSPy trace to build reflective
        # examples for instruction mutation.  We use a DummyLM so this populates
        # the trace at zero cost (no real LLM call), then patch the trace entry
        # with the runner's actual answer so reflection sees consistent data.
        trace_lm = DummyLM([{"answer": answer}])
        with dspy.context(lm=trace_lm):
            self.predict(problem=problem, seed=seed)

        return dspy.Prediction(
            answer=answer,
            trajectory=trajectory,
            status=status,
            instructions=system_prompt,
        )


def _extract_final_answer(trajectory: TrajectoryLog) -> str:
    if not trajectory.steps:
        return "[no-steps]"
    return trajectory.steps[-1].action
