"""Run a single GEM episode and persist a validated trajectory log.

Demonstrates the extended trajectory logging schema (F1/F2) with:
- Optional system prompt tracking
- Episode outcome derivation
- Per-step timestamps

Also supports a lightweight experiment smoke mode that:
- loads an ExperimentConfig YAML
- uses a real task model for 1-N environment steps
- records token usage and cost for each LLM call
"""

from __future__ import annotations

import argparse
import importlib
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litellm import completion, completion_cost

from trajectory_aware_gym.adapters import TrajectoryLogger
from trajectory_aware_gym.adapters.tool_runtime import ToolRuntime
from trajectory_aware_gym.adapters.trajectory_logger import LLMCallMetadata, ToolCall
from trajectory_aware_gym.config import settings
from trajectory_aware_gym.models.experiment import ExperimentConfig

DEFAULT_ENVIRONMENT_ID = "game:GuessTheNumber-v0-easy"
DEFAULT_GUESS_SEED = 123
DEFAULT_SMOKE_MAX_STEPS = 1
DEFAULT_SMOKE_MAX_TOKENS = 2048
DEFAULT_SMOKE_SYSTEM_PROMPT = (
    "You are a math problem solver. "
    "Solve the problem step by step, then give your final answer "
    "inside \\boxed{}.  For example: \\boxed{42}"
)


@dataclass(frozen=True)
class SmokeRunSpec:
    """Resolved runtime parameters for an experiment smoke run."""

    environment_id: str
    experiment_name: str | None
    model_id: str
    seed: int | None
    max_steps: int
    temperature: float
    system_prompt: str


@dataclass(frozen=True)
class SmokeEpisodeDetail:
    """Result of a single smoke episode."""

    problem: str
    action: str
    reward: float
    correct: bool
    tokens: int
    log_path: str


@dataclass(frozen=True)
class SmokeRunResult:
    """Summary of a completed smoke run."""

    environment_id: str
    experiment_name: str | None
    model_id: str
    total_reward: float
    episodes: int
    correct_count: int
    total_tokens: int
    total_cost_usd: float
    details: list[SmokeEpisodeDetail]


def choose_guess(observation: str, low: int, high: int) -> tuple[int, int, int]:
    """Choose next guess based on environment hint text."""
    normalized = observation.lower()
    if "higher than" in normalized:
        marker = "higher than"
        pivot = int(normalized.split(marker, maxsplit=1)[1].split(".", maxsplit=1)[0].strip())
        low = max(low, pivot + 1)
    elif "lower than" in normalized:
        marker = "lower than"
        pivot = int(normalized.split(marker, maxsplit=1)[1].split(".", maxsplit=1)[0].strip())
        high = min(high, pivot - 1)

    guess = (low + high) // 2
    return guess, low, high


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML into the frozen schema."""
    return ExperimentConfig.from_yaml(path)


def build_smoke_run_spec(args: argparse.Namespace) -> SmokeRunSpec:
    """Resolve smoke-run settings from CLI args and optional experiment config."""
    config: ExperimentConfig | None = None
    if args.experiment_config is not None:
        config = load_experiment_config(args.experiment_config)

    if config is None and args.task_model_id is None:
        raise ValueError("--task-model-id is required for smoke mode without --experiment-config")

    environment_id = (
        args.environment
        or (config.environment.gem_env_id if config is not None else None)
        or DEFAULT_ENVIRONMENT_ID
    )
    seed = (
        args.seed
        if args.seed is not None
        else (config.seeds.data_seed if config is not None else None)
    )
    model_id = args.task_model_id or config.task_models[0].model_id
    max_steps = args.max_steps if args.max_steps is not None else DEFAULT_SMOKE_MAX_STEPS

    if args.temperature is not None:
        temperature = args.temperature
    elif config is not None:
        temperature = (
            config.eval_protocol.temperature_train
            if args.mode == "train"
            else config.eval_protocol.temperature_eval
        )
    else:
        temperature = settings.gem.temperature_eval

    system_prompt = args.system_prompt or DEFAULT_SMOKE_SYSTEM_PROMPT

    return SmokeRunSpec(
        environment_id=environment_id,
        experiment_name=config.name if config is not None else None,
        model_id=model_id,
        seed=seed,
        max_steps=max_steps,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def build_smoke_messages(
    *,
    observation: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Construct a minimal chat prompt for one environment step."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation},
    ]


def _extract_text_content(content: Any) -> str:
    """Normalize LiteLLM message content into plain text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())

    return str(content).strip()


def _build_completion_kwargs(
    model_id: str,
    *,
    temperature: float,
    max_tokens: int = DEFAULT_SMOKE_MAX_TOKENS,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model_id.startswith("ollama_chat/"):
        kwargs["api_base"] = settings.ollama.api_base
    return kwargs


def generate_smoke_action(
    *,
    model_id: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, LLMCallMetadata]:
    """Run one LLM completion and convert usage into trajectory metadata."""
    if model_id.startswith("bedrock/"):
        settings.validate_aws()

    response = completion(
        messages=messages,
        **_build_completion_kwargs(
            model_id,
            temperature=temperature,
        ),
    )
    msg = response.choices[0].message
    action = _extract_text_content(msg.content)
    if not action:
        # Qwen3 thinking mode: answer may only appear in reasoning_content
        reasoning = getattr(msg, "reasoning_content", None) or ""
        action = _extract_text_content(reasoning) if reasoning else "[empty-action]"

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

    cost_usd: float | None = None
    try:
        maybe_cost = completion_cost(completion_response=response)
        cost_usd = float(maybe_cost)
    except (KeyError, TypeError, ValueError):
        cost_usd = None

    return action, LLMCallMetadata(
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def run_episode(
    environment_id: str,
    seed: int | None = None,
    system_prompt: str | None = None,
) -> tuple[float, int, str]:
    """Execute one episode and save a trajectory log to disk."""
    gem = importlib.import_module("gem")
    importlib.import_module("gem.envs")
    env = gem.make(environment_id)

    reset_kwargs = {"seed": seed} if seed is not None else {}
    observation, info = env.reset(**reset_kwargs)

    logger = TrajectoryLogger(environment_id=environment_id, seed=seed)

    tool_runtime = ToolRuntime()

    if system_prompt:
        logger.set_system_prompt(system_prompt)
    logger.set_initial_state(observation=observation, info=info)

    total_reward = 0.0
    low, high = 1, 10

    for _ in range(settings.gem.max_steps):
        # --- Update bounds from last observation, then compute guess via tool ---
        guess, low, high = choose_guess(observation, low, high)

        tool_call = {
            "tool": "python_exec",
            "arguments": {
                "code": textwrap.dedent(f"""\
                    low = {low}
                    high = {high}
                    print((low + high) // 2)
                """).strip(),
            },
        }

        tool_result = tool_runtime.execute(tool_call)

        logged_tool_call = ToolCall(
            tool_name=tool_call["tool"],
            tool_input=json.dumps(tool_call.get("arguments", {})),
            tool_output=json.dumps(tool_result),
            success=tool_result.get("status") == "success",
        )

        if tool_result["status"] == "success":
            guess = int(tool_result["output"].strip())

        action = f"\\\\boxed{{{guess}}}"

        observation, reward, terminated, truncated, step_info = env.step(action)
        logger.add_step(
            action=action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=step_info,
            tool_calls=[logged_tool_call],
        )

        total_reward += reward
        if terminated or truncated:
            break

    log_path = logger.save()

    if hasattr(env, "close"):
        env.close()

    return total_reward, len(logger.steps), str(log_path)


def run_smoke_episode(spec: SmokeRunSpec) -> SmokeRunResult:
    """Run N independent episodes against a single-turn GEM environment."""
    gem = importlib.import_module("gem")
    importlib.import_module("gem.envs")
    env = gem.make(spec.environment_id)

    details: list[SmokeEpisodeDetail] = []
    total_reward = 0.0
    total_tokens = 0
    total_cost = 0.0
    correct_count = 0

    for ep in range(spec.max_steps):
        reset_kwargs = {"seed": spec.seed + ep} if spec.seed is not None else {}
        observation, info = env.reset(**reset_kwargs)

        logger = TrajectoryLogger(environment_id=spec.environment_id, seed=spec.seed)
        if spec.system_prompt:
            logger.set_system_prompt(spec.system_prompt)

        initial_info = dict(info) if isinstance(info, dict) else {}
        if spec.experiment_name is not None:
            initial_info["experiment_name"] = spec.experiment_name
        initial_info["smoke_test"] = True
        initial_info["task_model_id"] = spec.model_id
        initial_info["episode_index"] = ep
        logger.set_initial_state(observation=observation, info=initial_info)

        messages = build_smoke_messages(
            observation=observation,
            system_prompt=spec.system_prompt,
        )
        action, llm_call = generate_smoke_action(
            model_id=spec.model_id,
            messages=messages,
            temperature=spec.temperature,
        )

        _, reward, terminated, truncated, step_info = env.step(action)
        logger.add_step(
            action=action,
            observation="<TERMINAL>",
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=step_info,
            llm_calls=[llm_call],
        )
        log_path = logger.save()

        is_correct = reward > 0
        total_reward += reward
        total_tokens += llm_call.total_tokens
        total_cost += llm_call.cost_usd or 0.0
        if is_correct:
            correct_count += 1

        details.append(
            SmokeEpisodeDetail(
                problem=observation,
                action=action,
                reward=reward,
                correct=is_correct,
                tokens=llm_call.total_tokens,
                log_path=str(log_path),
            )
        )

    if hasattr(env, "close"):
        env.close()

    return SmokeRunResult(
        environment_id=spec.environment_id,
        experiment_name=spec.experiment_name,
        model_id=spec.model_id,
        total_reward=total_reward,
        episodes=len(details),
        correct_count=correct_count,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        details=details,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run one GEM episode and log trajectory.")
    parser.add_argument(
        "--environment",
        default=None,
        help="GEM environment id",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for environment reset")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt to log")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the lightweight experiment smoke path instead of the GuessTheNumber heuristic path",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=None,
        help="Path to an experiment YAML, e.g. experiments/quick-test/config.yaml",
    )
    parser.add_argument(
        "--task-model-id",
        type=str,
        default=None,
        help="Full LiteLLM model string, e.g. bedrock/<inference-profile-id>",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "eval"),
        default="eval",
        help="Which experiment temperature to use in smoke mode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Number of independent episodes to run (defaults to 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the default smoke temperature",
    )
    parser.add_argument("--show-log", action="store_true", help="Print the full JSON log")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.smoke or args.experiment_config is not None or args.task_model_id is not None:
        spec = build_smoke_run_spec(args)
        result = run_smoke_episode(spec)

        print(f"\n{'=' * 70}")
        print(f"Experiment:  {result.experiment_name or '(ad hoc smoke run)'}")
        print(f"Environment: {result.environment_id}")
        print(f"Task model:  {result.model_id}")
        print(f"{'=' * 70}")

        for i, d in enumerate(result.details, 1):
            status = "CORRECT" if d.correct else "WRONG"
            problem_short = d.problem[:80] + ("..." if len(d.problem) > 80 else "")
            action_short = d.action[:120] + ("..." if len(d.action) > 120 else "")
            print(f"\n--- Episode {i}/{result.episodes} [{status}] ---")
            print(f"  Problem:  {problem_short}")
            print(f"  Action:   {action_short}")
            print(f"  Reward:   {d.reward:.1f}  |  Tokens: {d.tokens}")

        accuracy = result.correct_count / result.episodes if result.episodes else 0
        print(f"\n{'=' * 70}")
        print(f"Summary: {result.correct_count}/{result.episodes} correct ({accuracy:.0%})")
        print(f"Total tokens: {result.total_tokens}  |  Total cost: ${result.total_cost_usd:.6f}")
        print(f"{'=' * 70}")
        log_path = result.details[-1].log_path if result.details else None
    else:
        environment_id = args.environment or DEFAULT_ENVIRONMENT_ID
        seed = args.seed if args.seed is not None else DEFAULT_GUESS_SEED
        total_reward, steps, log_path = run_episode(
            environment_id,
            seed,
            args.system_prompt,
        )
        print(f"Environment:    {environment_id}")
        print(f"Steps:          {steps}")
        print(f"Total reward:   {total_reward:.3f}")
        print(f"Trajectory log: {log_path}")

    if args.show_log and log_path:
        payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
        print(f"\nSchema version: {payload['schema_version']}")
        print(f"Outcome:        {payload['episode_outcome']}")
        print(f"System prompt:  {payload.get('system_prompt', '(none)')}")


if __name__ == "__main__":
    main()
