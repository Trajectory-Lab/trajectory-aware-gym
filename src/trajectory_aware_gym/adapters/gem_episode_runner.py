"""Concrete GEM episode runner for prompt-conditioned trajectory execution."""

from __future__ import annotations

import importlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from litellm import completion, completion_cost  # type: ignore[import-untyped]
from litellm.exceptions import (  # type: ignore[import-untyped]
    APIConnectionError as LiteLLMConnectionError,
)
from litellm.exceptions import (
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.exceptions import (
    Timeout as LiteLLMTimeout,
)
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
)

from trajectory_aware_gym.adapters.tool_runtime import ToolRuntime
from trajectory_aware_gym.adapters.trajectory_logger import (
    LLMCallMetadata,
    ToolCall,
    TrajectoryLog,
    TrajectoryLogger,
    load_trajectory,
)
from trajectory_aware_gym.config import settings
from trajectory_aware_gym.metrics import EpisodeRawMetrics, extract_episode_raw_metrics

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESPONSE_TOKENS = 4096
DEFAULT_MAX_TOOL_ROUNDS = 3
TERMINAL_OBSERVATION = "<TERMINAL>"

type ChatMessage = dict[str, str]

_TOOL_NAME_ALIASES = {
    "web_search": "search",
}

# Providers that support OpenAI-style native tool calling via LiteLLM.
_NATIVE_TOOL_PREFIXES = ("bedrock/",)

# LiteLLM exception types that indicate transient, retryable failures.
_RETRYABLE_EXCEPTIONS = (
    RateLimitError,  # 429 — throttling
    ServiceUnavailableError,  # 503 — capacity pressure
    InternalServerError,  # 500 — transient server error
    LiteLLMTimeout,  # request timeout
    LiteLLMConnectionError,  # connection-level failures
)

# ── Inference concurrency semaphore ─────────────────────────────
_inference_semaphore: threading.Semaphore | None = None
_semaphore_lock = threading.Lock()


def _get_inference_semaphore() -> threading.Semaphore:
    """Lazily initialise the inference semaphore from ``settings.retry``."""
    global _inference_semaphore  # noqa: PLW0603
    if _inference_semaphore is None:
        with _semaphore_lock:
            if _inference_semaphore is None:
                _inference_semaphore = threading.Semaphore(settings.retry.inference_semaphore_size)
    return _inference_semaphore


def _reset_inference_semaphore() -> None:
    """Reset the module-level semaphore (for test isolation)."""
    global _inference_semaphore  # noqa: PLW0603
    _inference_semaphore = None


def _completion_with_retry(
    *,
    messages: list[ChatMessage],
    completion_kwargs: dict[str, Any],
) -> Any:
    """Call ``litellm.completion()`` with tenacity retry on transient errors.

    Retry parameters are read from ``settings.retry`` at call time so that
    tests can override them via monkeypatch.
    """
    retry_cfg = settings.retry

    if retry_cfg.jitter:
        wait_strategy = wait_exponential_jitter(
            initial=retry_cfg.initial_wait_seconds,
            max=retry_cfg.max_wait_seconds,
            exp_base=retry_cfg.exponential_base,
        )
    else:
        wait_strategy = wait_exponential(
            min=retry_cfg.initial_wait_seconds,
            max=retry_cfg.max_wait_seconds,
            exp_base=retry_cfg.exponential_base,
        )

    retryer = Retrying(
        stop=stop_after_attempt(retry_cfg.max_attempts),
        wait=wait_strategy,
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

    completion_kwargs["num_retries"] = retry_cfg.litellm_num_retries

    semaphore = _get_inference_semaphore()
    for attempt in retryer:
        with attempt:
            with semaphore:
                return completion(messages=messages, **completion_kwargs)


@dataclass(frozen=True)
class GEMEpisodeResult:
    """Structured output for one persisted GEM episode."""

    trajectory: TrajectoryLog
    log_path: Path | None
    raw_metrics: EpisodeRawMetrics


@dataclass(frozen=True)
class AgentStepResult:
    """Action chosen for one environment step plus its metadata."""

    action: str
    llm_calls: list[LLMCallMetadata]
    tool_calls: list[ToolCall]
    conversation: list[ChatMessage]


def build_smoke_messages(
    *,
    observation: str,
    system_prompt: str,
    history: list[ChatMessage] | None = None,
) -> list[ChatMessage]:
    """Construct a chat prompt for the current environment step."""
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": observation})
    return messages


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
    max_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
) -> dict[str, Any]:
    """Build LiteLLM ``completion()`` kwargs for a given provider.

    LiteLLM routes to different providers based on the model_id prefix:
      - ``ollama/``    → local Ollama server (base/completion models, /api/generate)
      - ``bedrock/``   → AWS Bedrock (managed inference)
      - ``sagemaker/`` → AWS SageMaker (custom TGI endpoints)
    """
    kwargs: dict[str, Any] = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # AWS providers need explicit region for LiteLLM's boto3 calls
    if model_id.startswith(("bedrock/", "sagemaker/")):
        aws_region = getattr(settings.aws, "region", None)
        if aws_region is not None:
            kwargs["aws_region_name"] = aws_region
    # Ollama models need the local server URL
    if model_id.startswith("ollama/"):
        kwargs["api_base"] = settings.ollama.api_base
        # Base (completion) models don't have a built-in stop condition;
        # without these, they repeat the prompt/answer indefinitely.
        kwargs["stop"] = ["<|endoftext|>", "<|im_end|>", "\n### User:", "\nHuman:"]
    return kwargs


def generate_smoke_action(
    *,
    model_id: str,
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
) -> tuple[str, LLMCallMetadata]:
    """Run one LLM completion and convert usage into trajectory metadata."""
    if model_id.startswith("bedrock/"):
        settings.validate_aws()

    response = _completion_with_retry(
        messages=messages,
        completion_kwargs=_build_completion_kwargs(
            model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
    response_payload = cast(Any, response)
    msg = response_payload.choices[0].message
    action = _extract_text_content(msg.content)
    if not action:
        reasoning = getattr(msg, "reasoning_content", None) or ""
        action = _extract_text_content(reasoning) if reasoning else "[empty-action]"

    # LiteLLM converts chat messages into a text prompt for completion-style
    # providers. The response often echoes the prompt. Strip provider-specific
    # prefixes so the GEM environment sees only the model's actual answer.
    if model_id.startswith("ollama/"):
        for prefix in ("### Assistant:\n", "### Assistant: ", "Assistant:\n", "Assistant: "):
            if action.startswith(prefix):
                action = action[len(prefix) :].strip()
                break

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

    cost_usd: float | None = None
    try:
        maybe_cost = completion_cost(completion_response=response)
        cost_usd = float(maybe_cost)
    except Exception:  # noqa: BLE001  # LiteLLM raises bare Exception for unmapped models
        cost_usd = None

    return action, LLMCallMetadata(
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def _normalize_tool_name(tool_name: str) -> str:
    return _TOOL_NAME_ALIASES.get(tool_name, tool_name)


def _supports_native_tools(model_id: str) -> bool:
    """Return True if the model provider supports native tool calling via LiteLLM."""
    return model_id.startswith(_NATIVE_TOOL_PREFIXES)


def _build_tool_descriptions(
    tool_runtime: ToolRuntime,
    tool_names: list[str],
) -> str:
    """Build a human-readable tool reference block from MCP tool schemas.

    Included in the text-based system prompt so the model knows what each
    tool does, what arguments it accepts, and what the output looks like —
    regardless of whether native tool calling is active.
    """
    all_schemas = tool_runtime.list_schemas()
    active = [s for s in all_schemas if s["name"] in tool_names]
    if not active:
        return ""

    parts: list[str] = []
    for schema in active:
        name = schema["name"]
        desc = schema.get("description", "").strip()
        params = schema.get("parameters", {})
        required = params.get("required", [])
        props = params.get("properties", {})

        arg_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            req = " (required)" if pname in required else ""
            arg_parts.append(f"    - {pname}: {ptype}{req}")

        block = f"### {name}\n{desc}"
        if arg_parts:
            block += "\n  Arguments:\n" + "\n".join(arg_parts)
        parts.append(block)

    return "\n\n".join(parts)


def _build_litellm_tools(
    tool_runtime: ToolRuntime,
    tool_names: list[str],
) -> list[dict[str, Any]]:
    """Convert ToolRuntime schemas to OpenAI-format tool definitions for LiteLLM."""
    all_schemas = tool_runtime.list_schemas()
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {}),
            },
        }
        for schema in all_schemas
        if schema["name"] in tool_names
    ]


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    """Extract a JSON tool-call object from model output text.

    Handles three cases:
    1. Entire response is a JSON object
    2. JSON wrapped in ```json … ``` fences
    3. JSON object embedded within surrounding prose
    """
    stripped = text.strip()
    if not stripped:
        return None

    # Fast path: entire text is JSON (raw or code-fenced).
    fenced = stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    for candidate in [stripped, fenced]:
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

    # Slow path: scan for a JSON object embedded in surrounding text.
    # Uses raw_decode so string contents (e.g. braces in code) are handled
    # correctly by the JSON parser.
    decoder = json.JSONDecoder()
    search_start = 0
    while search_start < len(stripped):
        brace_pos = stripped.find("{", search_start)
        if brace_pos == -1:
            break
        try:
            payload, end_pos = decoder.raw_decode(stripped, brace_pos)
        except json.JSONDecodeError:
            search_start = brace_pos + 1
            continue
        if isinstance(payload, dict) and "tool" in payload:
            return payload
        search_start = end_pos

    return None


def _format_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    return (
        f"Tool result for {tool_name}:\n"
        f"{json.dumps(result, ensure_ascii=True, sort_keys=True)}\n"
        "Continue reasoning and produce either another tool call or the final environment action."
    )


class GEMEpisodeRunner:
    """Run prompt-conditioned GEM episodes and persist trajectory logs."""

    def __init__(
        self,
        *,
        environment_id: str,
        model_id: str,
        temperature: float,
        max_steps: int,
        max_response_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
        seed: int | None = None,
        experiment_name: str | None = None,
        tools: list[str] | None = None,
        tool_runtime: ToolRuntime | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_response_tokens < 1:
            raise ValueError("max_response_tokens must be at least 1")
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")

        self._environment_id = environment_id
        self._model_id = model_id
        self._temperature = temperature
        self._max_steps = max_steps
        self._max_response_tokens = max_response_tokens
        self._seed = seed
        self._experiment_name = experiment_name
        self._tools = [_normalize_tool_name(tool_name) for tool_name in (tools or [])]
        self._tool_runtime = tool_runtime or ToolRuntime()
        self._max_tool_rounds = max_tool_rounds
        self._episode_history: list[GEMEpisodeResult] = []

        # Pre-build tool descriptions for the text-based prompt path.
        self._tool_descriptions = _build_tool_descriptions(self._tool_runtime, self._tools)

        # Pre-build native tool schemas for providers that support them.
        self._use_native_tools = bool(self._tools) and _supports_native_tools(model_id)
        self._litellm_tools: list[dict[str, Any]] | None = None
        if self._use_native_tools:
            schemas = _build_litellm_tools(self._tool_runtime, self._tools)
            if schemas:
                self._litellm_tools = schemas
            else:
                self._use_native_tools = False

    @property
    def episode_history(self) -> tuple[GEMEpisodeResult, ...]:
        """Immutable view of all episodes executed by this runner instance."""
        return tuple(self._episode_history)

    def clear_episode_history(self) -> None:
        """Reset in-memory episode history."""
        self._episode_history.clear()

    def run(
        self,
        prompt: str,
        *,
        episode_index: int = 0,
        seed_override: int | None = None,
        expected_observation: str | None = None,
    ) -> TrajectoryLog:
        """Run one episode and return the validated trajectory log.

        Used by GEMSolverModule during GEPA training. Persistence is
        disabled to avoid DB thrash across hundreds of GEPA rollouts —
        use ``run_episode(persist=True)`` for eval or debugging.
        """
        return self.run_episode(
            prompt,
            episode_index=episode_index,
            seed_override=seed_override,
            expected_observation=expected_observation,
            persist=False,
        ).trajectory

    def run_episode(
        self,
        prompt: str,
        *,
        episode_index: int = 0,
        seed_override: int | None = None,
        expected_observation: str | None = None,
        persist: bool = True,
    ) -> GEMEpisodeResult:
        """Run one episode and optionally persist its trajectory log."""
        if not prompt.strip():
            raise ValueError("prompt must not be blank")

        gem = importlib.import_module("gem")
        importlib.import_module("gem.envs")
        env = gem.make(self._environment_id)

        resolved_seed = seed_override
        if resolved_seed is None and self._seed is not None:
            resolved_seed = self._seed + episode_index
        reset_kwargs = {"seed": resolved_seed} if resolved_seed is not None else {}
        observation, info = env.reset(**reset_kwargs)
        normalized_observation = str(observation).strip()
        if (
            expected_observation is not None
            and normalized_observation != expected_observation.strip()
        ):
            msg = (
                "Seeded environment observation did not match the expected example problem. "
                f"{resolved_seed=}, expected={expected_observation[:120]!r}, "
                f"actual={normalized_observation[:120]!r}"
            )
            raise ValueError(msg)

        logger = TrajectoryLogger(environment_id=self._environment_id, seed=resolved_seed)
        logger.set_system_prompt(prompt)
        logger.set_initial_state(
            observation=str(observation),
            info=self._build_initial_info(info=info, episode_index=episode_index),
        )

        conversation: list[ChatMessage] = []
        try:
            current_observation = str(observation)
            for _ in range(self._max_steps):
                agent_step = self._run_agent_step(
                    observation=current_observation,
                    system_prompt=prompt,
                    history=conversation,
                )
                next_observation, reward, terminated, truncated, step_info = env.step(
                    agent_step.action
                )
                normalized_observation = self._normalize_observation(
                    observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )
                logger.add_step(
                    action=agent_step.action,
                    observation=normalized_observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=dict(step_info) if isinstance(step_info, dict) else {},
                    tool_calls=agent_step.tool_calls,
                    llm_calls=agent_step.llm_calls,
                )
                conversation = agent_step.conversation
                if terminated or truncated:
                    break
                current_observation = str(next_observation)

            log_path = logger.save() if persist else None
            if log_path is not None:
                trajectory = load_trajectory(log_path, run_id=logger.last_run_id)
            else:
                trajectory = logger.build_log()
            raw_metrics = extract_episode_raw_metrics(trajectory)
            result = GEMEpisodeResult(
                trajectory=trajectory, log_path=log_path, raw_metrics=raw_metrics
            )
            self._episode_history.append(result)
            return result
        finally:
            if hasattr(env, "close"):
                env.close()

    def _generate_action(
        self,
        messages: list[ChatMessage],
        *,
        include_tools: bool = True,
    ) -> tuple[str, LLMCallMetadata, list[dict[str, Any]]]:
        """Run one LLM completion, returning text, metadata, and native tool calls.

        When the provider supports native tool calling and tool schemas are
        configured, they are passed via LiteLLM's ``tools`` parameter.  The
        returned ``native_tool_calls`` list contains ``{"tool": …, "arguments":
        {…}}`` dicts extracted from ``message.tool_calls`` (empty when the
        provider doesn't return structured calls or none were emitted).

        Set *include_tools* to False to suppress native tool schemas (e.g. to
        force a text-only response after tool rounds are exhausted).
        """
        if self._model_id.startswith("bedrock/"):
            settings.validate_aws()

        completion_kwargs = _build_completion_kwargs(
            self._model_id,
            temperature=self._temperature,
            max_tokens=self._max_response_tokens,
        )
        if include_tools and self._litellm_tools:
            completion_kwargs["tools"] = self._litellm_tools

        response = _completion_with_retry(
            messages=messages,
            completion_kwargs=completion_kwargs,
        )
        response_payload = cast(Any, response)
        msg = response_payload.choices[0].message
        action = _extract_text_content(msg.content) if msg.content else ""
        if not action:
            reasoning = getattr(msg, "reasoning_content", None) or ""
            action = _extract_text_content(reasoning) if reasoning else "[empty-action]"

        if self._model_id.startswith("ollama/"):
            for prefix in ("### Assistant:\n", "### Assistant: ", "Assistant:\n", "Assistant: "):
                if action.startswith(prefix):
                    action = action[len(prefix) :].strip()
                    break

        # Extract native tool calls from the response when available.
        native_tool_calls: list[dict[str, Any]] = []
        raw_tool_calls = getattr(msg, "tool_calls", None)
        if raw_tool_calls:
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                name = getattr(fn, "name", None)
                args_raw = getattr(fn, "arguments", "{}")
                if not isinstance(name, str):
                    continue
                try:
                    arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    arguments = {}
                native_tool_calls.append({"tool": name, "arguments": arguments})

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

        cost_usd: float | None = None
        try:
            maybe_cost = completion_cost(completion_response=response)
            cost_usd = float(maybe_cost)
        except Exception:  # noqa: BLE001  # LiteLLM raises bare Exception for unmapped models
            cost_usd = None

        metadata = LLMCallMetadata(
            model_id=self._model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )
        return action, metadata, native_tool_calls

    def _run_agent_step(
        self,
        *,
        observation: str,
        system_prompt: str,
        history: list[ChatMessage],
    ) -> AgentStepResult:
        tool_calls: list[ToolCall] = []
        llm_calls: list[LLMCallMetadata] = []
        step_history = list(history)
        current_observation = observation
        action = "[empty-action]"

        for _ in range(self._max_tool_rounds):
            messages = build_smoke_messages(
                observation=current_observation,
                system_prompt=self._compose_system_prompt(system_prompt),
                history=step_history,
            )
            response_text, llm_call, native_tool_calls = self._generate_action(messages)
            llm_calls.append(llm_call)

            # Native tool calls take priority; fall back to text-based parsing.
            parsed_tool_call = self._resolve_tool_call(response_text, native_tool_calls)
            if parsed_tool_call is None:
                action = response_text
                step_history.extend(
                    [
                        {"role": "user", "content": current_observation},
                        {"role": "assistant", "content": action},
                    ]
                )
                break

            tool_name = parsed_tool_call["tool"]
            tool_result = self._tool_runtime.execute(parsed_tool_call)
            tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    tool_input=json.dumps(parsed_tool_call.get("arguments", {}), ensure_ascii=True),
                    tool_output=json.dumps(tool_result, ensure_ascii=True, sort_keys=True),
                    success=tool_result.get("status") == "success",
                )
            )
            step_history.extend(
                [
                    {"role": "user", "content": current_observation},
                    {"role": "assistant", "content": response_text},
                ]
            )
            current_observation = _format_tool_result(tool_name, tool_result)

        # Tool rounds exhausted without a text answer — one final call
        # without tool schemas to force a text-only response.
        if action == "[empty-action]" and tool_calls:
            messages = build_smoke_messages(
                observation=current_observation,
                system_prompt=system_prompt,
                history=step_history,
            )
            response_text, llm_call, _ = self._generate_action(messages, include_tools=False)
            llm_calls.append(llm_call)
            action = response_text
            step_history.extend(
                [
                    {"role": "user", "content": current_observation},
                    {"role": "assistant", "content": action},
                ]
            )

        return AgentStepResult(
            action=action,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            conversation=step_history,
        )

    def _compose_system_prompt(self, prompt: str) -> str:
        if not self._tools:
            return prompt

        # Always include text-based tool instructions regardless of native
        # support — smaller models (e.g. Llama 8B) inconsistently trigger
        # native tool calls, so the JSON fallback path must stay active.
        # Tool descriptions are pulled from MCP docstrings at init time.
        parts = [
            prompt,
            "## Available Tools",
            self._tool_descriptions,
            "## How to call a tool",
            'Respond with JSON only: {"tool": "<name>", "arguments": {...}}.',
            "If no tool is needed, respond with the final environment action only.",
        ]
        return "\n\n".join(parts)

    def _resolve_tool_call(
        self,
        response_text: str,
        native_tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Pick the best tool call from native or text-parsed sources.

        Validation here is deliberately minimal: we only check the first
        native call (matches provider convention of one tool call per
        response) and fall back to text-JSON if it is unusable. We do *not*
        scan past the first native call or attempt schema repair — the goal
        of GEPA is to evolve prompts that produce well-formed tool calls,
        and silently rescuing malformed ones contaminates that signal.
        Tool-side validation errors are surfaced naturally via
        ``ToolRuntime.execute`` as structured tool errors.
        """
        if native_tool_calls:
            tc = native_tool_calls[0]
            normalized = _normalize_tool_name(tc["tool"])
            arguments = tc.get("arguments", {})
            if normalized in self._tools and isinstance(arguments, dict):
                return {"tool": normalized, "arguments": arguments}
        return self._parse_tool_call(response_text)

    def _parse_tool_call(self, response_text: str) -> dict[str, Any] | None:
        payload = _extract_json_payload(response_text)
        if payload is None:
            return None

        tool_name = payload.get("tool")
        arguments = payload.get("arguments", {})
        if not isinstance(tool_name, str) or not isinstance(arguments, dict):
            return None

        normalized_name = _normalize_tool_name(tool_name)
        if normalized_name not in self._tools:
            return None

        return {"tool": normalized_name, "arguments": arguments}

    def _build_initial_info(self, *, info: Any, episode_index: int) -> dict[str, Any]:
        initial_info = dict(info) if isinstance(info, dict) else {}
        if self._experiment_name is not None:
            initial_info["experiment_name"] = self._experiment_name
        initial_info["task_model_id"] = self._model_id
        initial_info["episode_index"] = episode_index
        return initial_info

    @staticmethod
    def _normalize_observation(*, observation: Any, terminated: bool, truncated: bool) -> str:
        normalized = str(observation).strip()
        if normalized:
            return normalized
        if terminated or truncated:
            return TERMINAL_OBSERVATION
        return "[empty-observation]"
