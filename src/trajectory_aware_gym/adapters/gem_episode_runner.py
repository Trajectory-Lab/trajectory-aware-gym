"""Concrete GEM episode runner for prompt-conditioned trajectory execution."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from litellm import completion, completion_cost  # type: ignore[import-untyped]

from trajectory_aware_gym.adapters.tool_runtime import ToolRuntime
from trajectory_aware_gym.adapters.trajectory_logger import (
    LLMCallMetadata,
    ToolCall,
    TrajectoryLog,
    TrajectoryLogger,
    load_trajectory,
)
from trajectory_aware_gym.config import settings

DEFAULT_MAX_RESPONSE_TOKENS = 4096
DEFAULT_MAX_TOOL_ROUNDS = 3
TERMINAL_OBSERVATION = "<TERMINAL>"

type ChatMessage = dict[str, str]

_TOOL_NAME_ALIASES = {
    "web_search": "search",
}


@dataclass(frozen=True)
class GEMEpisodeResult:
    """Structured output for one persisted GEM episode."""

    trajectory: TrajectoryLog
    log_path: Path | None


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
    kwargs: dict[str, Any] = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model_id.startswith("bedrock/"):
        aws_region = getattr(settings.aws, "region", None)
        if aws_region is not None:
            kwargs["aws_region_name"] = aws_region
    if model_id.startswith("ollama_chat/"):
        kwargs["api_base"] = settings.ollama.api_base
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

    response = completion(
        messages=messages,
        **_build_completion_kwargs(
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


def _normalize_tool_name(tool_name: str) -> str:
    return _TOOL_NAME_ALIASES.get(tool_name, tool_name)


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    fenced = stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    candidates = [stripped, fenced]
    for candidate in candidates:
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
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

    def run(
        self,
        prompt: str,
        *,
        episode_index: int = 0,
        seed_override: int | None = None,
        expected_observation: str | None = None,
    ) -> TrajectoryLog:
        """Run one episode and return the validated trajectory log."""
        return self.run_episode(
            prompt,
            episode_index=episode_index,
            seed_override=seed_override,
            expected_observation=expected_observation,
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
            trajectory = load_trajectory(log_path) if log_path is not None else logger.build_log()
            return GEMEpisodeResult(trajectory=trajectory, log_path=log_path)
        finally:
            if hasattr(env, "close"):
                env.close()

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
            response_text, llm_call = generate_smoke_action(
                model_id=self._model_id,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_response_tokens,
            )
            llm_calls.append(llm_call)

            parsed_tool_call = self._parse_tool_call(response_text)
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

        return AgentStepResult(
            action=action,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            conversation=step_history,
        )

    def _compose_system_prompt(self, prompt: str) -> str:
        if not self._tools:
            return prompt

        tool_list = ", ".join(sorted(self._tools))
        return (
            f"{prompt}\n\n"
            f"Available tools: {tool_list}.\n"
            'If you need a tool, respond with JSON only: {{"tool": "<name>", "arguments": {{...}}}}.\n'
            "If no tool is needed, respond with the final environment action only."
        )

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
