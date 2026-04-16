"""Minimal LiteLLM + Bedrock tool-call probe.

Calls ``litellm.completion`` directly with a trivial tool schema and an
arithmetic prompt, then dumps whether ``tool_calls`` came back. Use this to
isolate whether the zero-tool-calls failure on orz57k-tool is:

  - A model capability issue (Llama 3.1 8B on Bedrock silently ignores
    the tools schema), or
  - A LiteLLM plumbing issue (the tools argument is being dropped), or
  - A Bedrock Converse API issue (tool_use works for some models but not
    others on the same account).

Runs the same probe against multiple models so we can compare.

Usage:

    uv run python scripts/debug_bedrock_tool_use.py
    uv run python scripts/debug_bedrock_tool_use.py --models bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, cast

from litellm import completion  # pyright: ignore[reportPrivateImportUsage]

from trajectory_aware_gym.adapters.gem_episode_runner import _build_completion_kwargs
from trajectory_aware_gym.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_MODELS = [
    "bedrock/us.meta.llama3-1-8b-instruct-v1:0",
    "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
]

_PROBE_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an arithmetic helper with access to a python_exec tool. "
            "For the user's question, call the python_exec tool to compute the "
            "answer. Do not answer in prose."
        ),
    },
    {"role": "user", "content": "What is 12 * 7?"},
]

_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Execute a Python snippet and return stdout. "
                "The code argument must print() any value you want to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute (must print output).",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    raw = getattr(message, "tool_calls", None)
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for tc in raw:
        fn = getattr(tc, "function", None)
        out.append(
            {
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", None),
                "function_name": getattr(fn, "name", None) if fn else None,
                "function_arguments": getattr(fn, "arguments", None) if fn else None,
            }
        )
    return out


def probe(model_id: str, *, temperature: float = 0.0, max_tokens: int = 256) -> dict[str, Any]:
    """Hit one model with tools enabled and return a structured summary."""
    kwargs = _build_completion_kwargs(model_id, temperature=temperature, max_tokens=max_tokens)
    kwargs["tools"] = _PROBE_TOOLS
    kwargs["num_retries"] = 0

    print(f"\n{'=' * 80}\nProbing: {model_id}\n{'=' * 80}")
    print(
        f"Completion kwargs: {json.dumps({k: v for k, v in kwargs.items() if k != 'tools'}, default=str)}"
    )
    print(f"Tools included: {[t['function']['name'] for t in kwargs['tools']]}")

    try:
        response = completion(messages=_PROBE_MESSAGES, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return {
            "model_id": model_id,
            "error": f"{type(exc).__name__}: {exc}",
            "tool_calls": [],
        }

    # LiteLLM returns either ModelResponse or a streaming wrapper; we never
    # stream here, so ModelResponse is guaranteed.
    response_any = cast(Any, response)
    choice = response_any.choices[0]
    message = choice.message
    text_content = getattr(message, "content", None) or ""
    tool_calls = _extract_tool_calls(message)
    finish_reason = getattr(choice, "finish_reason", None)

    print(f"\nFinish reason: {finish_reason}")
    print(f"Text content ({len(text_content)} chars): {text_content[:400]!r}")
    print(f"Tool calls emitted: {len(tool_calls)}")
    for idx, tc in enumerate(tool_calls):
        print(f"  [{idx}] name={tc['function_name']!r} args={tc['function_arguments']!r}")

    # Surface any reasoning/thought fields we might be ignoring.
    for extra_field in ("reasoning_content", "thinking", "tool_use"):
        val = getattr(message, extra_field, None)
        if val:
            preview = str(val)[:300]
            print(f"Extra field `{extra_field}`: {preview!r}")

    usage = getattr(response_any, "usage", None)
    return {
        "model_id": model_id,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls,
        "text_len": len(text_content),
        "text_preview": text_content[:200],
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=_DEFAULT_MODELS,
        help="LiteLLM model ids to probe (default: llama3-1-8b + claude-sonnet-4-5).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--verbose-litellm", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.verbose_litellm:
        import litellm

        litellm.set_verbose = True  # type: ignore[attr-defined]

    if any(m.startswith("bedrock/") for m in args.models):
        settings.validate_aws()

    summary = []
    for model_id in args.models:
        summary.append(probe(model_id, temperature=args.temperature, max_tokens=args.max_tokens))

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    for row in summary:
        tool_count = len(row.get("tool_calls", []))
        print(
            f"  {row['model_id']:<60}  "
            f"tool_calls={tool_count}  "
            f"finish={row.get('finish_reason')!r}  "
            f"error={row.get('error')}"
        )


if __name__ == "__main__":
    main()
