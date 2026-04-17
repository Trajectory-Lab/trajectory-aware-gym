"""Normalized cost computation for local models without native pricing APIs.

For H2 hypothesis validation, we need to compare compute costs across
providers (Ollama vs Bedrock). Since Ollama has no API pricing, we map
token counts to a Bedrock-equivalent USD amount using reference prices
configured in ``trajectory-aware-gym.yaml``.

The reference prices are manually maintained and sourced from the AWS
Bedrock pricing page. Each entry maps a local model ID to the closest
Bedrock model by parameter count. See the YAML comments for provenance.

Formula
-------
    cost = (prompt_tokens × input_per_1m / 1_000_000)
         + (completion_tokens × output_per_1m / 1_000_000)
"""

from __future__ import annotations


def compute_normalized_cost(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    reference_prices: dict[str, dict[str, float]],
) -> float | None:
    """Compute a Bedrock-equivalent USD cost for a local model.

    Parameters
    ----------
    model_id:
        Full model identifier, e.g. ``"ollama/qwen3-1.7b-base"``.
    prompt_tokens:
        Number of input tokens consumed.
    completion_tokens:
        Number of output tokens generated.
    reference_prices:
        Mapping from model_id to ``{"input_per_1m_tokens": float,
        "output_per_1m_tokens": float}``. Typically loaded from
        ``Settings().cost_normalization.reference_prices``.

    Returns
    -------
    float or None
        The estimated USD cost, or ``None`` if ``model_id`` is not found
        in ``reference_prices``.
    """
    prices = reference_prices.get(model_id)
    if prices is None:
        return None

    # cost = (prompt_tokens × input_per_1m / 1_000_000)
    #       + (completion_tokens × output_per_1m / 1_000_000)
    input_cost = prompt_tokens * prices["input_per_1m_tokens"] / 1_000_000
    output_cost = completion_tokens * prices["output_per_1m_tokens"] / 1_000_000
    return input_cost + output_cost
