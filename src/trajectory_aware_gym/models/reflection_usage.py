"""Typed summary of reflection-LM token and cost usage."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

_COST_DECIMAL_PLACES = 6


class ReflectionUsageSummary(BaseModel, frozen=True):
    """Reflection-LM usage with explicit known-vs-unknown accounting.

    `total_tokens` / `total_cost_usd` are `None` when coverage is partial;
    `known_*` fields always carry the sum over entries that did report data.
    Data-coverage fields are fractions in `[0, 1]`.
    """

    total_tokens: int | None = None
    known_total_tokens: int = Field(default=0, ge=0)
    token_data_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    total_cost_usd: float | None = None
    known_cost_usd: float = Field(default=0.0, ge=0.0)
    cost_data_coverage: float = Field(default=1.0, ge=0.0, le=1.0)

    @classmethod
    def empty(cls) -> ReflectionUsageSummary:
        """Summary for `reflection_lm=None` — nothing to measure, fully accounted."""
        return cls(
            total_tokens=0,
            known_total_tokens=0,
            token_data_coverage=1.0,
            total_cost_usd=0.0,
            known_cost_usd=0.0,
            cost_data_coverage=1.0,
        )

    @classmethod
    def from_saved(cls, payload: Any) -> ReflectionUsageSummary:
        """Reconstruct from a previously serialized `reflection_usage` dict.

        Missing fields fall back to the empty-summary defaults. Unknown keys are
        ignored. Returns an empty summary when `payload` is not a dict.
        """
        if not isinstance(payload, dict):
            return cls.empty()
        merged = cls.empty().model_dump()
        for key, value in payload.items():
            if key in merged:
                merged[key] = value
        return cls.model_validate(merged)

    @classmethod
    def from_history(cls, entries: list[dict[str, Any]]) -> ReflectionUsageSummary:
        """Build from per-call LM history, recording fractional coverage of missing usage/cost.

        Entries without numeric `usage.total_tokens` are counted as unknown tokens;
        entries whose `response` can't be costed are counted as unknown cost. When
        coverage is 1.0 the respective `total_*` field is populated; otherwise it
        remains `None` and only the `known_*` running sum is reported.
        """
        if not entries:
            return cls.empty()

        known_total_tokens = 0
        token_known_entries = 0
        known_cost_usd = 0.0
        cost_known_entries = 0

        for entry in entries:
            usage_total = entry.get("usage_total_tokens")
            if isinstance(usage_total, int | float) and not isinstance(usage_total, bool):
                known_total_tokens += int(usage_total)
                token_known_entries += 1

            entry_cost = entry.get("cost_usd")
            if isinstance(entry_cost, int | float) and not isinstance(entry_cost, bool):
                known_cost_usd += float(entry_cost)
                cost_known_entries += 1

        token_coverage = token_known_entries / len(entries)
        cost_coverage = cost_known_entries / len(entries)
        return cls(
            total_tokens=known_total_tokens if token_coverage == 1.0 else None,
            known_total_tokens=known_total_tokens,
            token_data_coverage=token_coverage,
            total_cost_usd=(
                round(known_cost_usd, _COST_DECIMAL_PLACES) if cost_coverage == 1.0 else None
            ),
            known_cost_usd=round(known_cost_usd, _COST_DECIMAL_PLACES),
            cost_data_coverage=cost_coverage,
        )
