"""Token-based cost computation from pricing config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from caw.models import UsageStats

_pricing_cache: dict[str, Any] | None = None


def _load_pricing() -> dict[str, Any]:
    global _pricing_cache
    if _pricing_cache is None:
        path = Path(__file__).parent / "pricing.json"
        if path.exists():
            _pricing_cache = json.loads(path.read_text())
        else:
            _pricing_cache = {}
    return _pricing_cache


def compute_cost(agent: str, model: str, usage: UsageStats) -> float:
    """Compute cost in USD from token counts and pricing config."""
    pricing = _load_pricing().get(agent, {}).get(model, {})
    cost = (
        usage.input_tokens * pricing.get("input", 0.0)
        + usage.cache_read_tokens * pricing.get("cached_input", 0.0)
        + usage.output_tokens * pricing.get("output", 0.0)
    ) / 1_000_000
    return cost
