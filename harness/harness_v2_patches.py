"""Model-agnostic FinanceBench harness improvements diagnosed from dev12 baseline traces.

1. Clearer numeric `finish` validation message for dimensionless ratios (unit="ratio"/"number").
2. Exact-duplicate tool-call loop-breaker note pointing the agent to existing receipts.
3. Concise system-prompt cue to compute standard financial ratios via `calculate` when not
   explicitly tabulated in the filing.
"""
from __future__ import annotations

import json
from typing import Any

EXTRA_SYSTEM_GUIDANCE = (
    "\nWhen a question asks for a financial ratio or margin (e.g., quick ratio, "
    "inventory turnover, operating/gross margin, CapEx/Revenue, ROA) that is not "
    "explicitly stated in the filing, retrieve the underlying line items from the "
    "financial statements and compute it with `calculate` rather than concluding it is "
    "not disclosed. For dimensionless numeric answers (such as ratios or multiples), "
    "set `unit` to \"ratio\" (or \"number\") and `scale` to \"ones\". Avoid repeating "
    "identical tool calls; once you have the needed receipts, synthesize and call `finish`."
)


def patch_system_prompt(base_prompt: str) -> str:
    return base_prompt + EXTRA_SYSTEM_GUIDANCE


def patch_tool_observation(
    name: str,
    args: dict[str, Any],
    content: str,
    seen_call_sigs: set[str],
    receipt_count: int,
) -> str:
    """Enhance error messages and break exact-duplicate tool-call loops."""
    # 1. Clearer numeric finish error when unit is omitted on a dimensionless ratio.
    if name == "finish" and "Numeric finish requires value, unit and scale" in content:
        return content.replace(
            "Numeric finish requires value, unit and scale",
            "Numeric finish requires non-empty value, unit (use ratio or number for dimensionless ratios), and scale (e.g., ones, thousand, million, billion)",
        )

    # 2. Exact-duplicate non-finish tool-call loop breaker.
    if name != "finish":
        sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        if sig in seen_call_sigs:
            note = (
                f"\n[HARNESS NOTE: You already ran this exact {name} call earlier in this episode "
                f"(you currently hold {receipt_count} source receipts in research state). "
                "Do not repeat identical tool calls — inspect a different page/query, run `calculate`, "
                "or call `finish` with citations to your existing receipts.]"
            )
            return content + note
        seen_call_sigs.add(sig)
    return content
