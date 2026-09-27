"""Deterministic process guidance for FinanceBench agent rollouts.

This module deliberately scores tool-use process rather than replacing answer or
evidence correctness. It supports guardrail observations for repeated searches and
weak phase-level alignment to filtered teacher traces without imitating an exact
tool-call order or exposing hidden chain-of-thought.
"""
from __future__ import annotations

import json
import re
from typing import Any

SEARCH_TOOLS = {"bm25_search", "search_tables", "grep_document"}
EVIDENCE_TOOLS = {"read", "read_table", "grep_document"}
PHASES = {
    "retrieval": SEARCH_TOOLS,
    "evidence": EVIDENCE_TOOLS,
    "calculation": {"calculate"},
    "conclusion": {"finish"},
}


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize(x) for x in value]
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip().lower())
    return value


def call_signature(call: dict[str, Any]) -> str:
    return json.dumps({"name": call.get("name", ""), "arguments": _normalize(call.get("arguments", {}))}, sort_keys=True, ensure_ascii=False)


def tool_names(calls: list[dict[str, Any]]) -> list[str]:
    return [str(call.get("name", "")) for call in calls]


def phases_for_tools(names: list[str]) -> set[str]:
    return {phase for phase, tools in PHASES.items() if any(name in tools for name in names)}


def guardrail_for(calls: list[dict[str, Any]]) -> str | None:
    """Return a concise intervention after a repeated or myopic search pattern."""
    if not calls:
        return None
    current = calls[-1]
    current_name = str(current.get("name", ""))
    previous = calls[:-1]
    signature = call_signature(current)
    repeats = sum(call_signature(call) == signature for call in previous)
    if repeats >= 1:
        return "You repeated an equivalent tool call. Change the query or tool, inspect bounded evidence, calculate if needed, or finish; do not repeat the same search."
    recent_names = tool_names(previous[-2:])
    if current_name in SEARCH_TOOLS and len(recent_names) == 2 and all(name in SEARCH_TOOLS for name in recent_names):
        return "You have searched repeatedly without inspecting evidence. Pivot now to grep_document, read, read_table, calculate, or finish if the evidence is sufficient."
    return None


def process_components(
    trace: list[dict[str, Any]],
    question: str = "",
    answer: str = "",
    guardrail_count: int = 0,
) -> dict[str, float]:
    calls: list[dict[str, Any]] = []
    for message in trace:
        if message.get("role") == "assistant":
            calls.extend(message.get("tool_calls") or [])
    names = tool_names(calls)
    signatures = [call_signature(call) for call in calls]
    repeated_calls = sum(1 for i, sig in enumerate(signatures) if sig in signatures[:i])
    evidence_acquired = float(any(name in EVIDENCE_TOOLS for name in names))
    calculator_used = float("calculate" in names)
    retrieval_used = float(any(name in SEARCH_TOOLS for name in names))
    pivoted = float(retrieval_used and evidence_acquired)
    calculation_needed = bool(re.search(r"(?i)(calculate|difference|percent|percentage|ratio|growth|increase|decrease|margin|average|per share)", question))
    arithmetic_compliance = calculator_used if calculation_needed else 1.0
    premature_answer = float(bool(answer.strip()) and not evidence_acquired and not calculator_used)
    repetition_penalty = min(1.0, repeated_calls / 2.0)
    guardrail_penalty = min(1.0, guardrail_count / 2.0)
    raw = 0.35 * pivoted + 0.20 * evidence_acquired + 0.15 * arithmetic_compliance
    raw -= 0.30 * repetition_penalty + 0.15 * guardrail_penalty + 0.20 * premature_answer
    process_quality = max(0.0, min(1.0, raw))
    return {
        "process_quality": process_quality,
        "retrieval_used": retrieval_used,
        "evidence_acquired": evidence_acquired,
        "pivoted_to_evidence": pivoted,
        "calculator_used": calculator_used,
        "arithmetic_compliance": arithmetic_compliance,
        "repeated_calls": float(repeated_calls),
        "guardrail_count": float(guardrail_count),
        "premature_answer": premature_answer,
    }


def teacher_phase_signal(trace: list[dict[str, Any]], teacher_trace: list[dict[str, Any]] | None) -> float:
    """Weak phase-set agreement; never rewards exact wording or tool order."""
    if not teacher_trace:
        return 0.0
    def calls_from(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        calls=[]
        for message in items:
            if message.get("role") == "assistant":
                calls.extend(message.get("tool_calls") or [])
        return calls
    student_phases=phases_for_tools(tool_names(calls_from(trace)))
    teacher_phases=phases_for_tools(tool_names(calls_from(teacher_trace)))
    if not student_phases or not teacher_phases:
        return 0.0
    return len(student_phases & teacher_phases) / len(teacher_phases)
