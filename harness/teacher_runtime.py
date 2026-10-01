"""Adapter that runs teacher tool calls through the current FinanceBench harness."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any


TOOL_NAMES = (
    "bm25_search",
    "grep_document",
    "search_tables",
    "read",
    "read_table",
    "finish",
)


@dataclass(frozen=True)
class TeacherToolResult:
    content: str
    should_stop: bool


class TeacherHarnessSession:
    """One stateful episode using the exact tools and scorer used by training."""

    def __init__(self, index: Any, max_turns: int = 8):
        import finance_env as fe

        self._finance_env = fe
        self.tools = fe.Bm25Tool(index, max_turns=max_turns)

    @property
    def system_prompt(self) -> str:
        return self._finance_env.FINANCE_TASK_INSTRUCTIONS

    @property
    def accepted(self) -> dict[str, Any] | None:
        return self.tools.state.accepted

    @property
    def state(self) -> Any:
        return self.tools.state

    def tool_specs(self, include_finish: bool = True) -> list[dict[str, Any]]:
        names = TOOL_NAMES if include_finish else TOOL_NAMES[:-1]
        return [getattr(self.tools, name).to_spec() for name in names]

    def start_turn(self) -> None:
        self.tools.state.turn += 1

    def execute(self, name: str, arguments: dict[str, Any], call_id: str) -> TeacherToolResult:
        from tinker_cookbook.tool_use.types import ToolInput

        if name not in TOOL_NAMES:
            return TeacherToolResult(f'{{"error":"Unknown tool {name}"}}', False)
        tool = self._finance_env.CoercingTool(getattr(self.tools, name))
        result = asyncio.run(tool.run(ToolInput(arguments=arguments, call_id=call_id)))
        content = "\n".join(str(message.get("content", "")) for message in result.messages)
        return TeacherToolResult(content=content, should_stop=bool(result.should_stop))

    def score(
        self,
        target: dict[str, Any],
        retrieval_weight: float = 0.0,
        *,
        question: str = "",
        judge: Any = None,
        judge_confidence_threshold: float = 0.85,
    ) -> dict[str, Any]:
        if os.environ.get('FINANCEBENCH_SCORER', 'components') != 'legacy':
            from component_reward import score_live
            result = asyncio.run(score_live(question=question, target=target, submission=self.accepted,
                                             receipts=self.state.receipts, judge=judge))
            self.state.last_score = result
            return result
        from workshop_reward import score_submission

        result = score_submission(target, self.accepted, self.state, retrieval_weight)
        if not (result.get("unresolved") and judge is not None and self.accepted):
            return result
        gold = json.dumps(target.get("required_facts") or target.get("aliases", []), ensure_ascii=False)
        evidence = "\n".join(receipt.get("text", "") for receipt in self.state.receipts.values())
        try:
            verdict = asyncio.run(judge.judge(
                question=question,
                gold=gold,
                candidate=self.accepted.get("answer_text", ""),
                evidence=evidence,
            ))
        except Exception as exc:
            result["judge_error"] = type(exc).__name__
            return result
        confidence = float(verdict.get("confidence", 0.0))
        result.update({
            "judge_verdict": verdict.get("verdict", "ambiguous"),
            "judge_confidence": confidence,
            "judge_cache_hit": bool(verdict.get("cache_hit", False)),
            "judge_numeric_ok": bool(verdict.get("numeric_ok", False)),
        })
        decisive = confidence >= judge_confidence_threshold and (
            verdict.get("verdict") == "contradicted"
            or (verdict.get("verdict") == "entailed" and verdict.get("numeric_ok") is True)
        )
        if decisive:
            result["A"] = float(verdict.get("verdict") == "entailed")
            result["unresolved"] = False
            result["correct"] = int(result["A"] == 1)
            result["grounded_success"] = int(result["correct"] and result["G"] == 1)
            result["reward"] = 0.0 if result["fabricated_citation"] else (
                (1 - retrieval_weight) * result["A"] * (0.5 + 0.5 * result["G"])
                + retrieval_weight * result["Ret"]
            )
        return result


def submission_answer_text(submission: dict[str, Any] | None) -> str:
    """Render a typed submission only for human-readable diagnostics."""
    if not submission:
        return ""
    kind = submission.get("answer_type")
    if kind == "numeric":
        parts = [submission.get("value", ""), submission.get("unit", ""), submission.get("scale", "")]
        return " ".join(str(part) for part in parts if str(part).strip())
    if kind == "boolean":
        return str(submission.get("decision", ""))
    return str(submission.get("answer_text", ""))
