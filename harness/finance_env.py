"""Tinker RL environment for the structured FinanceBench harness."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections.abc import Sequence
from typing import Annotated

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
from tinker_cookbook.tool_use import Tool, ToolInput, ToolResult, build_agent_tool_env, simple_tool_result, tool


def _as_list(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)] if value is not None else []


class CoercingTool:
    """Run a wrapped tool after coercing string list arguments to real lists."""

    def __init__(self, tool: Tool):
        self._tool = tool
        properties = (tool.to_spec().get("parameters") or {}).get("properties") or {}
        self._list_fields = {name for name, spec in properties.items() if spec.get("type") == "array"}

    @property
    def name(self):
        return self._tool.name

    def to_spec(self):
        return self._tool.to_spec()

    async def run(self, call: ToolInput):
        arguments = dict(call.arguments)
        coerced = dict(arguments)
        for key in self._list_fields:
            value = coerced.get(key)
            if not isinstance(value, str):
                continue
            try:
                parsed = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, list):
                coerced[key] = parsed
            elif value.strip():
                coerced[key] = [value]
        forwarded = ToolInput(arguments=coerced, call_id=call.call_id)
        result = await self._tool.run(forwarded)
        # A validated finish is terminal; the environment will grade once and
        # stop sampling instead of allowing later text to obscure submission.
        if self.name == "finish":
            result.should_stop = True
        return result

import sys
sys.path.insert(0, os.path.dirname(__file__))
import financebench_harness as hb
from reward_calculation import RewardConfig, answer_quality_with_judge, build_judge, reward_formula

FINANCE_TASK_INSTRUCTIONS = """You are a financial-filings retrieval agent.

Your job is to answer the user's question using the SEC filing corpus. Search
before answering. Start with bm25_search to identify the likely document and
passages, then use grep_document for exact accounting terms or regexes and
read/read_table to inspect bounded evidence. Use calculate only for arithmetic.
You may switch documents or issue refined searches when the evidence is not
sufficient. Do not invent values or rely on outside knowledge.

Available tools:
- bm25_search: ranked keyword search with optional company/year/document filters
- grep_document: document-scoped exact or regex search; choose grep_type text, pdfgrep, or rga
- search_tables: ranked search over table-like pages in one or all filings
- read: bounded page/passage retrieval with provenance
- read_table: retrieve a table candidate with neighboring headers and footnotes
- calculate: safe arithmetic for ratios, changes, and unit conversions

When you have enough evidence, call finish exactly once. Put the complete answer in answer_text/answer, set answer_type (text, decision, scalar, percentage, currency, date, or multipart) when known, include value/unit/scale for numeric answers, cite evidence_document/evidence_page and citations, and provide calc_id for derived results. Do not stop after a search hit unless it actually resolves the question.
"""


OBS_CAP = 4000
SEARCH_SNIPPET_CHARS = 300
MAX_SEARCH_HITS = 5
READ_VISIBLE_CHARS = 2800


def _compact_text(value: object, limit: int = SEARCH_SNIPPET_CHARS) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _compact_hit(hit: dict, *, kind: str) -> dict:
    """Expose only retrieval fields the model can use and cite."""
    keys = ["document_id", "page", "score", "section", "provenance", "start", "end"]
    if kind == "table":
        keys += ["table_id", "title", "statement_kind", "units"]
    else:
        keys += ["passage_id"]
    out = {key: hit[key] for key in keys if key in hit and hit[key] not in (None, "")}
    out["snippet"] = _compact_text(hit.get("text", hit.get("snippet", "")))
    return out


def _compact_matches(matches: list[dict], limit: int = MAX_SEARCH_HITS) -> list[dict]:
    return [
        {key: value for key, value in {
            "document_id": item.get("document_id"),
            "page": item.get("page"),
            "line": item.get("line"),
            "provenance": item.get("provenance"),
            "snippet": _compact_text(item.get("text", "")),
        }.items() if value not in (None, "")}
        for item in matches[:limit]
    ]


def _bound_read_payload(payload: dict) -> dict:
    """Keep read/table responses structured while exposing a continuation cursor."""
    out = dict(payload)
    text = str(out.get("text", ""))
    start = int(out.get("start", 0) or 0)
    visible = text[:READ_VISIBLE_CHARS]
    out["text"] = visible
    out["end"] = start + len(visible)
    total = int(out.get("total_chars", start + len(text)) or 0)
    out["total_chars"] = max(total, start + len(text))
    out["has_more"] = bool(out["end"] < out["total_chars"] or out.get("has_more", False))
    out["next_start"] = out["end"] if out["has_more"] else None
    if "neighbor_context" in out:
        out["neighbor_context"] = _compact_text(out["neighbor_context"], 450)
    return out


def _tool_result(payload: object) -> ToolResult:
    """Bound every tool observation without returning malformed JSON."""
    raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    if len(raw) <= OBS_CAP:
        return simple_tool_result(raw)
    envelope = {
        "truncated": True,
        "original_chars": len(raw),
        "content": _compact_text(raw, OBS_CAP - 120),
        "warning": "Use the tool's pagination fields or narrower query; content was bounded by OBS_CAP.",
    }
    return simple_tool_result(json.dumps(envelope, ensure_ascii=False))


class Bm25Tool:
    """Structured retrieval tools backed by one deterministic corpus index."""

    def __init__(self, index: hb.StructuredIndex):
        self.index = index

    @staticmethod
    async def build(max_docs: int | None = None, doc_names: list[str] | None = None) -> "Bm25Tool":
        return Bm25Tool(hb.build_index(max_docs=max_docs, doc_names=doc_names))

    @tool
    async def bm25_search(
        self,
        query_list: Annotated[list[str], "One or more keyword queries."],
        company: Annotated[str, "Optional company filter, for example 3M."] = "",
        year: Annotated[int, "Optional fiscal year; use -1 when unknown."] = -1,
        filing_type: Annotated[str, "Optional type such as 10-K or 10-Q."] = "",
        document_id: Annotated[str, "Optional exact filing name."] = "",
        scope: Annotated[str, "prose, tables, or both."] = "both",
        top_k: Annotated[int, "Maximum results to return."] = 3,
    ) -> ToolResult:
        filters = {"company": company, "year": year, "filing_type": filing_type, "document_id": document_id}
        top_k = max(1, min(int(top_k), 8))
        scope = (scope or "both").lower()
        out: dict[str, object] = {"queries": query_list, "filters": filters}
        prose = self.index.search_prose(query_list, filters, top_k) if scope in ("prose", "both") else []
        tables = self.index.search_tables(query_list, filters, top_k) if scope in ("tables", "table", "both") else []
        ranked = [(float(item.get("score", 0.0)), "prose", item) for item in prose]
        ranked += [(float(item.get("score", 0.0)), "table", item) for item in tables]
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:MAX_SEARCH_HITS]
        out["prose_hits"] = [_compact_hit(item, kind="prose") for _, kind, item in selected if kind == "prose"]
        out["table_hits"] = [_compact_hit(item, kind="table") for _, kind, item in selected if kind == "table"]
        out["returned_hits"] = len(selected)
        return _tool_result(out)

    @tool
    async def grep_document(
        self,
        document_id: Annotated[str, "Exact document_id returned by bm25_search."],
        patterns: Annotated[list[str], "Literal or regular-expression patterns."],
        page_start: Annotated[int, "First page to search, or -1 for all pages."] = -1,
        page_end: Annotated[int, "Last page to search, or -1 for all pages."] = -1,
        context_lines: Annotated[int, "Number of surrounding lines per match."] = 2,
        grep_type: Annotated[str, "Backend choice: text, pdfgrep, or rga."] = "text",
    ) -> ToolResult:
        result = self.index.grep_document(document_id, patterns, page_start, page_end, context_lines)
        return _tool_result({"grep_type_requested": grep_type, "backend_used": "page_text", "matches": _compact_matches(result), "returned_matches": min(len(result), MAX_SEARCH_HITS)})

    @tool
    async def search_tables(
        self,
        query_list: Annotated[list[str], "Terms such as revenue, cash flows, balance sheet, or PP&E."],
        document_id: Annotated[str, "Optional exact document_id; leave empty for corpus search."] = "",
        company: Annotated[str, "Optional company filter."] = "",
        year: Annotated[int, "Optional fiscal year; use -1 when unknown."] = -1,
        top_k: Annotated[int, "Maximum table candidates."] = 3,
    ) -> ToolResult:
        filters = {"document_id": document_id, "company": company, "year": year}
        hits = self.index.search_tables(query_list, filters, max(1, min(int(top_k), MAX_SEARCH_HITS)))
        return _tool_result({"queries": query_list, "table_hits": [_compact_hit(item, kind="table") for item in hits[:MAX_SEARCH_HITS]], "returned_hits": min(len(hits), MAX_SEARCH_HITS)})

    @tool
    async def read(
        self,
        document_id: Annotated[str, "Exact document_id."],
        page: Annotated[int, "One-based page number, or -1 when using passage_id."] = -1,
        start: Annotated[int, "Character start within the selected page/window."] = 0,
        end: Annotated[int, "Character end; bounded by the harness."] = hb.MAX_READ,
        passage_id: Annotated[str, "Optional passage_id returned by search."] = "",
    ) -> ToolResult:
        return _tool_result(_bound_read_payload(self.index.read(document_id, page, start, end, passage_id)))

    @tool
    async def read_table(
        self,
        table_id: Annotated[str, "Exact table_id returned by search_tables."],
        include_neighbors: Annotated[bool, "Include nearby page context for headers and footnotes."] = True,
        start: Annotated[int, "Character start within the table body."] = 0,
        end: Annotated[int, "Character end; bounded by the harness."] = hb.MAX_READ,
    ) -> ToolResult:
        return _tool_result(_bound_read_payload(self.index.read_table(table_id, include_neighbors, start, end)))

    @tool
    async def calculate(
        self,
        expression: Annotated[str, "Basic arithmetic expression using numbers and + - * / % **."],
    ) -> ToolResult:
        try:
            value = hb.calculate(expression)
            calc_id = hashlib.sha256(f"{expression}={value}".encode()).hexdigest()[:16]
            return _tool_result({"expression": expression, "value": value, "calc_id": calc_id, "operands_source_backed": False})
        except Exception as exc:
            return _tool_result({"error": str(exc)})


    @tool
    async def finish(
        self,
        answer: Annotated[str, "Compatibility alias for the complete final answer."] = "",
        evidence_document: Annotated[str, "Primary evidence document identifier, if known."] = "",
        evidence_page: Annotated[int, "Primary evidence page, or -1 when unknown."] = -1,
        answer_type: Annotated[str, "text, decision, scalar, percentage, currency, date, or multipart."] = "text",
        value: Annotated[str, "Typed requested value, preserving zero and sign."] = "",
        unit: Annotated[str, "Currency or measurement unit."] = "",
        scale: Annotated[str, "thousand, million, billion, or empty."] = "",
        decision: Annotated[str, "yes/no or other requested decision."] = "",
        answer_text: Annotated[str, "Preferred complete final answer text."] = "",
        citations: Annotated[list[str], "Exact provenance identifiers used in the answer."] = [],
        calc_id: Annotated[str, "Calculation receipt identifier for derived results."] = "",
    ) -> ToolResult:
        final_text = answer_text.strip() or answer.strip()
        return _tool_result({"finish": True, "answer": final_text, "answer_text": final_text, "answer_type": answer_type, "value": value, "unit": unit, "scale": scale, "decision": decision, "evidence_document": evidence_document, "evidence_page": evidence_page, "citations": citations, "calc_id": calc_id})


def _company_name(doc_name: str) -> str:
    import re
    base = doc_name.split("_")[0]
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base)
    if len(s) > 2 and s == s.upper() and not any(ch.isdigit() for ch in s):
        return " ".join(w.capitalize() for w in s.split())
    return s


def _format_question(row: dict) -> str:
    q = row["question"].strip()
    company = row.get("company") or _company_name(row["doc_name"])
    return q if company.lower() in q.lower() else f"About {company}: {q}"


def _message_text(message: Message) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(getattr(item, "text", item)))
        return " ".join(parts)
    return str(content)


class FinanceAnswerReward:
    """Layered FinanceBench reward with finish gating and semantic fallback."""

    def __init__(
        self,
        gold_answers: list[str],
        question: str = "",
        format_coef: float = 0.0,
        judge=None,
        reward_config: RewardConfig | None = None,
        require_finish: bool = True,
    ):
        self.gold_answers = gold_answers
        self.question = question
        self.format_coef = format_coef
        self.judge = judge
        self.reward_config = reward_config or RewardConfig(require_finish=require_finish)
        self.require_finish = require_finish

    @staticmethod
    def _finish_submission(history: list[Message]) -> dict[str, Any]:
        for msg in reversed(history):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                fn = getattr(tc, "function", None)
                if fn is not None:
                    name = getattr(fn, "name", "")
                    raw = getattr(fn, "arguments", "{}")
                elif isinstance(tc, dict):
                    obj = tc.get("function", tc)
                    name = obj.get("name", "")
                    raw = obj.get("arguments", "{}")
                else:
                    continue
                if name != "finish":
                    continue
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, json.JSONDecodeError):
                    args = {}
                if not isinstance(args, dict):
                    continue
                answer = str(args.get("answer_text") or args.get("answer") or "").strip()
                if answer:
                    try:
                        evidence_page = int(args.get("evidence_page", -1) or -1)
                    except (TypeError, ValueError):
                        # Multiple alternatives or non-integral page fields are
                        # validation failures, never a reason to crash rollout.
                        continue
                    return {
                        "answer": answer,
                        "answer_type": str(args.get("answer_type", "text")),
                        "value": str(args.get("value", "")),
                        "unit": str(args.get("unit", "")),
                        "scale": str(args.get("scale", "")),
                        "decision": str(args.get("decision", "")),
                        "citations": args.get("citations", []),
                        "calc_id": str(args.get("calc_id", "")),
                        "evidence_document": str(args.get("evidence_document", "")).strip(),
                        "evidence_page": evidence_page,
                    }
        return {}

    @staticmethod
    def _finish_answer(history: list[Message]) -> str:
        return str(FinanceAnswerReward._finish_submission(history).get("answer", ""))

    @staticmethod
    def _citation_seen(trace: list[dict], document_id: str, page: int) -> bool:
        if not document_id and page < 1:
            return True
        for item in trace:
            if item.get("role") != "tool":
                continue
            content = str(item.get("content", ""))
            if document_id and document_id.lower() not in content.lower():
                continue
            if page < 1:
                return True
            if re.search(rf"(?:page(?:_start|_end)?|pages)\s*[\"']?\s*[:=]\s*{page}(?:\D|$)", content, re.I) or re.search(rf"(?:page(?:_start|_end)?|pages)=?\s*[-:]?\s*{page}(?:\D|$)", content, re.I):
                return True
        return False

    @staticmethod
    def _history_trace(history: list[Message]) -> list[dict]:
        """Convert Tinker messages into the harness trace schema for grounding."""
        trace: list[dict] = []
        pending_tool_names: list[str] = []
        for message in history:
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
            content = _message_text(message) if isinstance(message, dict) else str(message)
            if role == "assistant":
                calls = []
                for call in (message.get("tool_calls") or []) if isinstance(message, dict) else []:
                    if isinstance(call, dict):
                        fn = call.get("function", call)
                        if isinstance(fn, dict):
                            name = fn.get("name", "")
                            raw = fn.get("arguments", {})
                        else:
                            name = getattr(fn, "name", "")
                            raw = getattr(fn, "arguments", {})
                        call_id = call.get("id", call.get("call_id", ""))
                    else:
                        fn = getattr(call, "function", None)
                        name = getattr(fn, "name", "") if fn is not None else getattr(call, "name", "")
                        raw = getattr(fn, "arguments", {}) if fn is not None else getattr(call, "arguments", {})
                        call_id = getattr(call, "id", getattr(call, "call_id", ""))
                    if isinstance(raw, str):
                        try:
                            arguments = json.loads(raw)
                        except (TypeError, json.JSONDecodeError):
                            arguments = {"raw": raw}
                    else:
                        arguments = raw if isinstance(raw, dict) else {"raw": str(raw)}
                    calls.append({"name": name, "arguments": arguments, "call_id": call_id})
                    if name:
                        pending_tool_names.append(name)
                trace.append({"role": "assistant", "content": content, "tool_calls": calls})
            elif role == "tool":
                call_id = message.get("tool_call_id", message.get("call_id", "")) if isinstance(message, dict) else getattr(message, "tool_call_id", "")
                name = message.get("name", "") if isinstance(message, dict) else getattr(message, "name", "")
                if not name and pending_tool_names:
                    name = pending_tool_names.pop(0)
                elif pending_tool_names:
                    pending_tool_names.pop(0)
                trace.append({"role": "tool", "call_id": call_id, "name": name, "content": content})
            else:
                trace.append({"role": role, "content": content})
        return trace

    @staticmethod
    def _empty_metrics(*, used_finish: float, answer_nonempty: float, finish_missing: float) -> dict[str, float | str]:
        return {
            "format": 0.0,
            "correct": 0.0,
            "quality": 0.0,
            "answer_quality": 0.0,
            "evidence_quality": 0.0,
            "grounded_quality": 0.0,
            "conclusion": 0.0,
            "details": 0.0,
            "strong_source": 0.0,
            "used_finish": used_finish,
            "finish_gate": 0.0 if finish_missing else 1.0,
            "finish_missing": finish_missing,
            "finish_penalty": finish_missing,
            "answer_nonempty": answer_nonempty,
            "judge_used": 0.0,
            "judge_confidence": 0.0,
            "judge_cache_hit": 0.0,
            "judge_entailment": 0.0,
            "judge_numeric_ok": 0.0,
            "judge_error_flag": 0.0,
            "hard_gate_veto": 0.0,
            "citation_present": 0.0,
            "citation_valid": 1.0,
        }

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, Any]]:
        submission = self._finish_submission(history)
        submitted = str(submission.get("answer", ""))
        used_finish = 1.0 if submitted else 0.0
        if self.require_finish and not submitted:
            final = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
            nonempty = float(bool(_message_text(final).strip())) if final is not None else 0.0
            return 0.0, self._empty_metrics(used_finish=used_finish, answer_nonempty=nonempty, finish_missing=1.0)
        if submitted:
            text = submitted
            formatted = 1.0
        else:
            final = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
            text = _message_text(final) if final is not None else ""
            formatted = float(bool(re.search(r"^\s*Answer:\s*", text, re.I | re.M)))
        if not text.strip():
            return 0.0, self._empty_metrics(used_finish=used_finish, answer_nonempty=0.0, finish_missing=0.0)

        trace = self._history_trace(history)
        strong_text, weak_text = hb._trace_evidence_text(trace)
        judge_evidence = "\n".join(x for x in (strong_text, weak_text) if x)
        candidates: list[dict[str, Any]] = []
        for gold in self.gold_answers:
            deterministic_quality, answer_parts = hb.score_answer(gold, text, question=self.question)
            quality, judge_parts = await answer_quality_with_judge(
                question=self.question,
                gold=gold,
                candidate=text,
                evidence=judge_evidence,
                judge=self.judge,
                config=self.reward_config,
            )
            evidence, evidence_parts = hb.score_evidence(gold, text, trace, question=self.question)
            citation_document = str(submission.get("evidence_document", ""))
            citation_page = int(submission.get("evidence_page", -1) or -1)
            citation_present = float(bool(citation_document or citation_page >= 1))
            citation_valid = float(self._citation_seen(trace, citation_document, citation_page))
            if citation_present and not citation_valid:
                evidence = 0.0
                evidence_parts["citation_valid"] = 0.0
            else:
                evidence_parts["citation_valid"] = 1.0
            evidence_parts["citation_present"] = citation_present
            citation_veto = bool(citation_present and not citation_valid)
            grounded = 0.0 if citation_veto else reward_formula(quality, evidence)
            candidates.append({
                "quality": quality,
                "deterministic_quality": deterministic_quality,
                "answer_parts": answer_parts,
                "judge_parts": judge_parts,
                "evidence": evidence,
                "evidence_parts": evidence_parts,
                "grounded": grounded,
                "citation_present": citation_present,
                "citation_valid": citation_valid,
                "citation_veto": float(citation_veto),
            })
        best = max(candidates, key=lambda item: item["grounded"], default={})
        answer_parts = best.get("answer_parts", {})
        judge_parts = best.get("judge_parts", {})
        evidence_parts = best.get("evidence_parts", {})
        trace_calls = sum(len(item.get("tool_calls") or []) for item in trace if item.get("role") == "assistant")
        trace_tool_messages = sum(item.get("role") == "tool" for item in trace)
        metrics: dict[str, Any] = {
            "format": formatted,
            "correct": float(best.get("quality", 0.0)),
            "quality": float(best.get("grounded", 0.0)),
            "answer_quality": float(best.get("quality", 0.0)),
            "deterministic_answer_quality": float(best.get("deterministic_quality", 0.0)),
            "evidence_quality": float(best.get("evidence", 0.0)),
            "grounded_quality": float(best.get("grounded", 0.0)),
            "conclusion": float(answer_parts.get("conclusion", 0.0)),
            "details": float(answer_parts.get("details", 0.0)),
            "strong_source": float(evidence_parts.get("strong_source", 0.0)),
            "trace_messages": float(len(trace)),
            "trace_tool_calls": float(trace_calls),
            "trace_tool_messages": float(trace_tool_messages),
            "trace_strong_chars": float(len(strong_text)),
            "trace_weak_chars": float(len(weak_text)),
            "used_finish": used_finish,
            "finish_gate": 1.0,
            "finish_missing": 0.0,
            "finish_penalty": 0.0,
            "answer_nonempty": 1.0,
            "citation_present": float(evidence_parts.get("citation_present", 0.0)),
            "citation_valid": float(evidence_parts.get("citation_valid", 1.0)),
            "citation_veto": float(best.get("citation_veto", 0.0)),
        }
        for key, value in judge_parts.items():
            if key.startswith("judge_") and isinstance(value, (int, float, bool)):
                metrics[key] = float(value)
            elif key == "semantic_length_factor" and isinstance(value, (int, float, bool)):
                metrics[key] = float(value)
        metrics["judge_entailment"] = float(judge_parts.get("judge_verdict") == "entailed")
        metrics["judge_error_flag"] = float(bool(judge_parts.get("judge_error")))
        metrics["hard_gate_veto"] = float(judge_parts.get("hard_gate", "pass") != "pass")
        return float(best.get("grounded", 0.0)), metrics

def _repaired_train_dev_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Reserve 12 stable, document-diverse development questions from train."""
    ordered = sorted(rows, key=lambda row: (str(row.get("doc_name", "")), str(row.get("financebench_id", ""))))
    selected: list[dict] = []
    seen_docs: set[str] = set()
    for row in ordered:
        doc = str(row.get("doc_name", ""))
        if doc not in seen_docs:
            selected.append(row)
            seen_docs.add(doc)
        if len(selected) == 12:
            break
    if len(selected) < 12:
        selected_ids = {row.get("financebench_id") for row in selected}
        selected.extend(row for row in ordered if row.get("financebench_id") not in selected_ids)[: 12 - len(selected)]
    dev_ids = {row.get("financebench_id") for row in selected}
    return [row for row in rows if row.get("financebench_id") not in dev_ids], selected


def load_financebench(split_name: str = "train") -> list[dict]:
    """Load frozen splits plus the deterministic repaired 96/12 train/dev split."""
    split_path = hb.BASE / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        rows = split.get("train", []) if split_name in {"train96", "dev"} else split.get(split_name, [])
    else:
        rows = [json.loads(line) for line in (hb.DATA / "financebench_merged.jsonl").read_text().splitlines()]
    if split_name in {"train96", "dev"}:
        train_rows, dev_rows = _repaired_train_dev_rows(rows)
        rows = train_rows if split_name == "train96" else dev_rows
    out = []
    for row in rows:
        answer = str(row.get("answer") or "").strip()
        if not answer:
            continue
        out.append({
            "question": _format_question(row),
            "answer": [answer],
            "doc": row.get("doc_name", ""),
            "company": row.get("company", ""),
            "financebench_id": row.get("financebench_id", ""),
            "evidence": row.get("evidence", []),
        })
    return out


def _initial_messages(datum: dict, renderer: Renderer, tool_obj: Bm25Tool) -> list[Message]:
    schemas = [
        tool_obj.bm25_search.to_spec(), tool_obj.grep_document.to_spec(),
        tool_obj.search_tables.to_spec(), tool_obj.read.to_spec(),
        tool_obj.read_table.to_spec(), tool_obj.calculate.to_spec(), tool_obj.finish.to_spec(),
    ]
    prefix = renderer.create_conversation_prefix_with_tools(tools=schemas, system_prompt=FINANCE_TASK_INSTRUCTIONS)
    return prefix + [{"role": "user", "content": datum["question"]}]


class FinanceSearchEnvGroupBuilder(EnvGroupBuilder):
    def __init__(self, datum, model_name, renderer_name, max_turns, group_size, tool_obj, format_coef=0.0, max_trajectory_tokens=32 * 1024, max_generation_tokens=1024, judge=None, reward_config: RewardConfig | None = None):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.tool_obj = tool_obj
        self.format_coef = format_coef
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.judge = judge
        self.reward_config = reward_config or RewardConfig.from_env()

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)
        initial_messages = _initial_messages(self.datum, renderer, self.tool_obj)
        reward_fn = FinanceAnswerReward(gold_answers=self.datum["answer"], question=self.datum.get("question", ""), format_coef=self.format_coef, judge=self.judge, reward_config=self.reward_config, require_finish=self.reward_config.require_finish)
        tools = [CoercingTool(t) for t in (self.tool_obj.bm25_search, self.tool_obj.grep_document, self.tool_obj.search_tables, self.tool_obj.read, self.tool_obj.read_table, self.tool_obj.calculate, self.tool_obj.finish)]
        parse_policy = ParseErrorPolicy(max_consecutive=1, penalty_per_error=0.0, terminal_reward=0.0, mask_error_turns=True)
        return [build_agent_tool_env(
            renderer=renderer,
            tools=tools,
            initial_messages=initial_messages,
            reward_fn=reward_fn,
            model_name=self.model_name,
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
            failed_parse_reward=0.0,
            terminate_on_parse_error=True,
            parse_error_policy=parse_policy,
        ) for _ in range(self.group_size)]

    def logging_tags(self) -> list[str]:
        return ["financebench", "structured_sparse_agent"]


class FinanceRLDataset(RLDataset):
    def __init__(self, builders, batch_size):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index):
        s = index * self.batch_size
        return self.builders[s:s + self.batch_size]

    def __len__(self):
        if not self.builders:
            return 0
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class FinanceDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 6
    format_coef: float = 0.0
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int = 1024
    seed: int = 0
    split_name: str = "train"

    async def __call__(self):
        reward_config = RewardConfig.from_env()
        judge = build_judge(reward_config)
        data = load_financebench(self.split_name)
        doc_names = sorted({str(item.get("doc", "")) for item in data if item.get("doc")})
        tool_obj = await Bm25Tool.build(doc_names=doc_names)
        rng = random.Random(self.seed)
        rng.shuffle(data)
        builders = [FinanceSearchEnvGroupBuilder(d, self.model_name_for_tokenizer, self.renderer_name, self.max_turns, self.group_size, tool_obj, self.format_coef, self.max_trajectory_tokens, self.max_generation_tokens, judge=judge, reward_config=reward_config) for d in data]
        return FinanceRLDataset(builders, self.batch_size), None
