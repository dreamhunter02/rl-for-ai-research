"""Tinker RL environment for the structured FinanceBench harness."""
from __future__ import annotations

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
from tinker_cookbook.tool_use import Tool, ToolResult, build_agent_tool_env, simple_tool_result, tool


def _as_list(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)] if value is not None else []


class CoercingTool:
    """Run a wrapped tool after coercing string list arguments to real lists."""

    def __init__(self, tool: Tool):
        self._tool = tool

    @property
    def name(self):
        return self._tool.name

    def to_spec(self):
        return self._tool.to_spec()

    async def run(self, call):
        arguments = dict(call.arguments) if hasattr(call, "arguments") else dict(call)
        coerced = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except Exception:
                    parsed = value
                if isinstance(parsed, list):
                    value = parsed
            coerced[key] = value
        call.__dict__.update(coerced)
        if hasattr(call, "model_copy"):
            call = call.model_copy(update=coerced)
        return await self._tool.run(call)

import sys
sys.path.insert(0, os.path.dirname(__file__))
import financebench_harness as hb

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

When you have enough evidence, call finish(answer, evidence_document, evidence_page) exactly once. Put the complete answer in the answer argument, including units and the requested conclusion. Do not stop after a search hit unless it actually resolves the question.
"""


OBS_CAP = 4000


def _tool_result(payload: object) -> ToolResult:
    """Bound every tool observation so one search cannot overflow context."""
    raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    if len(raw) <= OBS_CAP:
        return simple_tool_result(raw)
    envelope = {"truncated": True, "original_chars": len(raw), "content": raw[: OBS_CAP - 96]}
    return simple_tool_result(json.dumps(envelope, ensure_ascii=False))


class Bm25Tool:
    """Structured retrieval tools backed by one deterministic corpus index."""

    def __init__(self, index: hb.StructuredIndex):
        self.index = index

    @staticmethod
    async def build(max_docs: int | None = None) -> "Bm25Tool":
        return Bm25Tool(hb.build_index(max_docs=max_docs))

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
        if scope in ("prose", "both"):
            out["prose_hits"] = self.index.search_prose(query_list, filters, top_k)
        if scope in ("tables", "table", "both"):
            out["table_hits"] = self.index.search_tables(query_list, filters, top_k)
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
        return _tool_result({"grep_type_requested": grep_type, "backend_used": "page_text", "matches": result})

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
        hits = self.index.search_tables(query_list, filters, max(1, min(int(top_k), 8)))
        return _tool_result({"queries": query_list, "table_hits": hits})

    @tool
    async def read(
        self,
        document_id: Annotated[str, "Exact document_id."],
        page: Annotated[int, "One-based page number, or -1 when using passage_id."] = -1,
        start: Annotated[int, "Character start within the selected page/window."] = 0,
        end: Annotated[int, "Character end; bounded by the harness."] = hb.MAX_READ,
        passage_id: Annotated[str, "Optional passage_id returned by search."] = "",
    ) -> ToolResult:
        return _tool_result(self.index.read(document_id, page, start, end, passage_id))

    @tool
    async def read_table(
        self,
        table_id: Annotated[str, "Exact table_id returned by search_tables."],
        include_neighbors: Annotated[bool, "Include nearby page context for headers and footnotes."] = True,
    ) -> ToolResult:
        return _tool_result(self.index.read_table(table_id, include_neighbors))

    @tool
    async def calculate(
        self,
        expression: Annotated[str, "Basic arithmetic expression using numbers and + - * / % **."],
    ) -> ToolResult:
        try:
            value = hb.calculate(expression)
            return _tool_result({"expression": expression, "value": value})
        except Exception as exc:
            return _tool_result({"error": str(exc)})


    @tool
    async def finish(
        self,
        answer: Annotated[str, "Final answer to the user, with units and concise reasoning."],
        evidence_document: Annotated[str, "Primary evidence document identifier, if known."] = "",
        evidence_page: Annotated[int, "Primary evidence page, or -1 when unknown."] = -1,
    ) -> ToolResult:
        return _tool_result({"finish": True, "answer": answer, "evidence_document": evidence_document, "evidence_page": evidence_page})


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
    """Calibrated answer-quality reward; mechanics are logged separately."""

    def __init__(self, gold_answers: list[str], format_coef: float = 0.0):
        self.gold_answers = gold_answers
        self.format_coef = format_coef

    @staticmethod
    def _finish_answer(history: list[Message]) -> str:
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
                answer = str(args.get("answer", "")).strip() if isinstance(args, dict) else ""
                if answer:
                    return answer
        return ""

    @staticmethod
    def _history_trace(history: list[Message]) -> list[dict]:
        """Convert Tinker messages into the harness trace schema for grounding."""
        trace: list[dict] = []
        for message in history:
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
            content = _message_text(message) if isinstance(message, dict) else str(message)
            if role == "assistant":
                calls = []
                for call in (message.get("tool_calls") or []) if isinstance(message, dict) else []:
                    if isinstance(call, dict):
                        fn = call.get("function", call)
                        name = fn.get("name", "") if isinstance(fn, dict) else ""
                        raw = fn.get("arguments", {}) if isinstance(fn, dict) else {}
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
                trace.append({"role": "assistant", "content": content, "tool_calls": calls})
            elif role == "tool":
                call_id = message.get("tool_call_id", message.get("call_id", "")) if isinstance(message, dict) else getattr(message, "tool_call_id", "")
                trace.append({"role": "tool", "call_id": call_id, "content": content})
            else:
                trace.append({"role": role, "content": content})
        return trace

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        submitted = self._finish_answer(history)
        if submitted:
            text = submitted
            used_finish = 1.0
            formatted = 1.0
        else:
            final = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
            text = _message_text(final) if final is not None else ""
            used_finish = 0.0
            formatted = float(bool(re.search(r"^\s*Answer:\s*", text, re.I | re.M)))
        if not text.strip():
            return 0.0, {"format": formatted, "correct": 0.0, "quality": 0.0, "answer_quality": 0.0, "evidence_quality": 0.0, "grounded_quality": 0.0, "conclusion": 0.0, "details": 0.0, "used_finish": used_finish, "answer_nonempty": 0.0}
        scored = [hb.score_answer(gold, text) for gold in self.gold_answers]
        quality, parts = max(scored, key=lambda item: item[0], default=(0.0, {}))
        trace = self._history_trace(history)
        evidence, evidence_parts = max((hb.score_evidence(gold, text, trace) for gold in self.gold_answers), key=lambda item: item[0], default=(0.0, {}))
        grounded = hb.grounded_reward(quality, evidence)
        return grounded, {"format": formatted, "correct": quality, "quality": grounded, "answer_quality": quality, "evidence_quality": evidence, "grounded_quality": grounded, "conclusion": parts.get("conclusion", 0.0), "details": parts.get("details", 0.0), "strong_source": evidence_parts.get("strong_source", 0.0), "used_finish": used_finish, "answer_nonempty": 1.0}


def load_financebench(split_name: str = "train") -> list[dict]:
    """Load the frozen split, keeping held-out eval isolated from training."""
    split_path = hb.BASE / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        rows = split.get(split_name, [])
    else:
        rows = [json.loads(line) for line in (hb.DATA / "financebench_merged.jsonl").read_text().splitlines()]
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
    def __init__(self, datum, model_name, renderer_name, max_turns, group_size, tool_obj, format_coef=0.0, max_trajectory_tokens=32 * 1024):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.tool_obj = tool_obj
        self.format_coef = format_coef
        self.max_trajectory_tokens = max_trajectory_tokens

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)
        initial_messages = _initial_messages(self.datum, renderer, self.tool_obj)
        reward_fn = FinanceAnswerReward(gold_answers=self.datum["answer"], format_coef=self.format_coef)
        tools = [CoercingTool(t) for t in (self.tool_obj.bm25_search, self.tool_obj.grep_document, self.tool_obj.search_tables, self.tool_obj.read, self.tool_obj.read_table, self.tool_obj.calculate, self.tool_obj.finish)]
        return [build_agent_tool_env(renderer=renderer, tools=tools, initial_messages=initial_messages, reward_fn=reward_fn, model_name=self.model_name, max_turns=self.max_turns, max_trajectory_tokens=self.max_trajectory_tokens) for _ in range(self.group_size)]

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
        return len(self.builders) // self.batch_size


@chz.chz
class FinanceDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 6
    format_coef: float = 0.0
    max_trajectory_tokens: int = 32 * 1024
    seed: int = 0
    split_name: str = "train"

    async def __call__(self):
        tool_obj = await Bm25Tool.build()
        data = load_financebench(self.split_name)
        rng = random.Random(self.seed)
        rng.shuffle(data)
        builders = [FinanceSearchEnvGroupBuilder(d, self.model_name_for_tokenizer, self.renderer_name, self.max_turns, self.group_size, tool_obj, self.format_coef, self.max_trajectory_tokens) for d in data]
        return FinanceRLDataset(builders, self.batch_size), None
