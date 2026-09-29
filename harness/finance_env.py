"""Tinker RL environment for the structured FinanceBench harness."""
from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Sequence
from typing import Annotated, Any
import math

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, InitialObservationOverflow
from tinker_cookbook.tool_use.types import ToolInput
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
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
        raw_arguments = call.arguments if hasattr(call, "arguments") else call
        if not isinstance(raw_arguments, dict):
            # Let the SDK's validation wrapper return a recoverable tool error.
            return await self._tool.run(call)
        arguments = dict(raw_arguments)
        schema = self.to_spec().get("parameters", {}).get("properties", {})
        for key, value in list(arguments.items()):
            if schema.get(key, {}).get("type") != "array" or not isinstance(value, str):
                continue
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = value
            arguments[key] = parsed if isinstance(parsed, list) else [value]
        for key, value in list(arguments.items()):
            kind = schema.get(key, {}).get("type")
            if kind == "object" and isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    arguments[key] = parsed
            elif kind == "string" and value is not None and not isinstance(value, str):
                arguments[key] = str(value)
        return await self._tool.run(ToolInput(arguments=arguments, call_id=getattr(call, "call_id", None)))

import sys
sys.path.insert(0, os.path.dirname(__file__))
import financebench_harness as hb
from observations import bounded_observation, compact_hits
from workshop_reward import EpisodeState, validate_submission, validate_target, score_submission
from reward_calculation import RewardConfig, answer_quality_with_judge, build_judge, reward_formula

FINANCE_TASK_INSTRUCTIONS = """You are a financial-filings retrieval agent.

Your job is to answer the user's question using the SEC filing corpus. Search
before answering. Start with bm25_search to identify the likely document and
passages, then use grep_document for exact accounting terms or regexes and
read/read_table to inspect bounded evidence. Any derived numeric answer must come from calculate using named, source-backed operands. Direct extraction needs no calculation.
You may switch documents or issue refined searches when the evidence is not
sufficient. Do not invent values or rely on outside knowledge.

Available tools:
- bm25_search: ranked keyword search with optional company/year/document filters
- grep_document: document-scoped exact or regex search; choose grep_type text, pdfgrep, or rga
- search_tables: ranked search over table-like pages in one or all filings
- read: bounded page/passage retrieval with provenance
- read_table: retrieve a table candidate with neighboring headers and footnotes
- calculate: safe arithmetic for ratios, changes, and unit conversions

When you have enough evidence, call finish exactly once. For numeric answers use
answer_type="numeric", value as a decimal string, unit and scale; leave answer_text
empty. For yes/no-only answers use answer_type="boolean", decision="yes" or "no".
For text or multipart answers use answer_type="text", answer_text with all requested
facts. Cite receipt_id, document_id and one-based page from read/grep results.
For arithmetic, calculate accepts an expression with named variables and an operands
object mapping each variable to value, unit, scale, metric, period, receipt_id and an
exact quote. Include its calc_id in finish. Do not use constant-only calculations as
support. Read beyond snippets when needed. Use the final available turn to finish;
never continue searching when only one turn remains.
"""


OBS_CAP = 4000


def _tool_result(payload: object) -> ToolResult:
    return simple_tool_result(bounded_observation(payload))


class Bm25Tool:
    """Structured retrieval tools backed by one deterministic corpus index."""

    def __init__(self, index: hb.StructuredIndex, max_turns: int = 8):
        self.index = index
        self.max_turns = max_turns
        self.state = EpisodeState()

    def _result(self, payload, name="", should_stop=False):
        payload = dict(payload)
        remaining = max(0, self.max_turns - self.state.turn)
        payload["turns_remaining"] = remaining
        if remaining == 1: payload["next_action"] = "Next turn must be finish."
        if name in ("read", "read_table") and "text" in payload:
            payload["receipt_id"] = f"r{len(self.state.receipts)+1}"
        if name == "grep_document":
            for i, hit in enumerate(payload.get("matches", []), 1):
                hit["receipt_id"] = f"r{len(self.state.receipts)+i}"
                hit["text"] = hit.get("text", "")[:500]
        raw = bounded_observation(payload)
        delivered = json.loads(raw)
        if name in ("read", "read_table") and delivered.get("receipt_id"):
            self.state.record(name, delivered)
        if name == "grep_document":
            for hit in delivered.get("matches", []): self.state.record(name, hit)
        self.state.observations.append({"name": name, "payload": delivered, "visible_chars": len(raw)})
        return simple_tool_result(raw, should_stop=should_stop)

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
        top_k = max(1, min(int(top_k), 5))
        scope = (scope or "both").lower()
        out: dict[str, object] = {"queries": query_list, "filters": filters}
        if scope in ("prose", "both"):
            out["prose_hits"] = self.index.search_prose(query_list, filters, top_k)
        if scope in ("tables", "table", "both"):
            out["table_hits"] = self.index.search_tables(query_list, filters, top_k)
        combined = [(k, h) for k in ("prose_hits", "table_hits") for h in out.get(k, [])]
        combined.sort(key=lambda x: x[1].get("score", 0), reverse=True)
        for k in ("prose_hits", "table_hits"):
            if k in out: out[k] = compact_hits([h for key, h in combined[:top_k] if key == k], query_list)
        return self._result(out, "bm25_search")

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
        return self._result({"grep_type_requested": grep_type, "backend_used": "page_text", "matches": result}, "grep_document")

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
        return self._result({"queries": query_list, "table_hits": compact_hits(hits, query_list)}, "search_tables")

    @tool
    async def read(
        self,
        document_id: Annotated[str, "Exact document_id."],
        page: Annotated[int, "One-based page number, or -1 when using passage_id."] = -1,
        start: Annotated[int, "Character start within the selected page/window."] = 0,
        end: Annotated[int, "Absolute character end, or -1 for the next bounded window."] = -1,
        passage_id: Annotated[str, "Optional passage_id returned by search."] = "",
    ) -> ToolResult:
        return self._result(self.index.read(document_id, page, start, None if end < 0 else end, passage_id), "read")

    @tool
    async def read_table(
        self,
        table_id: Annotated[str, "Exact table_id returned by search_tables."],
        include_neighbors: Annotated[bool, "Return neighbor page identifiers for headers and footnotes."] = True,
        start: Annotated[int, "Character offset within table page."] = 0,
        end: Annotated[int, "Absolute end, or -1 for the next bounded table window."] = -1,
    ) -> ToolResult:
        return self._result(self.index.read_table(table_id, include_neighbors, start, None if end < 0 else end), "read_table")

    @tool
    async def calculate(
        self,
        expression: Annotated[str, "Arithmetic using named source-backed operands and + - * /."],
        operands: Annotated[dict[str, Any], "Variable to value/unit/scale/metric/period/receipt_id/quote mapping."] = {},
    ) -> ToolResult:
        try:
            return self._result(self.state.calculate(expression, operands), "calculate")
        except Exception as exc:
            return self._result({"error": str(exc)}, "calculate")

    @tool
    async def finish(
        self,
        answer_type: Annotated[str, "numeric, boolean, or text"],
        value: Annotated[str, "Single decimal numeric answer; no units here."] = "",
        unit: Annotated[str, "USD, EUR, percent, ratio, shares, or number."] = "",
        scale: Annotated[str, "ones, thousand, million, or billion."] = "",
        decision: Annotated[str, "yes or no for boolean answers."] = "",
        answer_text: Annotated[str, "Text/multipart answer; empty for numeric and boolean."] = "",
        citations: Annotated[list[dict[str, Any]], "Each contains receipt_id, document_id and one-based page."] = [],
        calc_id: Annotated[str, "Calculation record for a derived numeric result."] = "",
    ) -> ToolResult:
        if self.state.accepted is not None or self.state.terminal_ambiguous:
            self.state.accepted = None
            self.state.terminal_ambiguous = True
            self.state.validation_errors += 1
            return self._result({"error": "Multiple terminal submissions are ambiguous"}, "finish", True)
        try:
            self.state.accepted = validate_submission(dict(answer_type=answer_type, value=value, unit=unit,
                scale=scale, decision=decision, answer_text=answer_text, citations=citations, calc_id=calc_id))
        except ValueError as exc:
            self.state.validation_errors += 1
            return self._result({"error": str(exc)}, "finish")
        return self._result({"finish": True, "accepted": self.state.accepted}, "finish", True)


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
        episode_state=None,
        target=None,
    ):
        self.episode_state = episode_state
        self.target = target
        self.gold_answers = gold_answers
        self.question = question
        self.format_coef = format_coef
        self.judge = judge
        self.reward_config = reward_config or RewardConfig(require_finish=require_finish)
        self.require_finish = require_finish

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
        }

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, Any]]:
        if self.target is not None:
            state = self.episode_state
            result = score_submission(self.target, state.accepted, state, float(os.environ.get("RETRIEVAL_WEIGHT", "0")))
            if result["unresolved"] and self.judge is not None and state.accepted:
                try:
                    verdict = await self.judge.judge(question=self.question,
                        gold=json.dumps(self.target.get("required_facts") or self.target.get("aliases", [])),
                        candidate=state.accepted.get("answer_text", ""),
                        evidence="\n".join(r.get("text", "") for r in state.receipts.values()))
                    confidence = float(verdict.get("confidence", 0))
                    if confidence >= self.reward_config.judge_confidence_threshold and (verdict.get("verdict") == "contradicted" or (verdict.get("verdict") == "entailed" and verdict.get("numeric_ok") is True)):
                        result["A"] = float(verdict["verdict"] == "entailed")
                        result["unresolved"] = False
                        result["correct"] = int(result["A"] == 1)
                        result["grounded_success"] = int(result["correct"] and result["G"] == 1)
                        weight = float(os.environ.get("RETRIEVAL_WEIGHT", "0"))
                        result["reward"] = 0.0 if result["fabricated_citation"] else (1-weight)*result["A"]*(0.5+0.5*result["G"])+weight*result["Ret"]
                    result["judge_verdict"] = verdict.get("verdict", "ambiguous")
                except Exception as exc:
                    result["judge_error"] = type(exc).__name__
            state.last_score = result
            return result["reward"], {k: float(v) for k, v in result.items() if isinstance(v, (int, float, bool))}
        raise ValueError("Reviewed target and per-episode state are required; use historical scorers only for diagnostics")

def load_financebench(split_name: str = "train") -> list[dict]:
    """Load the frozen split, keeping held-out eval isolated from training."""
    split_path = __import__("pathlib").Path(os.environ.get("FINANCEBENCH_SPLIT", str(hb.BASE / "split.json")))
    if split_path.exists():
        split = json.loads(split_path.read_text())
        split_name = "train" if split_name == "train96" else split_name
        if split_name not in split:
            raise ValueError(f"Split {split_name!r} missing from {split_path}")
        rows = split[split_name]
    else:
        raise FileNotFoundError(f"Frozen split required: {split_path}")
    target_path = os.environ.get("FINANCEBENCH_TARGETS")
    targets = json.loads(__import__("pathlib").Path(target_path).read_text()) if target_path else {}
    out = []
    for row in rows:
        answer = str(row.get("answer") or "").strip()
        if not answer:
            continue
        qid = row.get("financebench_id", "")
        if target_path:
            if qid not in targets: raise ValueError(f"Missing reviewed target for {qid}")
            validate_target(targets[qid])
        out.append({
            "question": _format_question(row),
            "answer": [answer],
            "doc": row.get("doc_name", ""),
            "company": row.get("company", ""),
            "financebench_id": row.get("financebench_id", ""),
            "evidence": row.get("evidence", []),
            "target": targets.get(qid),
            "split": split_name,
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
    def __init__(self, datum, model_name, renderer_name, max_turns, group_size, tool_obj, format_coef=0.0, max_trajectory_tokens=32 * 1024, judge=None, reward_config: RewardConfig | None = None):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.tool_obj = tool_obj
        self.format_coef = format_coef
        self.max_trajectory_tokens = max_trajectory_tokens
        self.judge = judge
        self.reward_config = reward_config or RewardConfig.from_env()

    async def make_envs(self) -> Sequence[Env]:
        validate_target(self.datum.get("target") or {})
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)
        envs = []
        for _ in range(self.group_size):
            tool_obj = Bm25Tool(self.tool_obj.index, self.max_turns)
            initial_messages = _initial_messages(self.datum, renderer, tool_obj)
            reward_fn = FinanceAnswerReward(gold_answers=self.datum["answer"], question=self.datum.get("question", ""),
                judge=self.judge, reward_config=self.reward_config, episode_state=tool_obj.state, target=self.datum.get("target"))
            tools = [CoercingTool(getattr(tool_obj, name)) for name in ("bm25_search", "grep_document", "search_tables", "read", "read_table", "calculate", "finish")]
            env = build_agent_tool_env(renderer=renderer, tools=tools, initial_messages=initial_messages,
                reward_fn=reward_fn, model_name=self.model_name, max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens, failed_parse_reward=0.0, context_overflow_reward=0.0,
                parse_error_policy=ParseErrorPolicy(max_consecutive=1, mask_error_turns=True, terminal_reward=0.0))
            env.message_env.tool_execution = "sequential"
            env.message_env.example_id = self.datum.get("financebench_id", "")
            env.example_id = self.datum.get("financebench_id", "")
            envs.append(FinanceEpisodeEnv(env, tool_obj))
        return envs

    async def compute_group_rewards(self, trajectory_group, env_group):
        unresolved = any(env.tool_obj.state.last_score.get("unresolved") for env in env_group)
        totals = [sum(t.reward for t in trajectory.transitions) for trajectory in trajectory_group]
        audit_path = os.environ.get("WORKSHOP_GROUP_AUDIT")
        if audit_path:
            from pathlib import Path
            import uuid
            import statistics
            group_id = uuid.uuid4().hex
            path = Path(audit_path)
            trace_path = path.parent / "workshop_rollouts" / f"{group_id}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            def encode(obj):
                if hasattr(obj, "model_dump"): return obj.model_dump()
                if hasattr(obj, "__dict__"): return vars(obj)
                raise TypeError(type(obj).__name__)
            trace_path.write_text(json.dumps({"question_id": self.datum.get("financebench_id"),
                "episodes": [{"history": env.env.message_env.history, "state": vars(env.tool_obj.state),
                              "stop_reason": trajectory.stop_reason,
                              "total_reward_before_group_exclusion": total}
                             for env, trajectory, total in zip(env_group, trajectory_group, totals)]}, default=encode))
            record = {"group_id": group_id, "split": self.datum.get("split", "unknown"), "question_id": self.datum.get("financebench_id"),
                "sampled_trajectories": len(totals), "rewards": totals, "unresolved_group": unresolved,
                "all_zero": all(r == 0 for r in totals), "all_equal": len(set(totals)) <= 1,
                "reward_variance": statistics.pvariance(totals) if totals else 0,
                "retained": not unresolved and len(set(totals)) > 1, "trace_path": str(trace_path)}
            with path.open("a") as f: f.write(json.dumps(record)+"\n")
        if not unresolved: return [(0.0, {"unresolved_group": 0.0}) for _ in trajectory_group]
        out = []
        for trajectory in trajectory_group:
            for transition in trajectory.transitions:
                transition.metrics["parse_error_masked"] = 1.0
            out.append((-sum(t.reward for t in trajectory.transitions), {"unresolved_group": 1.0}))
        return out

    def logging_tags(self) -> list[str]:
        return ["financebench", "structured_sparse_agent"]


class FinanceEpisodeEnv(Env):
    """Count actual assistant turns, including parse retries, outside individual tools."""
    def __init__(self, env, tool_obj):
        self.env, self.tool_obj = env, tool_obj
        self.example_id = getattr(env, "example_id", None)
        self.rollout_limits = getattr(env, "rollout_limits", None)

    async def initial_observation(self):
        result = await self.env.initial_observation()
        if isinstance(result, InitialObservationOverflow):
            self._terminal_metrics(result.metrics)
        return result

    @staticmethod
    def _terminal_metrics(metrics):
        for key in ("F", "A", "G", "Ret", "correct", "grounded_success", "unresolved"):
            metrics.setdefault(key, 0.0)
        metrics.setdefault("used_finish", metrics["F"])
        metrics.setdefault("finish_gate", metrics["F"])
        metrics.setdefault("finish_missing", 1.0 - metrics["F"])
        metrics.setdefault("answer_quality", metrics["A"])
        metrics.setdefault("evidence_quality", metrics["G"])

    async def step(self, action, *, extra=None):
        self.tool_obj.state.turn += 1
        result = await self.env.step(action, extra=extra)
        if result.episode_done:
            # Budget/parse exits can bypass reward_fn; keep evaluator denominators whole.
            self._terminal_metrics(result.metrics)
        return result


class FinanceRLDataset(RLDataset):
    def __init__(self, builders, batch_size, epochs=1, seed=0):
        if batch_size < 1 or epochs < 1 or not builders: raise ValueError("Nonempty data and positive batch/epochs required")
        self.builders, self.batch_size, self.epochs, self.seed = builders, batch_size, epochs, seed
        self.per_epoch = math.ceil(len(builders)/batch_size)

    def get_batch(self, index):
        if not 0 <= index < len(self): raise IndexError(index)
        epoch, batch = divmod(index, self.per_epoch)
        order = list(self.builders)
        random.Random(self.seed + epoch).shuffle(order)
        start = batch * self.batch_size
        return order[start:start + self.batch_size]

    def __len__(self):
        return self.epochs * self.per_epoch


@chz.chz
class FinanceDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 8
    format_coef: float = 0.0
    max_trajectory_tokens: int = 32 * 1024
    seed: int = 0
    split_name: str = "train"
    epochs: int = 1

    async def __call__(self):
        reward_config = RewardConfig.from_env()
        judge = build_judge(reward_config)
        data = load_financebench(self.split_name)
        tool_obj = await Bm25Tool.build(doc_names=sorted({d['doc'] for d in data}))
        rng = random.Random(self.seed)
        rng.shuffle(data)
        builders = [FinanceSearchEnvGroupBuilder(d, self.model_name_for_tokenizer, self.renderer_name, self.max_turns, self.group_size, tool_obj, self.format_coef, self.max_trajectory_tokens, judge=judge, reward_config=reward_config) for d in data]
        return FinanceRLDataset(builders, self.batch_size, self.epochs, self.seed), None
