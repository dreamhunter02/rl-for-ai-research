"""Repair citation-only teacher failures by replaying and rescoring real traces."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import financebench_harness as hb
from generate_teacher_traces import split_documents
from teacher_runtime import TeacherHarnessSession, submission_answer_text

STRONG_TOOLS = {"read", "read_table", "grep_document"}
CORRECT_UNGROUNDED = re.compile(r"F=1(?:\.0)? A=1(?:\.0)? G=0(?:\.0)?")


def canonical_support_reads(index: Any, target: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Build bounded reads that expose each frozen support quote exactly."""
    requests: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int]] = set()
    for support in target.get("support", []):
        document_id = support.get("document_id", "")
        page = support.get("page")
        quote = support.get("quote", "")
        if not document_id or not isinstance(page, int) or not quote or len(quote) > hb.MAX_READ:
            return None
        page_record = next((
            item for item in index.pages_by_doc.get(document_id, []) if item.get("page") == page
        ), None)
        if page_record is None:
            return None
        start = page_record.get("text", "").find(quote)
        if start < 0:
            return None
        key = (document_id, page, start, start + len(quote))
        if key not in seen:
            requests.append({"document_id": document_id, "page": page, "start": start, "end": start + len(quote)})
            seen.add(key)
    return requests or None


def insert_tool_turn_before_finish(
    trace: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    observations: list[str],
) -> list[dict[str, Any]]:
    repaired = copy.deepcopy(trace)
    position = next((
        i for i, item in enumerate(repaired)
        if item.get("role") == "assistant"
        and any(call.get("name") == "finish" for call in item.get("tool_calls", []))
    ), None)
    if position is None or len(calls) != len(observations):
        raise ValueError("Cannot insert support-read turn")
    inserted = [{"role": "assistant", "tool_calls": copy.deepcopy(calls)}]
    inserted.extend(
        {"role": "tool", "call_id": call["call_id"], "content": observation}
        for call, observation in zip(calls, observations)
    )
    repaired[position:position] = inserted
    return repaired


def execute_support_reads(
    session: Any,
    requests: list[dict[str, Any]],
    trace_sha256: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Execute one evidence turn, then advance so finish cites prior-turn receipts."""
    session.start_turn()
    calls = []
    observations = []
    for request_number, request in enumerate(requests, 1):
        call_id = f"repair_read_{trace_sha256[:12]}_{request_number}"
        result = session.execute("read", request, call_id)
        calls.append({"name": "read", "arguments": request, "call_id": call_id})
        observations.append(result.content)
    session.start_turn()
    return calls, observations


def _citation(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "receipt_id": receipt["receipt_id"],
        "document_id": receipt["document_id"],
        "page": receipt["page"],
    }


def repair_citations(
    target: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    calculations: dict[str, dict[str, Any]],
    submission: dict[str, Any],
) -> list[dict[str, Any]] | None:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    supports = target.get("support", [])
    claims = list(dict.fromkeys(s.get("claim", "answer") for s in supports))
    for claim in claims:
        match = next((
            receipt
            for receipt in receipts.values()
            if receipt.get("tool") in STRONG_TOOLS
            and any(
                support.get("claim", "answer") == claim
                and support.get("quote")
                and support.get("document_id") == receipt.get("document_id")
                and support.get("page") == receipt.get("page")
                and support["quote"] in receipt.get("text", "")
                for support in supports
            )
        ), None)
        if match is None:
            return None
        if match["receipt_id"] not in seen:
            selected.append(_citation(match))
            seen.add(match["receipt_id"])

    if target.get("derived"):
        calculation = calculations.get(submission.get("calc_id", ""))
        if not calculation:
            return None
        for operand in calculation.get("operands", {}).values():
            receipt = receipts.get(operand.get("receipt_id", ""))
            if not receipt or receipt.get("tool") not in STRONG_TOOLS:
                return None
            if receipt["receipt_id"] not in seen:
                selected.append(_citation(receipt))
                seen.add(receipt["receipt_id"])
    return selected


def _flatten_calls(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        call
        for item in trace
        if item.get("role") == "assistant"
        for call in item.get("tool_calls", [])
    ]


def _replay(index: hb.StructuredIndex, trace: list[dict[str, Any]]) -> tuple[TeacherHarnessSession, dict[str, Any], str] | None:
    session = TeacherHarnessSession(index, max_turns=64)
    finish_calls: list[tuple[dict[str, Any], str]] = []
    for item in trace:
        if item.get("role") != "assistant" or not item.get("tool_calls"):
            continue
        session.start_turn()
        for call in item["tool_calls"]:
            if call.get("parse_error"):
                return None
            name = call.get("name", "")
            arguments = call.get("arguments") or {}
            call_id = str(call.get("call_id", ""))
            if name == "finish":
                finish_calls.append((arguments, call_id))
            else:
                session.execute(name, arguments, call_id)
    if len(finish_calls) != 1:
        return None
    submission, call_id = finish_calls[0]
    return session, copy.deepcopy(submission), call_id


def _replace_finish(trace: list[dict[str, Any]], submission: dict[str, Any], call_id: str, observation: str) -> list[dict[str, Any]]:
    repaired = copy.deepcopy(trace)
    for item in repaired:
        if item.get("role") == "assistant":
            for call in item.get("tool_calls", []):
                if call.get("name") == "finish" and str(call.get("call_id", "")) == call_id:
                    call["arguments"] = copy.deepcopy(submission)
        elif item.get("role") == "tool" and str(item.get("call_id", "")) == call_id:
            item["content"] = observation
    return repaired


def _candidates(input_dir: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(input_dir.glob("*.failed.jsonl")):
        model = path.name.removesuffix(".failed.jsonl")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            trace = record.get("trace") or []
            if not trace or not CORRECT_UNGROUNDED.search(str(record.get("error", ""))):
                continue
            digest = hashlib.sha256(json.dumps(trace, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            record["teacher_model"] = model
            record["source_file"] = str(path)
            record["trace_sha256"] = digest
            candidates.append(record)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reject-out", required=True)
    parser.add_argument("--inject-support-reads", action="store_true")
    args = parser.parse_args()

    from reward_calculation import RewardConfig, build_judge

    targets = json.loads(Path(args.targets).read_text())
    index = hb.build_index(doc_names=split_documents(args.split_file))
    reward_config = RewardConfig.from_env()
    judge = build_judge(reward_config)
    output = Path(args.out)
    rejected_output = Path(args.reject_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    repaired_count = 0
    rejected_count = 0
    candidates = _candidates(Path(args.input_dir))

    with output.open("w") as accepted_handle, rejected_output.open("w") as rejected_handle:
        for position, record in enumerate(candidates, 1):
            fid = record["financebench_id"]
            working_trace = record["trace"]
            replayed = _replay(index, working_trace)
            reason = "trace_not_replayable"
            repair_method = "replay_exact_support_citation"
            if replayed is not None:
                session, submission, finish_call_id = replayed
                citations = repair_citations(targets[fid], session.state.receipts, session.state.calculations, submission)
                if citations is not None and args.inject_support_reads:
                    reason = "already_repairable_without_injection"
                    citations = None
                elif citations is None and args.inject_support_reads:
                    requests = canonical_support_reads(index, targets[fid])
                    if requests is None:
                        reason = "canonical_support_not_readable"
                    else:
                        calls, observations = execute_support_reads(
                            session, requests, record["trace_sha256"]
                        )
                        working_trace = insert_tool_turn_before_finish(working_trace, calls, observations)
                        citations = repair_citations(
                            targets[fid], session.state.receipts, session.state.calculations, submission
                        )
                        reason = "canonical_support_read_failed"
                        repair_method = "inject_canonical_support_reads_and_replay"
                elif citations is None:
                    reason = "support_not_present_in_delivered_receipts"
                if citations is not None:
                    submission["citations"] = citations
                    finish = session.execute("finish", submission, finish_call_id)
                    score = session.score(
                        targets[fid],
                        question=record["question"],
                        judge=judge,
                        judge_confidence_threshold=reward_config.judge_confidence_threshold,
                    )
                    if finish.should_stop and (score.get("F"), score.get("A"), score.get("G")) == (1, 1, 1):
                        repaired_trace = _replace_finish(
                            working_trace, session.accepted, finish_call_id, finish.content
                        )
                        accepted = {
                            "financebench_id": fid,
                            "question": record["question"],
                            "gold": record.get("gold", ""),
                            "teacher_model": record["teacher_model"],
                            "tool_calls": _flatten_calls(repaired_trace),
                            "trace": repaired_trace,
                            "answer_text": submission_answer_text(session.accepted),
                            "submission": session.accepted,
                            "score": score,
                            "termination_reason": "finish",
                            "finish_ok": 1.0,
                            "repair": {
                                "method": repair_method,
                                "source_file": record["source_file"],
                                "source_trace_sha256": record["trace_sha256"],
                            },
                        }
                        accepted_handle.write(json.dumps(accepted, ensure_ascii=False) + "\n")
                        accepted_handle.flush()
                        repaired_count += 1
                        print(f"[{position}/{len(candidates)}] {fid} repaired", flush=True)
                        continue
                    reason = f"rescored_F{score.get('F', 0)}A{score.get('A', 0)}G{score.get('G', 0)}"
            rejected_handle.write(json.dumps({
                "financebench_id": fid,
                "teacher_model": record["teacher_model"],
                "source_trace_sha256": record["trace_sha256"],
                "reason": reason,
            }, ensure_ascii=False) + "\n")
            rejected_handle.flush()
            rejected_count += 1
    print(json.dumps({
        "candidates": len(candidates),
        "repaired": repaired_count,
        "rejected": rejected_count,
        "out": str(output),
        "reject_out": str(rejected_output),
    }), flush=True)


if __name__ == "__main__":
    main()
