"""Build transparent, source-backed SFT demonstrations for uncovered train questions."""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import financebench_harness as hb
from generate_teacher_traces import split_documents
from prepare_sft_dataset import eligible_record
from teacher_runtime import TeacherHarnessSession, submission_answer_text


NUMBER = re.compile(r"(?<![\w.])(?:\(\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*\)|[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!,))")


def _support_window(index: Any, support: dict[str, Any], max_read: int) -> dict[str, Any] | None:
    document_id = support.get("document_id", "")
    page = support.get("page")
    quote = support.get("quote", "")
    if not document_id or not isinstance(page, int) or not quote or len(quote) > max_read:
        return None
    page_record = next((
        item for item in index.pages_by_doc.get(document_id, []) if item.get("page") == page
    ), None)
    if page_record is None:
        return None
    start = page_record.get("text", "").find(quote)
    if start < 0:
        return None
    return {
        "document_id": document_id,
        "page": page,
        "start": start,
        "end": start + len(quote),
        "quote": quote,
        "claim": support.get("claim", "answer"),
    }


def choose_claim_supports(index: Any, target: dict[str, Any], max_read: int = hb.MAX_READ) -> list[dict[str, Any]] | None:
    selected = []
    claims = list(dict.fromkeys(s.get("claim", "answer") for s in target.get("support", [])))
    for claim in claims:
        candidates = [
            window
            for support in target.get("support", [])
            if support.get("claim", "answer") == claim
            for window in [_support_window(index, support, max_read)]
            if window is not None
        ]
        if not candidates:
            return None
        selected.append(min(candidates, key=lambda item: len(item["quote"])))
    return selected or None


def _contains_value(quote: str, value: str) -> bool:
    try:
        wanted = Decimal(str(value))
    except InvalidOperation:
        return False
    for token in NUMBER.findall(quote.replace("−", "-").replace("–", "-")):
        cleaned = token.replace(",", "").strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1].strip().lstrip("+")
        try:
            if Decimal(cleaned) == wanted:
                return True
        except InvalidOperation:
            pass
    return False


def choose_operand_support(index: Any, operand: dict[str, Any], max_read: int = hb.MAX_READ) -> dict[str, Any] | None:
    candidates = []
    for support in operand.get("support", []):
        window = _support_window(index, support, max_read)
        if (
            window is not None
            and str(operand.get("metric", "")).lower() in window["quote"].lower()
            and str(operand.get("period", "")) in window["quote"]
            and _contains_value(window["quote"], str(operand.get("value", "")))
        ):
            candidates.append(window)
    return min(candidates, key=lambda item: len(item["quote"])) if candidates else None


def _trace_turn(session: TeacherHarnessSession, trace: list[dict[str, Any]], calls: list[dict[str, Any]]) -> list[str]:
    session.start_turn()
    trace.append({"role": "assistant", "tool_calls": calls})
    observations = []
    for call in calls:
        result = session.execute(call["name"], call["arguments"], call["call_id"])
        observations.append(result.content)
        trace.append({"role": "tool", "call_id": call["call_id"], "content": result.content})
    return observations


def _finish_submission(target: dict[str, Any], citations: list[dict[str, Any]], calc_id: str = "") -> dict[str, Any]:
    answer_type = target["answer_type"]
    submission: dict[str, Any] = {"answer_type": answer_type, "citations": citations}
    if answer_type == "numeric":
        submission.update(value=str(target["value"]), unit=target["unit"], scale=target["scale"])
        if calc_id:
            submission["calc_id"] = calc_id
    elif answer_type == "boolean":
        submission["decision"] = target["decision"]
    else:
        aliases = target.get("aliases") or []
        if not aliases:
            raise ValueError("text target has no exact reviewed alias")
        submission["answer_text"] = aliases[0]
    return submission


def build_demonstration(index: Any, fid: str, target: dict[str, Any]) -> dict[str, Any]:
    if target.get("adjudication_status", "resolved") != "resolved":
        raise ValueError("target is unresolved")
    session = TeacherHarnessSession(index, max_turns=16)
    trace: list[dict[str, Any]] = []
    source_support = target.get("support", [])
    if not source_support:
        raise ValueError("target has no support")
    document_id = source_support[0]["document_id"]
    search_call = {
        "name": "bm25_search",
        "arguments": {"query_list": [target["question"]], "document_id": document_id, "top_k": 3},
        "call_id": f"gold_{fid}_search",
    }
    _trace_turn(session, trace, [search_call])

    operand_windows: dict[str, dict[str, Any]] = {}
    if target.get("derived"):
        for name, operand in target.get("operands", {}).items():
            window = choose_operand_support(index, operand)
            if window is None:
                raise ValueError(f"operand support is not readable: {name}")
            operand_windows[name] = window
        windows = list(operand_windows.values())
    else:
        windows = choose_claim_supports(index, target)
        if windows is None:
            raise ValueError("claim support is not readable")

    read_calls = [{
        "name": "read",
        "arguments": {key: window[key] for key in ("document_id", "page", "start", "end")},
        "call_id": f"gold_{fid}_read_{position}",
    } for position, window in enumerate(windows, 1)]
    read_observations = _trace_turn(session, trace, read_calls)
    receipt_ids = [json.loads(observation)["receipt_id"] for observation in read_observations]
    citations = [
        {"receipt_id": receipt_id, "document_id": window["document_id"], "page": window["page"]}
        for receipt_id, window in zip(receipt_ids, windows)
    ]

    calc_id = ""
    if target.get("derived"):
        operands = {}
        for (name, operand), receipt_id in zip(target["operands"].items(), receipt_ids):
            operands[name] = {
                key: operand[key] for key in ("value", "unit", "scale", "metric", "period")
            }
            operands[name].update(receipt_id=receipt_id, quote=operand_windows[name]["quote"])
        calculate_call = {
            "name": "calculate",
            "arguments": {"expression": target["expression"], "operands": operands},
            "call_id": f"gold_{fid}_calculate",
        }
        calculation = json.loads(_trace_turn(session, trace, [calculate_call])[0])
        if not calculation.get("calc_id"):
            raise ValueError(f"calculation failed: {calculation.get('error', 'unknown error')}")
        calc_id = calculation["calc_id"]

    submission = _finish_submission(target, citations, calc_id)
    finish_call = {"name": "finish", "arguments": submission, "call_id": f"gold_{fid}_finish"}
    finish_observation = _trace_turn(session, trace, [finish_call])[0]
    score = session.score(target, question=target["question"])
    if (score.get("F"), score.get("A"), score.get("G")) != (1.0, 1.0, 1.0):
        raise ValueError(f"demonstration did not pass full reward: {score}")
    return {
        "financebench_id": fid,
        "question": target["question"],
        "gold": target.get("original_answer", ""),
        "teacher_model": "gold_recovery_v1",
        "tool_calls": [call for item in trace if item.get("role") == "assistant" for call in item.get("tool_calls", [])],
        "trace": trace,
        "answer_text": submission_answer_text(session.accepted),
        "submission": session.accepted,
        "score": score,
        "termination_reason": "finish",
        "finish_ok": 1.0,
        "repair": {
            "method": "target_conditioned_source_verified_demonstration",
            "disclosure": "Synthetic supervised demonstration built from reviewed train target and indexed source; not a teacher-model rollout.",
            "finish_observation": finish_observation,
        },
    }


def _covered_ids(paths: list[str]) -> set[str]:
    covered = set()
    for name in paths:
        for line in Path(name).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if eligible_record(record):
                covered.add(record["financebench_id"])
    return covered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--existing", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--reject-out", required=True)
    args = parser.parse_args()

    targets = json.loads(Path(args.targets).read_text())
    train_rows = json.loads(Path(args.split_file).read_text())["train"]
    train_ids = [row if isinstance(row, str) else row["financebench_id"] for row in train_rows]
    covered = _covered_ids(args.existing)
    index = hb.build_index(doc_names=split_documents(args.split_file))
    output = Path(args.out)
    rejected_output = Path(args.reject_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    built = 0
    rejected = 0
    with output.open("w") as accepted_handle, rejected_output.open("w") as rejected_handle:
        for fid in train_ids:
            if fid in covered:
                continue
            try:
                record = build_demonstration(index, fid, targets[fid])
            except Exception as exc:
                rejected_handle.write(json.dumps({"financebench_id": fid, "reason": str(exc)}) + "\n")
                rejected += 1
                continue
            accepted_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            accepted_handle.flush()
            built += 1
            print(f"{fid} built", flush=True)
    print(json.dumps({"covered_before": len(covered), "built": built, "rejected": rejected}), flush=True)


if __name__ == "__main__":
    main()
