"""Convert fully verified current-harness teacher traces into chat SFT rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def identified_teacher(record: dict[str, Any]) -> bool:
    """Reject missing identities and explicitly synthetic gold-recovery records.

    This is an identity filter, not independent proof of model provenance.
    Original records and their migration lineage must remain available for audit.
    """
    model = str(record.get('teacher_model') or '').strip().lower()
    return bool(model and model != 'unknown' and not model.startswith('gold_recovery'))


def eligible_record(record: dict[str, Any]) -> bool:
    score = record.get("score") or {}
    calls = record.get("tool_calls") or []
    finish_calls = [call for call in calls if call.get("name") == "finish"]
    if any(c.get('name')=='calculate' for c in calls): return False
    for event in record.get('trace',[]):
        if not isinstance(event,dict) or event.get('role')!='tool': continue
        try: payload=json.loads(event.get('content',''))
        except (ValueError,TypeError): return False
        if isinstance(payload,dict) and payload.get('error'): return False
    return bool(
        record.get("termination_reason") == "finish"
        and score.get("F") == 1
        and score.get("A") == 1
        and score.get("G") == 1
        and not score.get("unresolved")
        and calls
        and len(finish_calls) == 1
        and calls[-1].get("name") == "finish"
        and not any(call.get("parse_error") for call in calls)
    )


def build_messages(record: dict[str, Any], system_prompt: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": record["question"]},
    ]
    for item in record.get("trace", []):
        role = item.get("role")
        if role == "assistant":
            calls = []
            for position, call in enumerate(item.get("tool_calls") or []):
                call_id = str(call.get("call_id") or f"call-{len(messages)}-{position}")
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(call["name"]),
                        "arguments": call.get("arguments") or {},
                    },
                })
            message: dict[str, Any] = {"role": "assistant", "content": item.get("content") or None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
        elif role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": str(item.get("call_id", "")),
                "content": str(item.get("content", "")),
            })
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--require-identified-teacher", action="store_true",
                        help="Exclude unknown identities and synthetic gold-recovery traces")
    args = parser.parse_args()

    from finance_env import Bm25Tool, FINANCE_TASK_INSTRUCTIONS

    tool_object = Bm25Tool(None)
    specs = [
        getattr(tool_object, name).to_spec()
        for name in ("bm25_search", "grep_document", "search_tables", "read", "read_table", "finish")
    ]
    tools = [{"type": "function", "function": spec} for spec in specs]
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    selected = [row for row in rows if eligible_record(row)
                and (not args.require_identified_teacher or identified_teacher(row))]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps({
        "financebench_id": row["financebench_id"],
        "messages": build_messages(row, FINANCE_TASK_INSTRUCTIONS),
        "tools": tools,
        "teacher_model": row.get("teacher_model", ""),
        "score": row["score"],
    }, ensure_ascii=False) + "\n" for row in selected))
    summary = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "excluded_rows": len(rows) - len(selected),
        "require_identified_teacher": args.require_identified_teacher,
        "unique_questions": len({row["financebench_id"] for row in selected}),
        "out": str(output),
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
