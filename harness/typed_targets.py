"""Evaluator-only typed FinanceBench targets.

Targets are generated offline and are never inserted into student prompts.  The
initial generator is conservative: it preserves the source answer/evidence and
marks inferred fields so a human or source-verification pass can correct them
before a final comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import financebench_harness as hb


def _answer_type(answer: str) -> str:
    text = str(answer or "").strip()
    if hb._decision(text) is not None:
        return "decision"
    mentions = [item for item in hb._number_mentions(text) if not hb._is_year(item[0])]
    if mentions:
        if any(item[1] for item in mentions):
            return "percentage"
        if "$" in text or any(word in text.lower() for word in ("dollar", "usd", "million", "billion", "thousand")):
            return "currency"
        return "scalar"
    return "text"


def _primary_value(answer: str, answer_type: str) -> Any:
    if answer_type == "decision":
        return hb._decision(answer)
    if answer_type in {"scalar", "percentage", "currency"}:
        mentions = [item for item in hb._number_mentions(answer) if not hb._is_year(item[0])]
        if mentions:
            return mentions[0][0]
    return None


def infer_target(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer") or "").strip()
    answer_type = _answer_type(answer)
    spans = []
    for item in row.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        spans.append({
            "document_id": item.get("doc_name") or row.get("doc_name", ""),
            "page": item.get("evidence_page_num"),
            "text": str(item.get("evidence_text") or item.get("evidence_text_full_page") or "")[:2000],
        })
    return {
        "schema_version": "financebench-target-v1",
        "financebench_id": row.get("financebench_id", ""),
        "question": str(row.get("question") or ""),
        "answer_type": answer_type,
        "answer_text": answer,
        "value": _primary_value(answer, answer_type),
        "unit": "" if answer_type not in {"currency", "percentage"} else ("percent" if answer_type == "percentage" else "currency"),
        "scale": "",
        "decision": hb._decision(answer),
        "tolerance": {"absolute": 0.01, "relative": 0.005} if answer_type in {"scalar", "percentage", "currency"} else None,
        "required_facts": sorted(hb._answer_tokens(answer)),
        "supporting_spans": spans,
        "source_verified": False,
    }


def write_targets(split_name: str, output: Path) -> int:
    split_path = hb.BASE / "split.json"
    split = json.loads(split_path.read_text())
    rows = split.get(split_name, [])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(infer_target(row), ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps({"split": args.split, "rows": write_targets(args.split, Path(args.output))}))


if __name__ == "__main__":
    main()
