"""Deterministic E1 capability gates for the repaired FinanceBench protocol."""
from __future__ import annotations
import asyncio, json
from pathlib import Path
from finance_env import FinanceAnswerReward

QUESTION = "What was the capital expenditure in USD millions?"
GOLD = "$1577.00"

def history(document: str = "TEST_2018_10K", page: int = 12, with_read: bool = True, finish: bool = True, citation_document: str | None = None, citation_page: int | None = None):
    h = [{"role": "assistant", "content": "", "tool_calls": [{"name": "bm25_search", "arguments": {"query_list": ["capital expenditure"]}, "call_id": "s1"}]}, {"role": "tool", "name": "bm25_search", "call_id": "s1", "content": json.dumps({"results": [{"document_id": document, "page": page, "snippet": "capital expenditure"}]})}]
    if with_read:
        h += [{"role": "assistant", "content": "", "tool_calls": [{"name": "read", "arguments": {"document_id": document, "page": page}, "call_id": "r1"}]}, {"role": "tool", "name": "read", "call_id": "r1", "content": json.dumps({"document_id": document, "page": page, "text": "Capital expenditures were $1,577 million."})}]
    if finish:
        cited_doc = citation_document or document
        cited_page = page if citation_page is None else citation_page
        h += [{"role": "assistant", "content": "", "tool_calls": [{"name": "finish", "arguments": {"answer_text": "Capital expenditures were $1,577 million.", "answer_type": "currency", "value": "1577", "unit": "USD", "scale": "million", "evidence_document": cited_doc, "evidence_page": cited_page, "citations": [f"{cited_doc}:page:{cited_page}"], "calc_id": ""}, "call_id": "f1"}]}]
    return h

async def run():
    cases = {}
    reward = FinanceAnswerReward([GOLD], question=QUESTION)
    value, metrics = await reward(history())
    cases["terminal_finish_oracle_evidence"] = {"pass": value > 0.8 and metrics["finish_gate"] == 1 and metrics["citation_valid"] == 1, "reward": value, "metrics": metrics}
    value, metrics = await reward(history(citation_document="NOT_SEEN", citation_page=99))
    cases["nonexistent_citation_veto"] = {"pass": value == 0 and metrics["citation_veto"] == 1, "reward": value, "metrics": metrics}
    value, metrics = await reward(history(with_read=False))
    cases["calculator_or_search_without_evidence_not_full_credit"] = {"pass": value < 1 and metrics["evidence_quality"] == 0, "reward": value, "metrics": metrics}
    value, metrics = await reward(history(finish=False))
    cases["missing_finish_exit"] = {"pass": value == 0 and metrics["finish_missing"] == 1, "reward": value, "metrics": metrics}
    return {"protocol": "2026-09-27-repaired-v1", "cases": cases, "pass": all(x["pass"] for x in cases.values())}

if __name__ == "__main__":
    out = asyncio.run(run())
    print(json.dumps(out, indent=2, sort_keys=True))
    raise SystemExit(0 if out["pass"] else 1)
