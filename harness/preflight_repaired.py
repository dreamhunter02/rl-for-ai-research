"""E0 preflight for the repaired FinanceBench/Tinker protocol."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
from pathlib import Path

import financebench_harness as hb
import finance_env as fe


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def tool_checks(index: hb.StructuredIndex) -> dict:
    tool_obj = fe.Bm25Tool(index)
    specs = [tool_obj.bm25_search.to_spec(), tool_obj.grep_document.to_spec(), tool_obj.search_tables.to_spec(), tool_obj.read.to_spec(), tool_obj.read_table.to_spec(), tool_obj.calculate.to_spec(), tool_obj.finish.to_spec()]
    arrays = {spec["name"]: [name for name, value in (spec.get("parameters", {}).get("properties") or {}).items() if value.get("type") == "array"] for spec in specs}
    from tinker_cookbook.tool_use import ToolInput
    search = await tool_obj.bm25_search.run(ToolInput({"query_list": ["revenue", "cash"], "top_k": 5}, "preflight-search"))
    read = await tool_obj.read.run(ToolInput({"document_id": index.pages[0]["document_id"], "page": index.pages[0]["page"], "start": 0, "end": hb.MAX_READ}, "preflight-read"))
    search_body = json.loads(search.messages[0]["content"])
    read_body = json.loads(read.messages[0]["content"])
    return {
        "tool_array_fields": arrays,
        "search_chars": len(search.messages[0]["content"]),
        "search_valid_json": True,
        "search_returned_hits": search_body.get("returned_hits"),
        "max_search_snippet_chars": max([len(x.get("snippet", "")) for x in search_body.get("prose_hits", []) + search_body.get("table_hits", [])] or [0]),
        "read_chars": len(read.messages[0]["content"]),
        "read_valid_json": True,
        "read_has_more": read_body.get("has_more"),
        "read_next_start": read_body.get("next_start"),
    }


def main() -> None:
    train = fe.load_financebench("train96")
    dev = fe.load_financebench("dev")
    evaluation = fe.load_financebench("eval")
    index = hb.build_index(max_docs=1, use_cache=True)
    result = {
        "protocol": "2026-09-27-repaired-v1",
        "python": platform.python_version(),
        "base": str(hb.BASE),
        "pdf_pages": os.environ.get("FINANCEBENCH_PDF_PAGES", "1"),
        "split_counts": {"train96": len(train), "dev12": len(dev), "eval42": len(evaluation)},
        "split_disjoint": not ({r["financebench_id"] for r in train} & {r["financebench_id"] for r in dev}),
        "index_probe": {"pages": len(index.pages), "passages": len(index.passages), "tables": len(index.tables)},
        "numeric_regressions": {
            "negative_zero": hb._numbers_match("Margin was -4.2%.", "Margin was 4.2%.") == (False, 0),
            "scale_equivalent": hb._numbers_match("Revenue was $4.2B.", "Revenue was $4200M.") == (True, 1),
            "scale_mismatch": hb._numbers_match("Revenue was $4.2B.", "Revenue was $4.2M.") == (False, 0),
        },
        "source_hashes": {str(p): sha256(p) for p in [Path("harness/financebench_harness.py"), Path("harness/finance_env.py"), Path("harness/reward_calculation.py"), Path("configs/nemotron35_lightning_grpo_reward_v2.json")]},
    }
    result["tool_checks"] = asyncio.run(tool_checks(index))
    out = Path("results/paper_2026_rl4llm_agents/e0_preflight.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
