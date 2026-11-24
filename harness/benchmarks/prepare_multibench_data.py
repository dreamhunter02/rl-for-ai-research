#!/usr/bin/env python3
"""Prepare local databases, SEC EDGAR cache, and stratified train/eval splits for:
  1. FinRetrieval (daloopa/finretrieval, arXiv:2603.04403)
  2. FinAgentBench (finagent-benchmark/finagent-benchmark)
  3. FinSearchComp (ByteSeedXpert/FinSearchComp, Global T2+T3)
"""
from __future__ import annotations

import json
import random
import sqlite3
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

ROOT = Path.home() / "financeharness-muse-glimmer"
DATA_DIR = ROOT / "data" / "multibench"
SEC_CACHE = DATA_DIR / "sec_cache"
DB_PATH = DATA_DIR / "finretrieval_db.sqlite"

SEC_HEADERS = {"User-Agent": "FinanceRetrievalResearch/1.0 (research@example.com)"}


def _extract_mcp_text(output: Any) -> str | None:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("text")]
            if parts:
                return "\n".join(parts)
        if isinstance(output.get("text"), str):
            return output["text"]
    return None


def build_finretrieval_db() -> dict[str, int]:
    fr_dir = sorted(
        Path.home().glob(".cache/huggingface/hub/datasets--daloopa--finretrieval/snapshots/*")
    )[0]
    tt = pd.read_parquet(fr_dir / "tool_traces.parquet")

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE companies (
            company_id INTEGER PRIMARY KEY,
            ticker TEXT,
            name TEXT,
            latest_quarter TEXT,
            raw_json TEXT
        );
        CREATE TABLE series (
            company_id INTEGER,
            series_id INTEGER,
            full_series_name TEXT,
            PRIMARY KEY (company_id, series_id)
        );
        CREATE TABLE fundamentals (
            company_id INTEGER,
            series_id INTEGER,
            record_id TEXT,
            title TEXT,
            fiscal_period TEXT,
            calendar_period TEXT,
            value_quarter TEXT,
            value_ytd TEXT,
            unit TEXT,
            raw_json TEXT,
            PRIMARY KEY (company_id, series_id, fiscal_period, calendar_period, record_id)
        );
        CREATE INDEX idx_series_cid ON series(company_id);
        CREATE INDEX idx_fund_cid_sid ON fundamentals(company_id, series_id);
        """
    )

    n_comp = n_ser = n_fund = 0
    for raw_tc in tt["tool_calls"].dropna():
        try:
            arr = json.loads(raw_tc)
        except Exception:
            continue
        for item in arr:
            if not isinstance(item, dict) or item.get("is_error"):
                continue
            nm = str(item.get("name") or "")
            inp = item.get("input") or {}
            txt = _extract_mcp_text(item.get("output"))
            if not txt:
                continue
            try:
                parsed = json.loads(txt)
            except Exception:
                continue

            if nm.endswith("discover_companies"):
                entries = []
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            entries.extend(v)
                elif isinstance(parsed, list):
                    entries = parsed
                for c in entries:
                    if isinstance(c, dict) and c.get("company_id") is not None:
                        cur.execute(
                            "INSERT OR REPLACE INTO companies VALUES (?, ?, ?, ?, ?)",
                            (
                                int(c["company_id"]),
                                str(c.get("ticker") or ""),
                                str(c.get("name") or ""),
                                str(c.get("latest_quarter") or ""),
                                json.dumps(c, ensure_ascii=False),
                            ),
                        )
                        n_comp += 1

            elif nm.endswith("discover_company_series"):
                cid = inp.get("company_id")
                if cid is None:
                    continue
                entries = parsed if isinstance(parsed, list) else parsed.get("series", []) if isinstance(parsed, dict) else []
                for s in entries:
                    if isinstance(s, dict) and s.get("id") is not None:
                        cur.execute(
                            "INSERT OR REPLACE INTO series VALUES (?, ?, ?)",
                            (
                                int(cid),
                                int(s["id"]),
                                str(s.get("full_series_name") or s.get("name") or s.get("title") or ""),
                            ),
                        )
                        n_ser += 1

            elif nm.endswith("get_company_fundamentals"):
                cid = inp.get("company_id")
                if cid is None:
                    continue
                entries = parsed if isinstance(parsed, list) else parsed.get("fundamentals", []) if isinstance(parsed, dict) else []
                for f in entries:
                    if not isinstance(f, dict):
                        continue
                    sid = f.get("series_id") or f.get("fundamental_id")
                    if sid is None and isinstance(inp.get("series_ids"), list) and len(inp["series_ids"]) == 1:
                        sid = inp["series_ids"][0]
                    if sid is None:
                        sid = f.get("id", 0)
                    rec_id = str(f.get("id") or "")
                    fp = str(f.get("fiscal_period") or f.get("period") or "")
                    cp = str(f.get("calendar_period") or "")
                    cur.execute(
                        "INSERT OR REPLACE INTO fundamentals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            int(cid),
                            int(sid),
                            rec_id,
                            str(f.get("title") or f.get("full_series_name") or ""),
                            fp,
                            cp,
                            "" if f.get("value_quarter") is None else str(f.get("value_quarter")),
                            "" if f.get("value_year_to_date") is None else str(f.get("value_year_to_date")),
                            str(f.get("unit") or ""),
                            json.dumps(f, ensure_ascii=False),
                        ),
                    )
                    n_fund += 1

    conn.commit()
    counts = {
        "companies": cur.execute("SELECT COUNT(*) FROM companies").fetchone()[0],
        "series": cur.execute("SELECT COUNT(*) FROM series").fetchone()[0],
        "fundamentals": cur.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0],
    }
    conn.close()
    print("Built finretrieval_db.sqlite:", counts)
    return counts


def stratified_split(
    rows: list[dict[str, Any]],
    key_fn,
    eval_n: int,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(key_fn(r))].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    eval_rows, train_rows = [], []
    total = len(rows)
    for k in sorted(buckets):
        b = buckets[k]
        take = max(1, round(eval_n * len(b) / total)) if b else 0
        eval_rows.extend(b[:take])
        train_rows.extend(b[take:])
    rng.shuffle(eval_rows)
    rng.shuffle(train_rows)
    # Adjust exact eval_n
    while len(eval_rows) > eval_n:
        train_rows.append(eval_rows.pop())
    while len(eval_rows) < eval_n and train_rows:
        eval_rows.append(train_rows.pop())
    return train_rows, eval_rows


def prepare_splits() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. FinRetrieval (500 -> 400 train / 50 dev / 50 eval)
    fr_dir = sorted(
        Path.home().glob(".cache/huggingface/hub/datasets--daloopa--finretrieval/snapshots/*")
    )[0]
    qdf = pd.read_parquet(fr_dir / "questions.parquet")
    fr_rows = []
    for _, r in qdf.iterrows():
        fr_rows.append(
            {
                "benchmark": "finretrieval",
                "id": f"finretrieval_{int(r["index"]):04d}",
                "index": int(r["index"]),
                "question": str(r["question"]),
                "gold_answer": str(r["answer"]),
                "gold_value": str(r["value"]),
                "gold_unit": str(r["unit"]),
                "category": str(r["category"]),
                "ticker": str(r["ticker"]),
                "company": str(r["company"]),
                "country": str(r["country"]),
                "is_us": bool(r["is_us"]),
                "fiscal_period": str(r["fiscal_period"]),
                "calendar_period": str(r["calendar_period"]),
                "period_type": str(r["period_type"]),
            }
        )
    fr_rest, fr_eval = stratified_split(fr_rows, lambda r: (r["category"], r["is_us"]), 50, seed=42)
    fr_train, fr_dev = stratified_split(fr_rest, lambda r: (r["category"], r["is_us"]), 50, seed=43)
    for name, subset in [("finretrieval_train", fr_train), ("finretrieval_dev", fr_dev), ("finretrieval_eval", fr_eval)]:
        (DATA_DIR / f"{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in subset) + "\n")
    print(f"FinRetrieval splits: train={len(fr_train)}, dev={len(fr_dev)}, eval={len(fr_eval)}")

    # 2. FinAgentBench (133 -> 93 train / 40 eval)
    fa_dir = sorted(
        Path.home().glob(".cache/huggingface/hub/datasets--finagent-benchmark--finagent-benchmark/snapshots/*")
    )[0]
    fa_raw = json.loads((fa_dir / "benchmark_questions.json").read_text())
    fa_rows = []
    for r in fa_raw:
        fa_rows.append(
            {
                "benchmark": "finagentbench",
                "id": f"finagentbench_{r["id"]}",
                "orig_id": r["id"],
                "question": str(r["question"]),
                "type": str(r.get("type") or ""),
                "difficulty": str(r.get("difficulty") or ""),
                "source_companies": list(r.get("source_companies") or []),
                "source_filing_types": list(r.get("source_filing_types") or []),
                "tolerance_pct": float(r.get("tolerance_pct") or 1.0),
                "gold_answer": str(r.get("gold_answer") or ""),
                "gold_answer_numeric": r.get("gold_answer_numeric"),
                "gold_evidence": str(r.get("gold_evidence") or ""),
                "explanation": str(r.get("explanation") or ""),
            }
        )
    fa_train, fa_eval = stratified_split(fa_rows, lambda r: (r["type"], r["difficulty"]), 40, seed=42)
    for name, subset in [("finagentbench_train", fa_train), ("finagentbench_eval", fa_eval)]:
        (DATA_DIR / f"{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in subset) + "\n")
    print(f"FinAgentBench splits: train={len(fa_train)}, eval={len(fa_eval)}")

    # 3. FinSearchComp Global T2+T3 (203 -> 153 train / 50 eval)
    fsc = load_dataset("ByteSeedXpert/FinSearchComp")["train"].to_pandas()
    fsc_hist = fsc[
        fsc["label"].isin(
            [
                "Simple_Historical_Lookup(Global)",
                "Complex_Historical_Investigation(Global)",
            ]
        )
    ]
    fsc_rows = []
    for _, r in fsc_hist.iterrows():
        fsc_rows.append(
            {
                "benchmark": "finsearchcomp",
                "id": f"finsearchcomp_{r["prompt_id"]}",
                "prompt_id": str(r["prompt_id"]),
                "question": str(r["prompt"]),
                "gold_answer": str(r["response_reference"]),
                "label": str(r["label"]),
                "judge_system_prompt": str(r["judge_system_prompt"]),
                "judge_prompt_template": str(r["judge_prompt_template"]),
            }
        )
    fsc_train, fsc_eval = stratified_split(fsc_rows, lambda r: r["label"], 50, seed=42)
    for name, subset in [("finsearchcomp_train", fsc_train), ("finsearchcomp_eval", fsc_eval)]:
        (DATA_DIR / f"{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in subset) + "\n")
    print(f"FinSearchComp Global T2+T3 splits: train={len(fsc_train)}, eval={len(fsc_eval)}")


def cache_sec_edgar_for_finagentbench() -> None:
    SEC_CACHE.mkdir(parents=True, exist_ok=True)
    tickers_file = SEC_CACHE / "company_tickers.json"
    if not tickers_file.exists():
        req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            tickers_file.write_bytes(resp.read())
    ticker_map_raw = json.loads(tickers_file.read_text())
    ticker_to_cik = {
        str(v["ticker"]).upper(): f"{int(v["cik_str"]):010d}"
        for v in ticker_map_raw.values()
    }

    fa_dir = sorted(
        Path.home().glob(".cache/huggingface/hub/datasets--finagent-benchmark--finagent-benchmark/snapshots/*")
    )[0]
    fa_raw = json.loads((fa_dir / "benchmark_questions.json").read_text())
    tickers = sorted({str(c).upper() for r in fa_raw for c in (r.get("source_companies") or [])})
    print(f"Pre-caching SEC EDGAR CompanyFacts for {len(tickers)} FinAgentBench tickers...")
    ok = 0
    for t in tickers:
        cik = ticker_to_cik.get(t)
        if not cik:
            print(f"  WARN: no CIK for {t}")
            continue
        facts_path = SEC_CACHE / f"companyfacts_{t}.json"
        if not facts_path.exists():
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            try:
                req = urllib.request.Request(url, headers=SEC_HEADERS)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    facts_path.write_bytes(resp.read())
                time.sleep(0.15)
            except Exception as e:
                print(f"  WARN: failed facts for {t} ({cik}): {e}")
                continue
        sub_path = SEC_CACHE / f"submissions_{t}.json"
        if not sub_path.exists():
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            try:
                req = urllib.request.Request(url, headers=SEC_HEADERS)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    sub_path.write_bytes(resp.read())
                time.sleep(0.15)
            except Exception as e:
                print(f"  WARN: failed submissions for {t} ({cik}): {e}")
        ok += 1
    print(f"Cached SEC EDGAR CompanyFacts for {ok}/{len(tickers)} companies.")


if __name__ == "__main__":
    build_finretrieval_db()
    prepare_splits()
    cache_sec_edgar_for_finagentbench()
