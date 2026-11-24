#!/usr/bin/env python3
"""Unified multi-benchmark tool environments and graders for:
  - FinRetrieval (local SQLite Daloopa structured DB)
  - FinAgentBench (cached SEC EDGAR XBRL + SEC filings + yfinance)
  - FinSearchComp (cached DuckDuckGo web search + fetch_url + yfinance + SEC EDGAR)
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import math
import operator
import os
import re
import sqlite3
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

ROOT = Path.home() / "financeharness-muse-glimmer"
DATA_DIR = ROOT / "data" / "multibench"
DB_PATH = DATA_DIR / "finretrieval_db.sqlite"
SEC_CACHE = DATA_DIR / "sec_cache"
WEB_CACHE = DATA_DIR / "web_cache"
FILING_CACHE = DATA_DIR / "filing_text_cache"

SEC_HEADERS = {"User-Agent": "FinanceRetrievalResearch/1.0 (research@example.com)"}
WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def safe_calculate(expression: str) -> str:
    try:
        expr = str(expression).strip().replace("^", "**")
        node = ast.parse(expr, mode="eval").body

        def _eval(n: ast.AST) -> float:
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
                return _OPS[type(n.op)](_eval(n.operand))
            if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
                left = _eval(n.left)
                right = _eval(n.right)
                if isinstance(n.op, ast.Pow) and abs(right) > 100:
                    raise ValueError("exponent too large")
                return _OPS[type(n.op)](left, right)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                fn = {
                    "abs": abs,
                    "round": round,
                    "min": min,
                    "max": max,
                    "sqrt": math.sqrt,
                    "log": math.log,
                }.get(n.func.id)
                if fn is None:
                    raise ValueError(f"unsupported function {n.func.id}")
                return float(fn(*[_eval(a) for a in n.args]))
            raise ValueError(f"unsupported syntax {type(n).__name__}")

        val = _eval(node)
        if not math.isfinite(val):
            return json.dumps({"error": "non-finite result"})
        return json.dumps({"expression": expr, "result": round(val, 8)})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:120]}"})


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 1}


class FinRetrievalDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def discover_companies(self, keywords: list[str] | str) -> str:
        if isinstance(keywords, str):
            keywords = [keywords]
        kw_list = [str(k).strip() for k in keywords if str(k).strip()]
        if not kw_list:
            return json.dumps({"error": "keywords must be a non-empty list of strings"})
        conn = self._conn()
        try:
            rows = conn.execute("SELECT company_id, ticker, name, latest_quarter FROM companies").fetchall()
        finally:
            conn.close()

        out: dict[str, list[dict[str, Any]]] = {}
        for kw in kw_list[:5]:
            kw_low = kw.lower()
            kw_toks = _tokenize(kw)
            scored = []
            for r in rows:
                ticker = (r["ticker"] or "").lower()
                name = (r["name"] or "").lower()
                score = 0.0
                if kw_low == ticker or kw_low == ticker.split(".")[0]:
                    score += 100.0
                if kw_low in name:
                    score += 50.0
                if kw_low in ticker:
                    score += 30.0
                toks = _tokenize(f"{ticker} {name}")
                overlap = len(kw_toks & toks)
                if overlap:
                    score += 10.0 * overlap / max(1, len(kw_toks))
                if score > 0:
                    scored.append((score, dict(r)))
            scored.sort(key=lambda x: (-x[0], x[1]["name"]))
            out[kw] = [item for _, item in scored[:12]]
        return json.dumps(out, ensure_ascii=False)

    def discover_company_series(
        self, company_id: int, keywords: list[str] | str, periods: list[str] | None = None
    ) -> str:
        if isinstance(keywords, str):
            keywords = [keywords]
        kw_toks = _tokenize(" ".join(str(k) for k in keywords))
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT series_id, full_series_name FROM series WHERE company_id = ?",
                (int(company_id),),
            ).fetchall()
            frows = conn.execute(
                "SELECT DISTINCT series_id, title FROM fundamentals WHERE company_id = ?",
                (int(company_id),),
            ).fetchall()
        finally:
            conn.close()

        by_id: dict[int, str] = {}
        for r in rows:
            by_id[int(r["series_id"])] = str(r["full_series_name"])
        for r in frows:
            sid = int(r["series_id"])
            if sid not in by_id and r["title"]:
                by_id[sid] = str(r["title"])

        if not by_id:
            return json.dumps({"company_id": company_id, "series": [], "note": "No series found for company_id."})

        scored = []
        for sid, name in by_id.items():
            toks = _tokenize(name)
            overlap = len(kw_toks & toks)
            substr = sum(1 for k in keywords if str(k).strip().lower() in name.lower())
            score = overlap * 3.0 + substr * 5.0
            scored.append((score, sid, name))
        scored.sort(key=lambda x: (-x[0], len(x[2]), x[1]))
        top = [{"series_id": sid, "full_series_name": name} for _, sid, name in scored[:25]]
        return json.dumps({"company_id": company_id, "count": len(by_id), "series": top}, ensure_ascii=False)

    def get_company_fundamentals(
        self, company_id: int, series_ids: list[int] | int, periods: list[str] | None = None
    ) -> str:
        if isinstance(series_ids, (int, str)):
            series_ids = [int(series_ids)]
        else:
            series_ids = [int(s) for s in series_ids][:20]
        if not series_ids:
            return json.dumps({"error": "series_ids must be non-empty"})

        conn = self._conn()
        try:
            ph = ",".join("?" * len(series_ids))
            srows = conn.execute(
                f"SELECT series_id, full_series_name FROM series WHERE company_id = ? AND series_id IN ({ph})",
                [int(company_id)] + series_ids,
            ).fetchall()
            titles = {str(r["full_series_name"]) for r in srows if r["full_series_name"]}

            frows = conn.execute(
                "SELECT series_id, record_id, title, fiscal_period, calendar_period, value_quarter, value_ytd, unit FROM fundamentals WHERE company_id = ?",
                (int(company_id),),
            ).fetchall()
        finally:
            conn.close()

        sid_set = set(series_ids)
        period_set = {str(p).upper().strip() for p in (periods or []) if str(p).strip()}
        matched = []
        for r in frows:
            if int(r["series_id"]) in sid_set or (r["title"] and str(r["title"]) in titles):
                rec = dict(r)
                if period_set:
                    fp = str(rec.get("fiscal_period") or "").upper()
                    cp = str(rec.get("calendar_period") or "").upper()
                    if not (
                        fp in period_set
                        or cp in period_set
                        or any(p in fp or p in cp for p in period_set)
                    ):
                        continue
                matched.append(rec)

        note = None
        if not matched and period_set:
            for r in frows:
                if int(r["series_id"]) in sid_set or (r["title"] and str(r["title"]) in titles):
                    matched.append(dict(r))
            if matched:
                note = "Exact period filter had 0 hits; returning all available periods for the requested series."

        matched.sort(key=lambda x: (str(x.get("title") or ""), str(x.get("fiscal_period") or ""), str(x.get("calendar_period") or "")))
        out: dict[str, Any] = {"company_id": company_id, "records": matched[:60]}
        if note:
            out["note"] = note
        return json.dumps(out, ensure_ascii=False)


_TICKER_CIK_CACHE: dict[str, str] | None = None


def _get_cik(ticker: str) -> str | None:
    global _TICKER_CIK_CACHE
    if _TICKER_CIK_CACHE is None:
        p = SEC_CACHE / "company_tickers.json"
        if not p.exists():
            SEC_CACHE.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                p.write_bytes(resp.read())
        raw = json.loads(p.read_text())
        _TICKER_CIK_CACHE = {str(v["ticker"]).upper(): f"{int(v['cik_str']):010d}" for v in raw.values()}
    return _TICKER_CIK_CACHE.get(str(ticker).upper().strip())


def _load_sec_json(ticker: str, kind: str) -> dict[str, Any] | None:
    t = str(ticker).upper().strip()
    SEC_CACHE.mkdir(parents=True, exist_ok=True)
    path = SEC_CACHE / f"{kind}_{t}.json"
    if path.exists():
        return json.loads(path.read_text())
    cik = _get_cik(t)
    if not cik:
        return None
    if kind == "companyfacts":
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    else:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        path.write_bytes(data)
        return json.loads(data)
    except Exception:
        return None


def tool_sec_company_facts(
    ticker: str, query: str, fiscal_years: list[int] | None = None, form: str | None = None
) -> str:
    data = _load_sec_json(ticker, "companyfacts")
    if not data:
        return json.dumps({"error": f"Could not load SEC companyfacts for ticker {ticker}"})
    facts = data.get("facts") or {}
    q_toks = _tokenize(query)
    fy_set = {int(y) for y in (fiscal_years or []) if str(y).isdigit()}
    form_up = str(form).upper().strip() if form else None

    candidates = []
    for taxonomy, concepts in facts.items():
        if not isinstance(concepts, dict):
            continue
        for cname, cdata in concepts.items():
            label = str(cdata.get("label") or "")
            desc = str(cdata.get("description") or "")
            cname_words = re.sub(r"([a-z])([A-Z])", r"\1 \2", cname)
            toks = _tokenize(f"{cname_words} {label}")
            overlap = len(q_toks & toks)
            if overlap == 0:
                desc_overlap = len(q_toks & _tokenize(desc))
                if desc_overlap < 2:
                    continue
                score = desc_overlap * 0.5
            else:
                # Favor high precision on concept tokens + exact label substring
                prec = overlap / max(1, len(toks))
                rec = overlap / max(1, len(q_toks))
                score = overlap * 3.0 + prec * 8.0 + rec * 6.0 + (5.0 if query.lower() in label.lower() else 0.0)
                if taxonomy == "us-gaap":
                    score += 1.0
            candidates.append((score, taxonomy, cname, label, cdata.get("units") or {}))

    candidates.sort(key=lambda x: -x[0])
    results = []
    for score, taxonomy, cname, label, units in candidates[:12]:
        entries = []
        for uname, ulist in units.items():
            for item in ulist:
                fy = item.get("fy")
                fm = str(item.get("form") or "").upper()
                if fy_set and (fy is None or int(fy) not in fy_set):
                    continue
                if form_up and fm != form_up:
                    continue
                entries.append(
                    {
                        "val": item.get("val"),
                        "unit": uname,
                        "fy": fy,
                        "fp": item.get("fp"),
                        "form": fm,
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "filed": item.get("filed"),
                    }
                )
        entries.sort(key=lambda e: (str(e.get("end") or ""), str(e.get("filed") or "")), reverse=True)
        dedup = []
        seen = set()
        for e in entries:
            k = (e["fy"], e["fp"], e["form"], e["start"], e["end"])
            if k not in seen:
                seen.add(k)
                dedup.append(e)
            if len(dedup) >= 8:
                break
        if dedup:
            results.append({"concept": f"{taxonomy}:{cname}", "label": label, "values": dedup})
    return json.dumps({"ticker": ticker.upper(), "query": query, "matches": results[:8]}, ensure_ascii=False)


def tool_sec_list_filings(ticker: str, form_types: list[str] | None = None, limit: int = 15) -> str:
    data = _load_sec_json(ticker, "submissions")
    if not data:
        return json.dumps({"error": f"Could not load SEC submissions for ticker {ticker}"})
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    fdates = recent.get("filingDate") or []
    rdates = recent.get("reportDate") or []
    accs = recent.get("accessionNumber") or []
    docs = recent.get("primaryDocument") or []
    want = {str(f).upper().strip() for f in (form_types or ["10-K", "10-Q", "8-K"])}
    out = []
    for fm, fd, rd, acc, doc in zip(forms, fdates, rdates, accs, docs):
        if str(fm).upper() in want:
            out.append(
                {
                    "form": fm,
                    "filing_date": fd,
                    "report_date": rd,
                    "accession_number": acc,
                    "primary_document": doc,
                }
            )
            if len(out) >= limit:
                break
    return json.dumps({"ticker": ticker.upper(), "entity": data.get("name"), "filings": out}, ensure_ascii=False)


def tool_sec_search_filing(ticker: str, accession_number: str, query: str) -> str:
    cik = _get_cik(ticker)
    if not cik:
        return json.dumps({"error": f"Unknown ticker {ticker}"})
    data = _load_sec_json(ticker, "submissions")
    if not data:
        return json.dumps({"error": f"No submissions for {ticker}"})
    recent = (data.get("filings") or {}).get("recent") or {}
    primary_doc = None
    for acc, doc in zip(recent.get("accessionNumber") or [], recent.get("primaryDocument") or []):
        if acc == accession_number:
            primary_doc = doc
            break
    if not primary_doc:
        return json.dumps({"error": f"Accession {accession_number} not found in recent filings for {ticker}"})

    FILING_CACHE.mkdir(parents=True, exist_ok=True)
    safe_acc = accession_number.replace("-", "")
    cache_file = FILING_CACHE / f"{ticker.upper()}_{safe_acc}.txt"
    if cache_file.exists():
        text = cache_file.read_text(errors="ignore")
    else:
        cik_int = str(int(cik))
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{safe_acc}/{primary_doc}"
        try:
            req = urllib.request.Request(url, headers=SEC_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_html = resp.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(raw_html, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = re.sub(r"[ \t]+", " ", soup.get_text("\n"))
            text = re.sub(r"\n{3,}", "\n\n", text)
            cache_file.write_text(text)
        except Exception as exc:
            return json.dumps({"error": f"Failed fetching filing: {type(exc).__name__}: {str(exc)[:120]}"})

    q_toks = _tokenize(query)
    chunks = [text[i : i + 1200] for i in range(0, len(text), 900)]
    scored = []
    for idx, ch in enumerate(chunks):
        ch_low = ch.lower()
        overlap = len(q_toks & _tokenize(ch))
        phrase = 5 if query.lower() in ch_low else 0
        if overlap + phrase > 0:
            scored.append((overlap * 2 + phrase, idx, ch))
    scored.sort(key=lambda x: -x[0])
    excerpts = [{"chunk": idx, "score": sc, "text": ch[:1200]} for sc, idx, ch in scored[:5]]
    return json.dumps(
        {"ticker": ticker.upper(), "accession_number": accession_number, "query": query, "excerpts": excerpts},
        ensure_ascii=False,
    )


def tool_yfinance_history(
    ticker: str, start_date: str, end_date: str, auto_adjust: bool = False
) -> str:
    import yfinance as yf

    try:
        df = yf.download(
            str(ticker).strip(),
            start=str(start_date).strip(),
            end=str(end_date).strip(),
            auto_adjust=bool(auto_adjust),
            progress=False,
        )
        if df is None or df.empty:
            return json.dumps({"ticker": ticker, "rows": [], "note": "No price data found for range."})
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        rows = []
        for idx, r in df.head(40).iterrows():
            rows.append(
                {
                    "date": str(idx)[:10],
                    "open": round(float(r["Open"]), 4) if "Open" in r and math.isfinite(float(r["Open"])) else None,
                    "high": round(float(r["High"]), 4) if "High" in r and math.isfinite(float(r["High"])) else None,
                    "low": round(float(r["Low"]), 4) if "Low" in r and math.isfinite(float(r["Low"])) else None,
                    "close": round(float(r["Close"]), 4) if "Close" in r and math.isfinite(float(r["Close"])) else None,
                    "volume": int(r["Volume"]) if "Volume" in r and math.isfinite(float(r["Volume"])) else None,
                }
            )
        return json.dumps(
            {"ticker": ticker, "auto_adjust": bool(auto_adjust), "rows": rows},
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:140]}"})


def tool_yfinance_info(ticker: str) -> str:
    import yfinance as yf

    try:
        tk = yf.Ticker(str(ticker).strip())
        info = tk.info or {}
        keep = [
            "shortName", "longName", "sector", "industry", "currency", "exchange",
            "marketCap", "enterpriseValue", "totalRevenue", "grossProfits", "ebitda",
            "netIncomeToCommon", "trailingEps", "forwardEps", "trailingPE", "forwardPE",
            "dividendRate", "dividendYield", "sharesOutstanding", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ]
        summary = {k: info.get(k) for k in keep if info.get(k) is not None}
        return json.dumps({"ticker": ticker, "info": summary}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:140]}"})


def tool_web_search(query: str, max_results: int = 6) -> str:
    WEB_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"ddgs:{query}:{max_results}".encode()).hexdigest()
    cpath = WEB_CACHE / f"search_{key}.json"
    if cpath.exists():
        return cpath.read_text()
    try:
        from ddgs import DDGS

        res = []
        for bk in ("auto", "bing", "yahoo", "lite"):
            try:
                res = list(DDGS().text(str(query), max_results=min(int(max_results), 10), backend=bk))
                if res:
                    break
            except Exception:
                continue
        out = json.dumps(
            {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in res
                ],
            },
            ensure_ascii=False,
        )
        cpath.write_text(out)
        return out
    except Exception as exc:
        return json.dumps({"error": f"web_search failed: {type(exc).__name__}: {str(exc)[:140]}"})


def tool_fetch_url(url: str, query: str | None = None) -> str:
    WEB_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"url:{url}".encode()).hexdigest()
    cpath = WEB_CACHE / f"url_{key}.txt"
    if cpath.exists():
        text = cpath.read_text(errors="ignore")
    else:
        try:
            req = urllib.request.Request(str(url), headers=WEB_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read(2_000_000).decode("utf-8", errors="ignore")
            soup = BeautifulSoup(raw, "lxml")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = re.sub(r"[ \t]+", " ", soup.get_text("\n"))
            text = re.sub(r"\n{3,}", "\n\n", text)
            cpath.write_text(text)
        except Exception as exc:
            return json.dumps({"error": f"fetch_url failed: {type(exc).__name__}: {str(exc)[:140]}"})

    if query:
        q_toks = _tokenize(query)
        chunks = [text[i : i + 1200] for i in range(0, min(len(text), 80_000), 900)]
        scored = sorted(
            [(len(q_toks & _tokenize(ch)), idx, ch) for idx, ch in enumerate(chunks)],
            key=lambda x: -x[0],
        )
        excerpts = [ch for sc, _, ch in scored[:4] if sc > 0]
        if excerpts:
            return json.dumps({"url": url, "excerpts": excerpts}, ensure_ascii=False)
    return json.dumps({"url": url, "text": text[:4000]}, ensure_ascii=False)


class MultiBenchSession:
    """One stateful episode for FinRetrieval, FinAgentBench, or FinSearchComp."""

    def __init__(self, benchmark: str, fr_db: FinRetrievalDB | None = None):
        self.benchmark = benchmark
        self.fr_db = fr_db or FinRetrievalDB()
        self.accepted: dict[str, Any] | None = None
        self.seen_sigs: set[str] = set()

    def system_prompt(self) -> str:
        if self.benchmark == "finretrieval":
            return (
                "You are a precision financial data retrieval agent. Use the structured financial "
                "database tools (`discover_companies`, `discover_company_series`, `get_company_fundamentals`) "
                "and `calculate` to retrieve the exact requested value, checking fiscal vs. calendar period "
                "and units carefully. Pay close attention to whether the question asks for a quarterly value "
                "(`value_quarter`), year-to-date/half-year value (`value_ytd`), or sum across periods. "
                "Avoid repeating identical tool calls. Once you have verified the exact figure, call `finish`."
            )
        if self.benchmark == "finagentbench":
            return (
                "You are an expert SEC filings research agent. Use `sec_company_facts`, `sec_list_filings`, "
                "`sec_search_filing`, `yfinance_info`, and `calculate` to answer financial questions accurately. "
                "Check fiscal year, units, and signs carefully. If a question is adversarial, asks about a "
                "non-existent metric/filing, or cannot be answered from SEC filings, explicitly state why and "
                "set `is_unanswerable=true` in `finish`. Avoid repeating identical tool calls. Submit your "
                "final answer by calling `finish`."
            )
        return (
            "You are an expert financial research and search agent. Use `sec_company_facts` for US public-company "
            "quarterly/annual financials (revenue, margins, income, EPS), `yfinance_history` for historical stock "
            "prices (check adjusted vs. unadjusted), `web_search` and `fetch_url` for macro stats, index changes, "
            "and event dates, and `calculate` for arithmetic. Do not burn your tool budget on repetitive searches: "
            "once you have enough evidence, synthesize your concise answer and call `finish`."
        )

    def tool_specs(self) -> list[dict[str, Any]]:
        calc_spec = {
            "name": "calculate",
            "description": "Evaluate a mathematical expression accurately (supports +, -, *, /, **, abs, round, sqrt).",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
        if self.benchmark == "finretrieval":
            return [
                {
                    "name": "discover_companies",
                    "description": "Search the structured financial database for companies by name or ticker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["keywords"],
                    },
                },
                {
                    "name": "discover_company_series",
                    "description": "Search available financial/KPI series for a company_id by keywords.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_id": {"type": "integer"},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["company_id", "keywords"],
                    },
                },
                {
                    "name": "get_company_fundamentals",
                    "description": "Retrieve fundamental/KPI time-series values for given company_id and series_ids.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_id": {"type": "integer"},
                            "series_ids": {"type": "array", "items": {"type": "integer"}},
                            "periods": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["company_id", "series_ids"],
                    },
                },
                calc_spec,
                {
                    "name": "finish",
                    "description": "Submit the final verified answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "Concise final answer with value and unit."},
                            "numeric_value": {"type": "string", "description": "Numeric value as string if applicable."},
                            "unit": {"type": "string", "description": "Unit/scale (e.g. USD millions, %, Actual)."},
                        },
                        "required": ["answer"],
                    },
                },
            ]
        if self.benchmark == "finagentbench":
            return [
                {
                    "name": "sec_company_facts",
                    "description": "Search SEC EDGAR XBRL financial facts for a US ticker (e.g., AAPL, MSFT).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "query": {"type": "string"},
                            "fiscal_years": {"type": "array", "items": {"type": "integer"}},
                            "form": {"type": "string", "description": "Optional form filter like 10-K or 10-Q."},
                        },
                        "required": ["ticker", "query"],
                    },
                },
                {
                    "name": "sec_list_filings",
                    "description": "List recent SEC filings (10-K, 10-Q, 8-K) for a ticker to get accession numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "form_types": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["ticker"],
                    },
                },
                {
                    "name": "sec_search_filing",
                    "description": "Search the full text of a specific SEC filing by ticker and accession_number.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "accession_number": {"type": "string"},
                            "query": {"type": "string"},
                        },
                        "required": ["ticker", "accession_number", "query"],
                    },
                },
                {
                    "name": "yfinance_info",
                    "description": "Get summary company financials and market metrics via Yahoo Finance.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string"}},
                        "required": ["ticker"],
                    },
                },
                calc_spec,
                {
                    "name": "finish",
                    "description": "Submit the final verified answer (or mark is_unanswerable=true for adversarial/unanswerable questions).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "numeric_value": {"type": "string"},
                            "unit": {"type": "string"},
                            "is_unanswerable": {"type": "boolean"},
                        },
                        "required": ["answer"],
                    },
                },
            ]
        return [
            {
                "name": "web_search",
                "description": "Search the web for financial data, macro statistics, dates, news, and filings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "fetch_url",
                "description": "Fetch and extract text from a web page URL, optionally filtering by query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "yfinance_history",
                "description": "Get historical daily stock prices (Open, High, Low, Close, Volume) for a ticker and date range (YYYY-MM-DD).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "auto_adjust": {"type": "boolean", "description": "False for unadjusted prices (default), True for split/dividend-adjusted."},
                    },
                    "required": ["ticker", "start_date", "end_date"],
                },
            },
            {
                "name": "yfinance_info",
                "description": "Get company summary financials and market metrics via Yahoo Finance.",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
            {
                "name": "sec_company_facts",
                "description": "Search SEC EDGAR XBRL financial facts for a US ticker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "query": {"type": "string"},
                        "fiscal_years": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["ticker", "query"],
                },
            },
            calc_spec,
            {
                "name": "finish",
                "description": "Submit the final verified answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "Concise final answer with exact units/rounding requested."},
                    },
                    "required": ["answer"],
                },
            },
        ]

    def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        if name == "finish":
            ans = str(args.get("answer") or "").strip()
            if not ans:
                return json.dumps({"error": "finish requires a non-empty answer string."}), False
            self.accepted = dict(args)
            return json.dumps({"status": "accepted", "submission": self.accepted}, ensure_ascii=False), True

        sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        is_repeat = sig in self.seen_sigs
        self.seen_sigs.add(sig)

        try:
            if name == "calculate":
                out = safe_calculate(args.get("expression", ""))
            elif name == "discover_companies":
                out = self.fr_db.discover_companies(args.get("keywords", []))
            elif name == "discover_company_series":
                out = self.fr_db.discover_company_series(
                    args.get("company_id", 0), args.get("keywords", []), args.get("periods")
                )
            elif name == "get_company_fundamentals":
                out = self.fr_db.get_company_fundamentals(
                    args.get("company_id", 0), args.get("series_ids", []), args.get("periods")
                )
            elif name == "sec_company_facts":
                out = tool_sec_company_facts(
                    args.get("ticker", ""),
                    args.get("query", ""),
                    args.get("fiscal_years"),
                    args.get("form"),
                )
            elif name == "sec_list_filings":
                out = tool_sec_list_filings(args.get("ticker", ""), args.get("form_types"))
            elif name == "sec_search_filing":
                out = tool_sec_search_filing(
                    args.get("ticker", ""), args.get("accession_number", ""), args.get("query", "")
                )
            elif name == "yfinance_history":
                out = tool_yfinance_history(
                    args.get("ticker", ""),
                    args.get("start_date", ""),
                    args.get("end_date", ""),
                    args.get("auto_adjust", False),
                )
            elif name == "yfinance_info":
                out = tool_yfinance_info(args.get("ticker", ""))
            elif name == "web_search":
                out = tool_web_search(args.get("query", ""), args.get("max_results", 6))
            elif name == "fetch_url":
                out = tool_fetch_url(args.get("url", ""), args.get("query"))
            else:
                out = json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as exc:
            out = json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:140]}"})

        if len(out) > 8000:
            out = out[:7900] + "\n...[truncated]"
        if is_repeat:
            out += (
                f"\n[HARNESS NOTE: You already executed this exact {name} call earlier. "
                "Do not repeat identical tool calls — try a different query/tool or call `finish`.]"
            )
        return out, False


class MultiBenchJudge:
    """Fast deterministic + disk-cached DeepSeek-V4.1-Flash grader for all 3 benchmarks."""

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.fireworks.ai/inference/v1/chat/completions",
        model: str = "accounts/fireworks/models/deepseek-v4p1-flash",
        cache_dir: str | Path = ROOT / "run" / "multibench_judge_cache",
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _call_llm_sync(self, sys_prompt: str, user_prompt: str) -> str:
        digest = hashlib.sha256(
            json.dumps([self.model, sys_prompt, user_prompt], ensure_ascii=False).encode()
        ).hexdigest()
        cpath = self.cache_dir / f"{digest}.json"
        if cpath.exists():
            return json.loads(cpath.read_text())["content"]

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    env = json.load(resp)
                content = str(env["choices"][0]["message"].get("content") or "")
                with tempfile.NamedTemporaryFile(mode="w", dir=self.cache_dir, delete=False) as tf:
                    json.dump({"content": content}, tf, ensure_ascii=False)
                    tmp = tf.name
                os.replace(tmp, cpath)
                return content
            except Exception:
                if attempt == 2:
                    raise
                import time
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    @staticmethod
    def _parse_float(s: Any) -> float | None:
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return float(s) if math.isfinite(float(s)) else None
        cleaned = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if not m:
            return None
        try:
            v = float(m.group(0))
            return v if math.isfinite(v) else None
        except Exception:
            return None

    async def score(
        self,
        row: dict[str, Any],
        submission: dict[str, Any] | None,
        calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        F = int(submission is not None)
        if not submission:
            return {"F": 0, "correct": 0, "reward": 0.0, "method": "no_finish"}

        bm = row["benchmark"]
        cand_text = str(submission.get("answer") or "")
        cand_num = submission.get("numeric_value")
        cand_unit = submission.get("unit")
        full_cand = cand_text
        if cand_num:
            full_cand += f" [numeric_value={cand_num}, unit={cand_unit or ''}]"
        if submission.get("is_unanswerable"):
            full_cand += " [is_unanswerable=true]"

        if bm == "finretrieval":
            gv = self._parse_float(row.get("gold_value"))
            cv = self._parse_float(cand_num) if cand_num is not None else self._parse_float(cand_text)
            if gv is not None and cv is not None:
                tol = max(1e-6, abs(gv) * 0.015)
                if abs(gv - cv) <= tol:
                    return {"F": 1, "correct": 1, "reward": 1.0, "method": "numeric_exact"}
            sys_p = (
                "You are a strict financial benchmark grader. Compare the student's final answer against "
                "the ground truth value and unit. Accept equivalent units/scales (e.g., $1.5B == $1,500M) "
                "and rounding within 1.5%. Return JSON only: {\"correct\": true/false, \"reason\": \"brief\"}."
            )
            usr_p = json.dumps(
                {
                    "question": row["question"],
                    "ground_truth_value": row.get("gold_value"),
                    "ground_truth_unit": row.get("gold_unit"),
                    "ground_truth_answer": row.get("gold_answer"),
                    "student_answer": full_cand,
                },
                ensure_ascii=False,
            )
            raw = await asyncio.to_thread(self._call_llm_sync, sys_p, usr_p)
            ok = self._parse_bool_json(raw)
            return {"F": 1, "correct": int(ok), "reward": 1.0 if ok else 0.05, "method": "llm_judge"}

        if bm == "finagentbench":
            gn = self._parse_float(row.get("gold_answer_numeric"))
            tol_pct = float(row.get("tolerance_pct") or 1.0) / 100.0
            cv = self._parse_float(cand_num) if cand_num is not None else None
            if gn is not None and cv is not None and row.get("type") != "adversarial":
                if abs(gn - cv) <= max(1e-6, abs(gn) * max(tol_pct, 0.015)):
                    return {"F": 1, "correct": 1, "reward": 1.0, "method": "numeric_exact"}
            sys_p = (
                "You are a strict financial benchmark grader for FinAgentBench. Compare the student's answer "
                "against the reference gold answer and explanation. For numeric questions, accept equivalent "
                "scales/units (e.g., $391.0 billion == 391,035 million == 391035000000) within the stated "
                "tolerance. For adversarial/unanswerable questions, mark correct=true ONLY if the student "
                "correctly identifies the false premise or unanswerability rather than fabricating an answer. "
                "Return JSON only: {\"correct\": true/false, \"reason\": \"brief\"}."
            )
            usr_p = json.dumps(
                {
                    "question": row["question"],
                    "question_type": row.get("type"),
                    "tolerance_pct": row.get("tolerance_pct"),
                    "gold_answer": row.get("gold_answer"),
                    "gold_answer_numeric": row.get("gold_answer_numeric"),
                    "explanation": row.get("explanation"),
                    "student_answer": full_cand,
                },
                ensure_ascii=False,
            )
            raw = await asyncio.to_thread(self._call_llm_sync, sys_p, usr_p)
            ok = self._parse_bool_json(raw)
            return {"F": 1, "correct": int(ok), "reward": 1.0 if ok else 0.05, "method": "llm_judge"}

        sys_p = str(row.get("judge_system_prompt") or "") + (
            "\n\nEnd your response with a single JSON line: {\"correct\": true} or {\"correct\": false}."
        )
        tmpl = str(row.get("judge_prompt_template") or "<Question>:\n{prompt}\n\n<Reference Answer>:\n{response_reference}\n\n<Student Answer>:\n{response}")
        usr_p = (
            tmpl.replace("{prompt}", row["question"])
            .replace("{response_reference}", str(row.get("gold_answer") or ""))
            .replace("{response}", full_cand)
        )
        raw = await asyncio.to_thread(self._call_llm_sync, sys_p, usr_p)
        ok = self._parse_bool_json(raw)
        return {"F": 1, "correct": int(ok), "reward": 1.0 if ok else 0.05, "method": "finsearchcomp_judge"}

    @staticmethod
    def _parse_bool_json(raw: str) -> bool:
        txt = raw.strip()
        for m in reversed(list(re.finditer(r"\{[^{}]*\}", txt))):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "correct" in obj:
                    return bool(obj["correct"])
            except Exception:
                continue
        low = txt.lower()
        if '"correct": true' in low or "verdict: correct" in low or low.endswith("yes"):
            return True
        return False
