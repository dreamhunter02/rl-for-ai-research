"""Structured, agentic FinanceBench retrieval harness.

The harness keeps retrieval mechanical and auditable: page-preserving PDF text,
filename metadata, separate prose/table candidate indexes, BM25 search, bounded
regex search, and provenance-preserving reads. It deliberately does not use
answer-derived metadata or an LLM parser.
"""
from __future__ import annotations

import ast
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BASE = Path(os.environ.get("FINANCEBENCH_BASE", os.path.expanduser("~/Documents/Research/rl-for-ai-research"))).expanduser()
FILINGS = Path(os.environ.get("FINANCEBENCH_FILINGS", BASE / "filings")).expanduser()
TEXTDIR = Path(os.environ.get("FINANCEBENCH_TEXT", BASE / "text")).expanduser()
DATA = Path(os.environ.get("FINANCEBENCH_DATA", BASE / "data")).expanduser()
CACHE = Path(os.environ.get("FINANCEBENCH_CACHE", BASE / "artifacts" / "page_cache")).expanduser()
TOP_K = 5
PASSAGE_LEN = 2200
PASSAGE_OVERLAP = 300
MAX_READ = 8000
PAGE_CACHE_VERSION = 2

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9%$./-]*")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def _tokens(text: str) -> list[str]:
    return [x.lower() for x in _TOKEN_RE.findall(text)]


def _clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _company_name(doc_name: str) -> str:
    base = doc_name.split("_")[0]
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base)
    if len(s) > 2 and s == s.upper() and not any(ch.isdigit() for ch in s):
        return " ".join(w.capitalize() for w in s.split())
    return s


def document_metadata(doc_name: str) -> dict[str, Any]:
    year_match = _YEAR_RE.search(doc_name)
    upper = doc_name.upper()
    if "10K" in upper or "ANNUALREPORT" in upper:
        filing_type = "10-K"
    elif "10Q" in upper:
        filing_type = "10-Q"
    elif "8K" in upper:
        filing_type = "8-K"
    elif "EARNINGS" in upper:
        filing_type = "earnings"
    else:
        filing_type = "other"
    return {
        "document_id": doc_name,
        "company": _company_name(doc_name),
        "year": int(year_match.group(1)) if year_match else None,
        "filing_type": filing_type,
    }


def _section_hint(text: str) -> str:
    """Conservative section hint; unknown is preferable to a false section."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in lines[:25]:
        low = line.lower()
        if len(line) < 140 and (
            re.search(r"\bitem\s+[0-9]{1,2}[a-z]?\b", low)
            or any(k in low for k in (
                "consolidated statement", "balance sheet", "cash flows",
                "financial statements", "management's discussion",
                "risk factors", "legal proceedings", "business overview",
            ))
        ):
            return line[:140]
    return ""


def _extract_pages(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        return [_clean_text(page.extract_text() or "") for page in reader.pages]
    except Exception:
        return []


def load_pages(doc_name: str, use_cache: bool = True, use_pdf_pages: bool | None = None) -> list[dict[str, Any]]:
    """Load page text.

    The default fast path uses the durable extracted text cache. Set
    FINANCEBENCH_PDF_PAGES=1 to pay the pypdf cost and preserve true PDF page
    boundaries; this is useful for a provenance-quality indexing pass, but is
    not required for the first agent-training smoke tests.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"{doc_name}.json"
    if use_pdf_pages is None:
        use_pdf_pages = os.environ.get("FINANCEBENCH_PDF_PAGES", "1") == "1"
    pdf_path = FILINGS / f"{doc_name}.pdf"
    txt_path = TEXTDIR / f"{doc_name}.txt"
    source_mode = "pdf" if use_pdf_pages and pdf_path.exists() else "text"
    if use_cache and cache_path.exists():
        try:
            value = json.loads(cache_path.read_text())
            if isinstance(value, dict) and value.get("version") == PAGE_CACHE_VERSION and value.get("source") == source_mode and isinstance(value.get("rows"), list) and value["rows"]:
                return value["rows"]
        except Exception:
            pass
    pages = _extract_pages(pdf_path) if use_pdf_pages and pdf_path.exists() else []
    if not pages and txt_path.exists():
        pages = [_clean_text(txt_path.read_text(errors="ignore"))]
    if not pages and pdf_path.exists():
        pages = _extract_pages(pdf_path)
    if not pages:
        pages = [""]
    rows = []
    for i, text in enumerate(pages, start=1):
        rows.append({
            **document_metadata(doc_name),
            "page": i,
            "text": text,
            "section": _section_hint(text),
        })
    if not any(row.get("text", "").strip() for row in rows):
        raise FileNotFoundError(f"Missing or empty filing text for {doc_name}; checked {pdf_path} and {txt_path}")
    cache_path.write_text(json.dumps({"version": PAGE_CACHE_VERSION, "source": source_mode, "rows": rows}, ensure_ascii=False))
    return rows


def _table_candidate(page: dict[str, Any]) -> bool:
    text = page["text"]
    low = text.lower()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    content_lines = [x for x in lines if not re.fullmatch(r"\d{1,4}", x) and x.lower() != "table of contents"]
    head = "\n".join(x.lower() for x in content_lines[:18])
    if re.search(r"state(\s*ment)?s?\s+(of|and)\s+cash\s*flows?|cash\s*flows?\s*statement", head):
        return True
    if re.search(r"state(\s*ment)?s?\s+of\s+operations|statement\s+of\s+income|income\s*statement", head):
        return True
    if re.search(r"balance\s*sheets?", head):
        return True
    if re.search(r"comprehensive\s+income", head):
        return True
    if re.search(r"stockholders'?\s*equity", head):
        return True
    numeric = sum(bool(re.search(r"\d", x)) for x in content_lines)
    return len(content_lines) >= 10 and numeric / max(len(content_lines), 1) >= 0.45


def _units(text: str) -> str:
    m = re.search(r"\(([^\n]{0,100}(?:million|thousand|percent|per share|\$)[^\n]{0,100})\)", text, re.I)
    return m.group(1).strip() if m else ""


def _statement_kind(text: str) -> str:
    """Conservative mechanical statement label.

    Most pages are intentionally labeled ``other``. A statement-specific
    label is used only when a canonical title appears in the top title block;
    later note prose must not poison retrieval. This is deliberately simple,
    not an accounting classifier; the agent can inspect ambiguous candidates.
    """
    low = (text or "").lower()
    lines = [x.strip() for x in low.splitlines() if x.strip()]
    content_lines = [x for x in lines if not re.fullmatch(r"\d{1,4}", x) and x.lower() != "table of contents"]
    # Amazon's statement title is split across two extracted lines, so the 18-line
    # top block is still small enough to avoid prose from the note body. For
    # pages that contain two statement blocks, the first title is the governing
    # statement kind because later blocks are subordinate summary rows.
    head = " ".join(" ".join(x.lower() for x in content_lines[:18]).split())
    cash_matches = list(re.finditer(r"state(\s*ment)?s?\s+(of|and)\s+cash\s*flows?|cash\s*flows?\s*statement", head))
    operations_matches = list(re.finditer(r"state(\s*ment)?s?\s+of\s+operations|statement\s+of\s+income|income\s*statement", head))
    if cash_matches and (not operations_matches or cash_matches[0].start() < operations_matches[0].start()):
        return "cashflow"
    if operations_matches:
        return "operations"
    if re.search(r"balance\s*sheets?", head):
        return "balance"
    if re.search(r"comprehensive\s+income", head):
        return "comprehensive"
    if re.search(r"stockholders'?\s*equity", head):
        return "equity"
    return "other"


def _table_intent_bonus(query: str, item: dict[str, Any]) -> float:
    """Prefer the requested financial statement without answer-linked metadata."""
    q = query.lower()
    kind = item.get("statement_kind", "other")
    bonus = 0.0
    if any(x in q for x in ("income statement", "statements of operations", "net income attributable", "net income to shareholders")):
        bonus += 8.0 if kind == "operations" else (-2.0 if kind == "cashflow" else 0.0)
    if any(x in q for x in ("cash flow", "cash from operations", "cash provided by operating")):
        bonus += 8.0 if kind == "cashflow" else 0.0
    if any(x in q for x in ("balance sheet", "current liabilities", "current assets", "total liabilities")):
        bonus += 8.0 if kind == "balance" else 0.0
    if "comprehensive income" in q:
        bonus += 8.0 if kind == "comprehensive" else 0.0
    if "stockholders" in q or "shareholders' equity" in q:
        bonus += 8.0 if kind == "equity" else 0.0
    return bonus


class StructuredIndex:
    """In-memory indexes over mechanical page/prose/table records."""

    def __init__(self, pages: list[dict[str, Any]], passages: list[dict[str, Any]], tables: list[dict[str, Any]]):
        self.pages = pages
        self.passages = passages
        self.tables = tables
        self.pages_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.passages_by_id: dict[str, dict[str, Any]] = {}
        self.tables_by_id: dict[str, dict[str, Any]] = {}
        for page in pages:
            self.pages_by_doc[page["document_id"]].append(page)
        for item in passages:
            self.passages_by_id[item["passage_id"]] = item
        for item in tables:
            self.tables_by_id[item["table_id"]] = item
        self._prepare_bm25(passages, "prose")
        self._prepare_bm25(tables, "table")

    def _prepare_bm25(self, items: list[dict[str, Any]], namespace: str) -> None:
        counters = []
        df = Counter()
        for item in items:
            toks = _tokens(item.get("text", ""))
            c = Counter(toks)
            counters.append(c)
            for token in c:
                df[token] += 1
            item["_length"] = len(toks)
        setattr(self, f"_{namespace}_counters", counters)
        setattr(self, f"_{namespace}_df", df)
        setattr(self, f"_{namespace}_avgdl", sum(x.get("_length", 0) for x in items) / max(1, len(items)))

    def _allowed(self, item: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key in ("company", "document_id", "filing_type"):
            wanted = str(filters.get(key) or "").strip().lower()
            if wanted and str(item.get(key, "")).lower() != wanted:
                return False
        year = filters.get("year")
        if year not in (None, "", -1, 0) and item.get("year") != int(year):
            return False
        section = str(filters.get("section") or "").strip().lower()
        if section and section not in str(item.get("section", "")).lower():
            return False
        return True

    def _score(self, query: str, items: list[dict[str, Any]], counters: list[Counter], df: Counter, avgdl: float, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        q = _tokens(query)
        if not q:
            return []
        n = len(items)
        scored = []
        for i, item in enumerate(items):
            if not self._allowed(item, filters):
                continue
            length = counters[i]
            dl = max(1, item.get("_length", 0))
            score = 0.0
            for token in q:
                if token not in df:
                    continue
                idf = math.log((n - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
                freq = length.get(token, 0)
                score += idf * (freq * 2.5) / (freq + 1.5 * (0.25 + 0.75 * dl / max(avgdl, 1.0)))
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**item, "score": round(score, 5)} for score, item in scored[:top_k]]

    def search_prose(self, queries: list[str], filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        hits: dict[str, dict[str, Any]] = {}
        for query in queries:
            for item in self._score(query, self.passages, self._prose_counters, self._prose_df, self._prose_avgdl, filters, top_k):
                key = item["passage_id"]
                if key not in hits or item["score"] > hits[key]["score"]:
                    hits[key] = item
        return sorted(hits.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    def search_tables(self, queries: list[str], filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        hits: dict[str, dict[str, Any]] = {}
        for query in queries:
            for item in self._score(query, self.tables, self._table_counters, self._table_df, self._table_avgdl, filters, max(top_k, 12)):
                item = {**item, "score": round(item["score"] + _table_intent_bonus(query, item), 5)}
                key = item["table_id"]
                if key not in hits or item["score"] > hits[key]["score"]:
                    hits[key] = item
        return sorted(hits.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    def grep_document(self, document_id: str, patterns: list[str], page_start: int = -1, page_end: int = -1, context_lines: int = 2) -> list[dict[str, Any]]:
        results = []
        wanted = [p for p in patterns if p]
        for page in self.pages_by_doc.get(document_id, []):
            if page_start > 0 and page["page"] < page_start:
                continue
            if page_end > 0 and page["page"] > page_end:
                continue
            lines = page["text"].splitlines() or [page["text"]]
            for line_no, line in enumerate(lines):
                try:
                    matched = any(re.search(pattern, line, re.I) for pattern in wanted)
                except re.error:
                    matched = any(p.lower() in line.lower() for p in wanted)
                if matched:
                    lo = max(0, line_no - max(0, context_lines))
                    hi = min(len(lines), line_no + context_lines + 1)
                    results.append({
                        "document_id": document_id,
                        "page": page["page"],
                        "line": line_no + 1,
                        "text": "\n".join(lines[lo:hi]),
                        "provenance": f"{document_id}:page={page['page']}:line={line_no + 1}",
                    })
        return results[:30]

    def read(self, document_id: str, page: int = -1, start: int = 0, end: int = MAX_READ, passage_id: str = "") -> dict[str, Any]:
        if passage_id and passage_id in self.passages_by_id:
            item = self.passages_by_id[passage_id]
            out = {k: v for k, v in item.items() if not k.startswith("_")}
            out.update({"start": item.get("start", 0), "end": item.get("end", len(item.get("text", ""))), "has_more": False, "next_start": None})
            return out
        pages = self.pages_by_doc.get(document_id, [])
        selected = [x for x in pages if page < 1 or x["page"] == page]
        if not selected:
            return {"error": f"Unknown document/page: {document_id}/{page}"}
        text = "\n\n".join(x["text"] for x in selected)
        start = max(0, int(start))
        requested_end = max(start, int(end))
        bounded_end = min(len(text), requested_end, start + MAX_READ)
        return {
            "document_id": document_id,
            "page_start": selected[0]["page"],
            "page_end": selected[-1]["page"],
            "start": start,
            "end": bounded_end,
            "total_chars": len(text),
            "has_more": bounded_end < len(text),
            "next_start": bounded_end if bounded_end < len(text) else None,
            "text": text[start:bounded_end],
            "provenance": f"{document_id}:pages={selected[0]['page']}-{selected[-1]['page']}:chars={start}-{bounded_end}",
        }

    def read_table(self, table_id: str, include_neighbors: bool = True, start: int = 0, end: int = MAX_READ) -> dict[str, Any]:
        item = self.tables_by_id.get(table_id)
        if not item:
            return {"error": f"Unknown table_id: {table_id}"}
        out = {k: v for k, v in item.items() if not k.startswith("_")}
        text = str(out.get("text", ""))
        start = max(0, int(start))
        bounded_end = min(len(text), max(start, int(end)), start + MAX_READ)
        out["start"] = start
        out["end"] = bounded_end
        out["total_chars"] = len(text)
        out["has_more"] = bounded_end < len(text)
        out["next_start"] = bounded_end if bounded_end < len(text) else None
        out["text"] = text[start:bounded_end]
        if include_neighbors:
            page = item["page"]
            doc = item["document_id"]
            pages = self.pages_by_doc.get(doc, [])
            neighbors = [x["text"] for x in pages if abs(x["page"] - page) <= 1]
            out["neighbor_context"] = "\n\n".join(neighbors)[:MAX_READ]
        return out


def build_index(max_docs: int | None = None, doc_names: list[str] | None = None, use_cache: bool = True) -> StructuredIndex:
    files = sorted(x.stem for x in FILINGS.glob("*.pdf"))
    if doc_names is not None:
        wanted = set(doc_names)
        files = [x for x in files if x in wanted]
    if max_docs:
        files = files[:max_docs]
    pages = []
    passages = []
    tables = []
    for doc_name in files:
        doc_pages = load_pages(doc_name, use_cache=use_cache)
        pages.extend(doc_pages)
        for page in doc_pages:
            text = page["text"]
            if len(text) < 40:
                continue
            step = max(1, PASSAGE_LEN - PASSAGE_OVERLAP)
            for i, start in enumerate(range(0, len(text), step)):
                chunk = text[start:start + PASSAGE_LEN]
                if len(chunk) < 40:
                    continue
                passages.append({
                    **{k: page[k] for k in ("document_id", "company", "year", "filing_type", "page", "section")},
                    "passage_id": f"p:{doc_name}:page:{page['page']}:chunk:{i}",
                    "start": start,
                    "end": start + len(chunk),
                    "text": chunk,
                })
            if _table_candidate(page):
                lines = [x.strip() for x in text.splitlines() if x.strip()]
                title = next((x for x in lines[:40] if any(k in x.lower() for k in ("statements of operations", "income statement", "balance sheet", "cash flows", "comprehensive income", "stockholders' equity"))), "")
                tables.append({
                    **{k: page[k] for k in ("document_id", "company", "year", "filing_type", "page", "section")},
                    "table_id": f"t:{doc_name}:page:{page['page']}",
                    "title": title[:180],
                    "statement_kind": _statement_kind(text),
                    "units": _units(text),
                    "text": text,
                })
    return StructuredIndex(pages, passages, tables)


# Compatibility helpers for standalone callers.
_ANSWER_MARKER_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer\s*(?:\*\*)?\s*:\s*(?:\*\*)?"
)
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\$?\d[\d,]*(?:\.\d+)?(?:%|[KMB](?:illion)?|x)?(?![A-Za-z0-9])", re.I)
_REWARD_STOP_WORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as", "is", "was", "were", "by", "from", "that", "this", "their", "into", "had", "have", "has", "does", "did", "do", "not", "no", "yes", "than", "based", "during", "year", "fy", "approximately", "about", "company", "level", "answer", "there", "are", "according", "reported", "data", "ended", "respectively"}
_REWARD_TOKEN_ALIASES = {"operations": "operation", "operating": "operation", "operated": "operation", "cashflows": "cashflow", "cash-flow": "cashflow", "revenues": "revenue", "expenses": "expense", "liabilities": "liability", "assets": "asset", "investments": "investment", "bonds": "bond", "resorts": "resort", "margins": "margin", "ratios": "ratio", "companies": "company", "years": "year", "quarters": "quarter"}

def _norm(s: str) -> str:
    return (s or "").lower().strip().replace("$", "").replace(",", "").replace(" ", "")

def extract_answer_text(text: str) -> str:
    raw = str(text or "").strip()
    matches = list(_ANSWER_MARKER_RE.finditer(raw))
    if matches:
        answer = raw[matches[-1].end():].strip()
        if answer:
            return answer
    return raw

def _answer_tokens(text: str) -> set[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if token in _REWARD_STOP_WORDS or len(token) < 2:
            continue
        token = _REWARD_TOKEN_ALIASES.get(token, token)
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return set(tokens)

def _decision(text: str) -> str | None:
    value = extract_answer_text(text)
    first = value[:500]
    # Do not treat a table heading such as "increase/(decrease)" as the
    # answer's direction; the surrounding narrative may state the actual
    # conclusion later in the same response.
    first = re.sub(r"\bincrease\s*/\s*\(\s*decrease\s*\)", "decrease", first, flags=re.I)
    first = re.sub(r"\bdecrease\s*/\s*\(\s*increase\s*\)", "increase", first, flags=re.I)
    match = re.match(r"\s*(?:the\s+answer\s+is\s+)?(yes|no|true|false)\b", first, re.I)
    if match:
        return {"true": "yes", "false": "no"}.get(match.group(1).lower(), match.group(1).lower())
    match = re.search(r"\b(?:answer|conclusion)\s*(?:is|:)\s*(yes|no|true|false)\b", first, re.I)
    if match:
        return {"true": "yes", "false": "no"}.get(match.group(1).lower(), match.group(1).lower())
    first_sentence = first.lower()
    stripped = first_sentence.strip()
    if stripped in {"increase", "increased", "up", "positive", "yes"}:
        return "yes"
    if stripped in {"decrease", "decreased", "decline", "declined", "down", "negative", "no"}:
        return "no"
    if re.match(r"\s*(?:lower|higher|decrease|increase|decline|drop|change)\b", first_sentence):
        return None
    negative = list(re.finditer(r"\b(?:not|negative|decline|declined|decrease|decreased|fell|fall|drop|dropped|lower|down|no\s+material)\b", first_sentence))
    positive = list(re.finditer(r"\b(?:positive|increase|increased|grew|improve|improved|largest|yes|up)\b", first_sentence))
    if positive and (not negative or positive[0].start() < negative[0].start()):
        return "yes"
    if negative:
        return "no"
    return None

def _number_mentions(text: str) -> list[tuple[float, bool, str]]:
    mentions = []
    # Normalize common PDF/typographic minus signs before parsing amounts.
    raw_text = (text or "").replace("−", "-").replace("–", "-").replace("—", "-")
    lower = raw_text.lower()
    for match in _NUM_RE.finditer(raw_text):
        token = match.group(0)
        # "3M" is usually the company name 3M, not a three-million amount;
        # retain monetary forms such as "$3M" and explicit "3 million".
        if token == "3M":
            continue
        try:
            value = float(token.replace("$", "").replace(",", "").replace("%", "").rstrip("KkMmBbXx"))
        except ValueError:
            continue
        if _is_year(value):
            mentions.append((value, False, token))
            continue
        context = lower[max(0, match.start() - 12):match.start()] + " " + lower[match.end():match.end() + 12]
        percent = token.endswith("%") or bool(re.search(r"\bpercent\b", context)) or "%" in lower[match.end():match.end() + 2]
        scale = 1.0
        if token.lower().endswith("b") or re.search(r"\b(?:billion|bn)\b", context):
            scale = 1e9
        elif token.lower().endswith("m") or re.search(r"\b(?:million|mm)\b", context):
            scale = 1e6
        elif token.lower().endswith("k") or re.search(r"\bthousand\b", context):
            scale = 1e3
        mentions.append((value / 100.0 if percent else value * scale, percent, token))
    return mentions

def _is_year(value: float) -> bool:
    return 1900 <= abs(value) <= 2100 and abs(value - round(value)) < 1e-9

def _numbers_match(gold: str, candidate: str) -> tuple[bool, int]:
    gold_mentions = [(v, pct) for v, pct, _ in _number_mentions(gold) if not _is_year(v)]
    candidate_mentions = [(v, pct) for v, pct, _ in _number_mentions(candidate) if not _is_year(v)]
    if not gold_mentions:
        return False, 0
    matched = 0
    for gold_value, gold_pct in gold_mentions:
        found = False
        for candidate_value, candidate_pct in candidate_mentions:
            # Scale is normalized only from explicit currency/scale markers;
            # do not speculate by dividing arbitrary numbers by powers of ten.
            comparable_value = candidate_value
            if gold_value == 0:
                close = abs(comparable_value) <= (0.005 if gold_pct else 0.01)
            elif gold_pct or candidate_pct:
                # Preserve the sign: -4.2% and +4.2% are contradictions.
                close = abs(gold_value - comparable_value) <= max(0.01, abs(gold_value) * 0.005)
            else:
                close = abs(gold_value - comparable_value) <= max(0.01, abs(gold_value) * 0.005)
            if close:
                found = True
                break
        matched += int(found)
    return matched == len(gold_mentions), matched

def _text_overlap(gold: str, candidate: str) -> tuple[float, float, int]:
    gold_tokens = _answer_tokens(gold)
    candidate_tokens = _answer_tokens(candidate)
    if not gold_tokens or not candidate_tokens:
        return 0.0, 0.0, 0
    overlap = len(gold_tokens.intersection(candidate_tokens))
    return overlap / len(gold_tokens), overlap / min(len(gold_tokens), len(candidate_tokens)), overlap

def _explicit_contradiction(gold: str, candidate: str) -> bool:
    g = extract_answer_text(gold).lower()
    c = extract_answer_text(candidate).lower()
    positive_words = ("positive", "increased", "grew", "up", "improved")
    negative_words = ("negative", "declined", "decreased", "down", "fell")
    gold_positive = any(re.search(r"\b" + re.escape(word) + r"\b", g) for word in positive_words)
    gold_negative = any(re.search(r"\b" + re.escape(word) + r"\b", g) for word in negative_words)
    if not (gold_positive and gold_negative):
        if gold_negative and any(re.search(r"\b" + re.escape(word) + r"\b", c) for word in positive_words) and not any(re.search(r"\b" + re.escape(word) + r"\b", c) for word in negative_words):
            return True
        if gold_positive and any(re.search(r"\b" + re.escape(word) + r"\b", c) for word in negative_words) and not any(re.search(r"\b" + re.escape(word) + r"\b", c) for word in positive_words):
            return True
    gold_int_negative = re.search(r"international.{0,40}\b(?:decline|declined|decreased|down)\b", g)
    gold_int_positive = re.search(r"international.{0,40}\b(?:increase|increased|grew|up)\b", g)
    cand_int_negative = re.search(r"international.{0,40}\b(?:decline|declined|decreased|down)\b", c)
    cand_int_positive = re.search(r"international.{0,40}\b(?:increase|increased|grew|up)\b", c)
    if gold_int_negative and cand_int_positive and not cand_int_negative:
        return True
    if gold_int_positive and cand_int_negative and not cand_int_positive:
        return True
    return False

def score_answer(ground_truth: str, model_answer_text: str) -> tuple[float, dict[str, float]]:
    candidate = extract_answer_text(model_answer_text)
    if not candidate.strip():
        return 0.0, {"quality": 0.0, "conclusion": 0.0, "details": 0.0, "answer_nonempty": 0.0}
    gold = str(ground_truth or "").strip()
    if not gold:
        return 0.0, {"quality": 0.0, "conclusion": 0.0, "details": 0.0, "answer_nonempty": 1.0}
    if _norm(gold) == _norm(candidate):
        return 1.0, {"quality": 1.0, "conclusion": 1.0, "details": 1.0, "answer_nonempty": 1.0}
    gold_decision = _decision(gold)
    candidate_decision = _decision(candidate)
    number_match, number_count = _numbers_match(gold, candidate)
    recall, short_coverage, overlap = _text_overlap(gold, candidate)
    has_gold_numbers = bool([x for x, _, _ in _number_mentions(gold) if not _is_year(x)])
    candidate_has_numbers = bool([x for x, _, _ in _number_mentions(candidate) if not _is_year(x)])
    if _explicit_contradiction(gold, candidate):
        return 0.0, {"quality": 0.0, "conclusion": 0.0, "details": 0.0, "answer_nonempty": 1.0}
    numeric_conflict = has_gold_numbers and candidate_has_numbers and not number_match and number_count == 0
    conclusion = 0.0
    details = 0.0
    if gold_decision is not None:
        if candidate_decision is not None and candidate_decision != gold_decision:
            return 0.0, {"quality": 0.0, "conclusion": 0.0, "details": 0.0, "answer_nonempty": 1.0}
        if candidate_decision is None and not number_match:
            return 0.0, {"quality": 0.0, "conclusion": 0.0, "details": 0.0, "answer_nonempty": 1.0}
        conclusion = 1.0 if candidate_decision == gold_decision else 0.9
        if has_gold_numbers:
            details = 1.0 if number_match else (0.0 if numeric_conflict and len(_answer_tokens(candidate)) <= 8 else (0.9 if number_count > 0 or recall >= 0.2 else min(0.75, max(0.25, recall))))
        elif overlap >= 2 or len(_answer_tokens(gold)) <= 6:
            details = 1.0
        else:
            details = 0.9
    else:
        if number_match:
            details = 1.0
            conclusion = 1.0 if overlap >= 1 or number_count else 0.9
        elif overlap >= 2 and (short_coverage >= 0.5 or recall >= 0.35):
            conclusion = 1.0
            details = max(0.9, short_coverage)
        elif recall >= 0.6:
            conclusion = 1.0
            details = 0.9
        elif overlap >= 1 and "operation" in _answer_tokens(candidate) and "operation" in _answer_tokens(gold):
            conclusion = 1.0
            details = 0.9
    quality = 0.5 * conclusion + 0.5 * details
    if quality < 0.9 and overlap >= 2 and short_coverage >= 0.67 and not has_gold_numbers:
        quality = 0.9
    quality = float(min(1.0, quality))
    return quality, {"quality": quality, "conclusion": float(conclusion), "details": float(details), "answer_nonempty": 1.0}

def _trace_evidence_text(trace: list[dict[str, Any]] | None) -> tuple[str, str]:
    """Return (strong, weak) evidence text encountered by the agent.

    Reads, table reads, and grep are stronger provenance than ranked search
    snippets. Search results remain usable evidence, but are down-weighted so
    search-only answers do not receive the same grounding credit as answers
    backed by a bounded document read.
    """
    call_names: dict[str, str] = {}
    for item in trace or []:
        if item.get("role") != "assistant":
            continue
        for call in item.get("tool_calls") or []:
            if isinstance(call, dict):
                call_names[str(call.get("call_id", call.get("id", "")))] = str(call.get("name", ""))
                fn = call.get("function") or {}
                if isinstance(fn, dict):
                    call_names[str(call.get("id", ""))] = str(fn.get("name", ""))

    def flatten(value: Any) -> str:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return value
            if parsed != value:
                return flatten(parsed)
            return value
        if isinstance(value, dict):
            parts = []
            for key, child in value.items():
                if key in {"text", "snippet", "content", "neighbor_context", "title", "section", "document_id", "provenance"}:
                    parts.append(flatten(child))
                elif key in {"prose_hits", "table_hits", "matches", "read", "table"}:
                    parts.append(flatten(child))
            return " ".join(x for x in parts if x)
        if isinstance(value, list):
            return " ".join(flatten(x) for x in value)
        return ""

    strong_parts: list[str] = []
    weak_parts: list[str] = []
    for item in trace or []:
        if item.get("role") != "tool":
            continue
        text = flatten(item.get("content", ""))
        if not text:
            continue
        name = call_names.get(str(item.get("call_id", "")), str(item.get("name", "")))
        if not name:
            # Tinker tool wrappers can lose call metadata while preserving the
            # structured result; infer the provenance class from stable result keys.
            if '"matches"' in text or '"backend_used"' in text:
                name = "grep_document"
            elif '"table"' in text or '"table_id"' in text:
                name = "read_table"
            elif '"prose_hits"' in text or '"table_hits"' in text:
                name = "bm25_search"
        if name in {"read", "read_table", "grep_document"}:
            strong_parts.append(text)
        elif name in {"bm25_search", "search_tables"}:
            weak_parts.append(text)
    return "\n".join(strong_parts), "\n".join(weak_parts)


def _number_supported(mention: tuple[float, bool, str], evidence_text: str) -> bool:
    expected, expected_pct, _ = mention
    for observed, observed_pct, _ in _number_mentions(evidence_text):
        # Compare explicitly normalized units; do not infer scale by division.
        value = observed
        if expected_pct or observed_pct:
            close = abs(expected - value) <= max(0.01, abs(expected) * 0.005)
        elif expected == 0:
            close = abs(value) <= 0.01
        else:
            close = abs(expected - value) <= max(0.01, abs(expected) * 0.005)
        if close:
            return True
    return False


def score_evidence(ground_truth: str, model_answer_text: str, trace: list[dict[str, Any]] | None) -> tuple[float, dict[str, float]]:
    """Score whether answer-bearing claims were actually encountered in tools.

    This is deliberately conservative: required gold numbers must appear in
    both the answer and the trajectory evidence. Reads/grep/table reads get
    full credit; search snippets alone receive a bounded discount.
    """
    candidate = extract_answer_text(model_answer_text)
    if not candidate.strip():
        return 0.0, {"evidence": 0.0, "strong_source": 0.0, "gold_claim_support": 0.0, "candidate_claim_support": 0.0}
    strong, weak = _trace_evidence_text(trace)
    all_evidence = "\n".join(x for x in (strong, weak) if x)
    if not all_evidence:
        return 0.0, {"evidence": 0.0, "strong_source": 0.0, "gold_claim_support": 0.0, "candidate_claim_support": 0.0}

    gold_numbers = [x for x in _number_mentions(ground_truth) if not _is_year(x[0])]
    candidate_numbers = [x for x in _number_mentions(candidate) if not _is_year(x[0])]
    gold_in_answer = (sum(_number_supported(x, candidate) for x in gold_numbers) / len(gold_numbers)) if gold_numbers else 1.0
    gold_in_trace = (sum(_number_supported(x, all_evidence) for x in gold_numbers) / len(gold_numbers)) if gold_numbers else 1.0
    candidate_in_trace = (sum(_number_supported(x, all_evidence) for x in candidate_numbers) / len(candidate_numbers)) if candidate_numbers else 1.0

    if gold_numbers:
        claim_support = min(gold_in_answer, gold_in_trace)
        # Unsupported additional numeric claims should reduce grounding, but
        # one derived/rounded number should not erase otherwise good support.
        claim_support *= max(0.5, candidate_in_trace)
    else:
        gold_tokens = _answer_tokens(ground_truth)
        candidate_tokens = _answer_tokens(candidate)
        evidence_tokens = _answer_tokens(all_evidence)
        relevant_gold = {x for x in gold_tokens if len(x) > 2}
        relevant_candidate = {x for x in candidate_tokens if len(x) > 2}
        gold_token_support = (len(relevant_gold & evidence_tokens) / len(relevant_gold)) if relevant_gold else 1.0
        candidate_token_support = (len(relevant_candidate & evidence_tokens) / len(relevant_candidate)) if relevant_candidate else 1.0
        claim_support = min(1.0, max(gold_token_support, candidate_token_support))

    strong_source = 1.0 if strong else 0.75 if weak else 0.0
    evidence = float(max(0.0, min(1.0, claim_support * strong_source)))
    return evidence, {
        "evidence": evidence,
        "strong_source": float(bool(strong)),
        "gold_claim_support": float(gold_in_answer * gold_in_trace),
        "candidate_claim_support": float(candidate_in_trace),
    }


def grounded_reward(answer_quality: float, evidence_quality: float) -> float:
    """Keep answer correctness primary while penalizing unsupported claims."""
    return float(max(0.0, min(1.0, answer_quality * (0.5 + 0.5 * evidence_quality))))


def reward(ground_truth: str, model_answer_text: str) -> float:
    return score_answer(ground_truth, model_answer_text)[0]


class _SafeMath(ast.NodeVisitor):
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Constant, ast.Mod)
    def visit(self, node):
        if not isinstance(node, self.allowed):
            raise ValueError("only basic arithmetic is allowed")
        return super().visit(node)


def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    _SafeMath().visit(tree)
    value = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("expression did not produce a finite number")
    return str(value)
