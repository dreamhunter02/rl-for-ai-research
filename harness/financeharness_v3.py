#!/usr/bin/env python3
"""Financeharness v3 — Uniform stateful retrieval harness core across all 4 benchmarks:
  - FinanceBench, FinRetrieval, FinAgentBench, FinSearchComp

Design principles (RL4LLM Financeharness + MintHarness-inspired upgrades):
  1. Discovery is not evidence: only inspection tools create citable receipts (`r1, r2, ...`).
  2. Automatic Receipt Ledger + Working Memory packed into a bounded prompt (no memory tool).
  3. MintHarness-style deterministic structured metadata header on every packed receipt
     (`[r3 | ENTITY | SOURCE | PERIOD | UNIT]`) while preserving verbatim provenance.
  4. Sandboxed `run_python` execution tool (2s timeout, no FS/net, math/statistics/decimal).
  5. Dynamic entity/ticker/acronym alias normalization (e.g., JnJ/J&J <-> JOHNSON_JOHNSON)
     without abandoning fast, zero-vector-DB BM25 lexical precision.
  6. Unified action-budget cues, duplicate-call loop breaker, and negative-evidence abstention.
"""
from __future__ import annotations

import ast
import difflib
import io
import json
import math
import subprocess
import sys
import re
import statistics
from contextlib import redirect_stdout
from decimal import Decimal
from typing import Any


# ---------------- 1. Dynamic Entity / Acronym / Alias Normalization ----------------

_STATIC_ALIASES: dict[str, set[str]] = {
    "johnson & johnson": {"jnj", "j&j", "johnson", "johnson_johnson", "johnson and johnson"},
    "3m": {"mmm", "3m", "3m company"},
    "meta": {"meta", "fb", "facebook", "meta platforms"},
    "alphabet": {"googl", "goog", "google", "alphabet"},
    "microsoft": {"msft", "microsoft"},
    "apple": {"aapl", "apple"},
    "nvidia": {"nvda", "nvidia"},
    "jpmorgan": {"jpm", "jpmorgan", "jp morgan", "jpmorgan chase"},
    "bank of america": {"bac", "bofa", "bank of america"},
    "procter & gamble": {"pg", "p&g", "procter", "procter gamble", "procter_gamble"},
    "verizon": {"vz", "verizon"},
    "ulta beauty": {"ulta", "ulta beauty", "ultabeauty"},
    "general mills": {"gis", "general mills", "generalmills"},
    "american express": {"axp", "amex", "american express"},
    "activision blizzard": {"atvi", "activision", "activision blizzard"},
    "foot locker": {"fl", "foot locker", "footlocker"},
    "amcor": {"amcr", "amcor"},
    "pfizer": {"pfe", "pfizer"},
    "adobe": {"adbe", "adobe"},
    "aes": {"aes", "aes corporation"},
}

_ALIAS_LOOKUP: dict[str, set[str]] = {}
for _canon, _forms in _STATIC_ALIASES.items():
    _all = {_canon} | _forms
    for _f in _all:
        _ALIAS_LOOKUP[_f.lower()] = _all


def normalize_entity_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def expand_entity_aliases(query_or_company: str, corpus_names: list[str] | None = None) -> list[str]:
    """Expand a company name/ticker/acronym into canonical & corpus-matched forms."""
    raw = str(query_or_company or "").strip()
    if not raw:
        return []
    low = raw.lower().strip()
    norm = normalize_entity_token(raw)
    expanded: set[str] = {raw, low, norm}

    # Static alias lookup (exact or token hit)
    if low in _ALIAS_LOOKUP:
        expanded |= _ALIAS_LOOKUP[low]
    if norm in _ALIAS_LOOKUP:
        expanded |= _ALIAS_LOOKUP[norm]
    for tok in norm.split():
        if tok in _ALIAS_LOOKUP and len(tok) >= 2:
            expanded |= _ALIAS_LOOKUP[tok]

    # Dynamic acronym & fuzzy match against actual corpus document/company names
    if corpus_names:
        norm_no_space = norm.replace(" ", "")
        for cname in corpus_names:
            cnorm = normalize_entity_token(cname)
            words = [w for w in cnorm.split() if w and not w.isdigit() and w not in {"10k", "10q", "8k", "inc", "corp", "co", "ltd", "plc"}]
            if not words:
                continue
            acronym = "".join(w[0] for w in words)
            # e.g. "johnson johnson" -> "jj", "jnj"
            acronyms = {acronym}
            if len(words) == 2:
                acronyms.add(f"{words[0][0]}n{words[1][0]}")
                acronyms.add(f"{words[0][0]}&{words[1][0]}")
            if norm_no_space in acronyms or low in acronyms:
                expanded.add(cname)
                expanded.add(" ".join(words))
            elif difflib.SequenceMatcher(None, norm, " ".join(words)).ratio() >= 0.78:
                expanded.add(cname)
                expanded.add(" ".join(words))

    return sorted(x for x in expanded if x)


# ---------------- 2. Sandboxed `run_python` Execution Tool ----------------

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_FORBIDDEN_AST_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)


def _python_worker(code: str, out_q: multiprocessing.Queue) -> None:
    try:
        tree = ast.parse(code, mode="exec")
        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN_AST_NODES):
                out_q.put({"error": f"Forbidden syntax: {type(node).__name__} (math, statistics, Decimal are already pre-imported)"})
                return
            if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
                out_q.put({"error": "Dunder attribute access is not allowed."})
                return

        env: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "math": math,
            "statistics": statistics,
            "Decimal": Decimal,
        }
        buf = io.StringIO()
        last_val = None
        # If the final statement is an expression, evaluate and return it as `result`
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            exec_body = ast.Module(body=tree.body[:-1], type_ignores=[])
            expr_node = ast.Expression(body=tree.body[-1].value)
            with redirect_stdout(buf):
                exec(compile(exec_body, "<run_python>", "exec"), env, env)
                last_val = eval(compile(expr_node, "<run_python>", "eval"), env, env)
        else:
            with redirect_stdout(buf):
                exec(compile(tree, "<run_python>", "exec"), env, env)

        stdout = buf.getvalue()
        if len(stdout) > 2000:
            stdout = stdout[:1950] + "...[truncated]"
        res: dict[str, Any] = {"ok": True}
        if stdout.strip():
            res["stdout"] = stdout.strip()
        if last_val is not None:
            if isinstance(last_val, float) and math.isfinite(last_val):
                res["result"] = round(last_val, 8)
            else:
                res["result"] = str(last_val)[:500]
        out_q.put(res)
    except Exception as exc:
        out_q.put({"error": f"{type(exc).__name__}: {str(exc)[:180]}"})


def run_python_sandbox(code: str, timeout_s: float = 2.0) -> str:
    """Execute pure Python financial calculations in a 2s isolated subprocess."""
    code = str(code or "").strip()
    if not code:
        return json.dumps({"error": "Empty code string."})
    if len(code) > 4000:
        return json.dumps({"error": "Code exceeds 4000-character limit."})
    # Validate AST before spawning subprocess
    try:
        tree = ast.parse(code, mode="exec")
        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN_AST_NODES):
                return json.dumps({"error": f"Forbidden syntax: {type(node).__name__} (math, statistics, Decimal are pre-imported)"})
            if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
                return json.dumps({"error": "Dunder attribute access is not allowed."})
    except SyntaxError as exc:
        return json.dumps({"error": f"SyntaxError: {str(exc)[:160]}"})

    runner = (
        "import ast, io, json, math, statistics\n"
        "from contextlib import redirect_stdout\n"
        "from decimal import Decimal\n"
        "code = json.loads(input())\n"
        "safe_b = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) "
        "for k in ['abs','all','any','bool','dict','enumerate','float','int','len','list','max','min','pow','print','range','round','set','sorted','str','sum','tuple','zip']}\n"
        "env = {'__builtins__': safe_b, 'math': math, 'statistics': statistics, 'Decimal': Decimal}\n"
        "tree = ast.parse(code, mode='exec')\n"
        "buf = io.StringIO(); last_val = None\n"
        "try:\n"
        "    if tree.body and isinstance(tree.body[-1], ast.Expr):\n"
        "        exec_body = ast.Module(body=tree.body[:-1], type_ignores=[])\n"
        "        expr_node = ast.Expression(body=tree.body[-1].value)\n"
        "        with redirect_stdout(buf):\n"
        "            exec(compile(exec_body, '<run_python>', 'exec'), env, env)\n"
        "            last_val = eval(compile(expr_node, '<run_python>', 'eval'), env, env)\n"
        "    else:\n"
        "        with redirect_stdout(buf):\n"
        "            exec(compile(tree, '<run_python>', 'exec'), env, env)\n"
        "    out = {'ok': True}\n"
        "    s = buf.getvalue().strip()\n"
        "    if s: out['stdout'] = s[:1950]\n"
        "    if last_val is not None:\n"
        "        out['result'] = round(last_val, 8) if isinstance(last_val, float) and math.isfinite(last_val) else str(last_val)[:500]\n"
        "    print(json.dumps(out, ensure_ascii=False))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': f'{type(e).__name__}: {str(e)[:180]}'}))\n"
    )
    try:
        cp = subprocess.run(
            [sys.executable, "-I", "-c", runner],
            input=json.dumps(code),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        if cp.stdout.strip():
            return cp.stdout.strip().splitlines()[-1]
        return json.dumps({"error": cp.stderr.strip()[:180] or "No output"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Execution timed out after {timeout_s}s"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:140]}"})


# ---------------- 3. MintHarness-Style Structured Receipt Header ----------------

_UNIT_RE = re.compile(
    r"\b(in millions|in thousands|in billions|\$ in millions|\$ in thousands|usd millions|usd thousands|eur millions|aud millions|percent|%)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(20\d{2}|FY\s*20\d{2}|Q[1-4]\s*20\d{2}|20\d{2}[QH][1-4])\b", re.IGNORECASE)


def format_receipt_header(receipt_id: str, meta: dict[str, Any], text: str) -> str:
    """Build a deterministic 1-line MintHarness-style structured metadata header."""
    parts = [receipt_id]
    entity = meta.get("entity") or meta.get("ticker") or meta.get("document_id") or meta.get("company_id")
    if entity:
        parts.append(str(entity))
    src = meta.get("source") or meta.get("tool") or meta.get("form")
    if src and str(src) != str(entity):
        parts.append(str(src))
    if meta.get("page"):
        parts.append(f"p.{meta['page']}")
    period = meta.get("period")
    if not period:
        yrs = sorted(set(m.upper().replace(" ", "") for m in _YEAR_RE.findall(text[:1500])))
        if yrs:
            period = ",".join(yrs[:4])
    if period:
        parts.append(f"period:{period}")
    unit = meta.get("unit")
    if not unit:
        um = _UNIT_RE.search(text[:1500])
        if um:
            unit = um.group(1)
    if unit:
        parts.append(f"unit:{unit}")
    return "[" + " | ".join(parts) + "]"


# ---------------- 4. Uniform ResearchMemory & Receipt Ledger ----------------

def _toks(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 1}


class UniformResearchMemory:
    """Automatic Receipt Ledger + Working Memory packer for ALL 4 finance benchmarks."""

    DISCOVERY_TOOLS = {
        "bm25_search", "search_tables", "discover_companies",
        "discover_company_series", "sec_list_filings", "web_search",
    }
    INSPECTION_TOOLS = {
        "read", "read_table", "grep_document",
        "get_company_fundamentals", "sec_company_facts", "sec_search_filing",
        "yfinance_history", "yfinance_info", "fetch_url",
    }

    def __init__(self, question: str, *, max_receipts_in_view: int = 8, max_state_chars: int = 9000):
        self.question = question
        self.q_toks = _toks(question)
        self.max_receipts_in_view = max_receipts_in_view
        self.max_state_chars = max_state_chars
        self.receipts: dict[str, dict[str, Any]] = {}
        self.recent_actions: list[str] = []
        self.failed_paths: list[str] = []
        self.calculations: list[str] = []
        self.seen_sigs: set[str] = set()
        self.repeated_calls: list[str] = []
        self._next_rid = 1

    def record_tool_result(
        self, name: str, args: dict[str, Any], content: str, meta: dict[str, Any] | None = None
    ) -> str | None:
        """Record a tool execution into Working Memory and (if inspection) Receipt Ledger.
        Returns receipt_id if a citable receipt was issued, else None.
        """
        sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        short_arg = json.dumps(args, ensure_ascii=False)[:100]
        is_dup = sig in self.seen_sigs
        self.seen_sigs.add(sig)
        if is_dup and name != "finish":
            self.repeated_calls.append(f"{name}({short_arg})")

        low = content.lower()
        is_empty = (
            '"error"' in low
            or '"records": []' in low
            or '"matches": []' in low
            or '"excerpts": []' in low
            or '"results": []' in low
            or '"series": []' in low
        )
        if is_empty:
            self.failed_paths.append(f"{name}({short_arg}) -> empty/error")
            self.recent_actions.append(f"{name}({short_arg}) [empty]")
            return None

        if name in ("calculate", "run_python"):
            self.calculations.append(f"{name}: {content[:220]}")
            self.recent_actions.append(f"{name} -> {content[:80]}")
            return None

        if name in self.INSPECTION_TOOLS:
            rid = f"r{self._next_rid}"
            self._next_rid += 1
            m = dict(meta or {})
            m.setdefault("tool", name)
            for k in ("ticker", "company_id", "document_id", "url", "accession_number", "page"):
                if k in args and k not in m:
                    m[k] = args[k]
            header = format_receipt_header(rid, m, content)
            self.receipts[rid] = {
                "receipt_id": rid,
                "header": header,
                "meta": m,
                "text": content,
                "toks": _toks(header + " " + content[:3000]),
                "order": self._next_rid,
            }
            self.recent_actions.append(f"{name}({short_arg}) -> {rid}")
            return rid

        self.recent_actions.append(f"{name}({short_arg})")
        return None

    def render_state_block(self, calls_used: int, max_calls: int) -> str:
        """Render bounded Working Memory + Top-K Question-Relevant Receipts."""
        lines = [f"=== HARNESS RESEARCH STATE (calls: {calls_used}/{max_calls}, receipts: {len(self.receipts)}) ==="]
        if self.recent_actions:
            lines.append("Recent actions: " + " | ".join(self.recent_actions[-6:]))
        if self.failed_paths:
            lines.append("Failed/empty paths (do not repeat): " + " | ".join(self.failed_paths[-4:]))
        if self.repeated_calls:
            lines.append("Repeated calls detected: " + " | ".join(self.repeated_calls[-3:]))
        if self.calculations:
            lines.append("Verified Python/calc outputs: " + " | ".join(self.calculations[-4:]))

        if self.receipts:
            scored = []
            for r in self.receipts.values():
                ov = len(self.q_toks & r["toks"])
                scored.append((ov, r["order"], r))
            scored.sort(key=lambda x: (-x[0], -x[1]))
            lines.append("Citable Evidence Ledger (top relevant receipts):")
            for _, _, r in scored[: self.max_receipts_in_view]:
                snippet = r["text"].replace("\n", " ")[:480]
                lines.append(f"  {r['header']} {snippet}")

        block = "\n".join(lines)
        if len(block) > self.max_state_chars:
            block = block[: self.max_state_chars - 20] + "\n...[state trimmed]"
        return block

    def pack(
        self,
        initial: list[dict[str, Any]],
        turns: list[list[dict[str, Any]]],
        *,
        calls_used: int,
        max_calls: int,
        recent_turns: int = 3,
        max_chars: int = 26000,
    ) -> list[dict[str, Any]]:
        """Pack system + user + Harness Research State + last `recent_turns` complete turns."""
        out = [dict(initial[0])]
        user_msg = str(initial[1]["content"])
        if self.recent_actions or self.receipts:
            state_block = self.render_state_block(calls_used, max_calls)
            user_msg = f"{user_msg}\n\n{state_block}"
        out.append({"role": "user", "content": user_msg})

        kept: list[list[dict[str, Any]]] = []
        used = sum(len(json.dumps(m, ensure_ascii=False)) for m in out)
        for t in reversed(turns[-recent_turns:]):
            sz = sum(len(json.dumps(m, ensure_ascii=False)) for m in t)
            if kept and used + sz > max_chars:
                break
            kept.append(t)
            used += sz
        for t in reversed(kept):
            out.extend(t)
        return out
