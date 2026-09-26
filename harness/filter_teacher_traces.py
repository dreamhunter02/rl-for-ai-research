"""Filter GPT 5.6 teacher traces for an SFT-safe first pass.

The automatic FinanceBench reward is strict and rejects many correct
rephrasings, so this script starts from all saved traces, then removes
obvious failures: empty answers, answers containing "not available", answers
that disagree with the gold sign, and answers whose numeric values are not
within 2% of the gold's first numeric value (if the gold is numeric).
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

_NUM = re.compile(r"(?<![A-Za-z0-9])[-+]?\$?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9])")

def nums(s):
    out=[]
    for m in _NUM.findall(s or ""):
        try:
            v=float(m.replace("$","").replace(",","").replace("%",""))
            if m.endswith("%"):
                v=v/100.0
            out.append(v)
        except ValueError:
            pass
    return out

def is_numeric_gold(g):
    return bool(nums(g))

def sign_conflict(gold, ans):
    g=nums(gold); a=nums(ans)
    if g and a and len(a)==1:
        gv=g[0]
        if gv != 0:
            return (a[0] > 0 and gv < 0) or (a[0] < 0 and gv > 0)
    return False

def numeric_close(gold, ans):
    g=nums(gold); a=nums(ans)
    if not g:
        return True
    if not a:
        return False
    gv=g[0]
    return any(abs(av-gv)/abs(gv) <= 0.02 if gv else abs(av) <= 1e-9 for av in a)

def clean_answer(text):
    lines=[l for l in (text or "").splitlines() if l.strip()]
    for l in reversed(lines):
        if re.match(r"^\s*Answer:", l, re.I):
            return re.sub(r"^\s*Answer:\s*", "", l, flags=re.I).strip()
    return lines[-1].strip() if lines else ""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", default="")
    args=ap.parse_args()
    p_in=Path(args.inp); p_out=Path(args.out); p_drop=Path(args.drop) if args.drop else None
    ok=[]; bad=[]
    for line in p_in.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        ans=clean_answer(r.get("answer_text",""))
        gold=r.get("gold","")
        reason=""
        if not ans:
            reason="empty"
        elif ans.lower() in ("not available","n/a","none","na","null"):
            reason="unavailable"
        elif is_numeric_gold(gold) and not numeric_close(gold, ans):
            reason="numeric_mismatch"
        elif is_numeric_gold(gold) and sign_conflict(gold, ans):
            reason="sign_conflict"
        if reason:
            r["filter_reason"]=reason
            bad.append(r)
        else:
            r["clean_answer"]=ans
            ok.append(r)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in ok))
    if p_drop:
        p_drop.parent.mkdir(parents=True, exist_ok=True)
        p_drop.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in bad))
    print(json.dumps({"kept": len(ok), "dropped": len(bad), "out": str(p_out), "drop": str(p_drop) if p_drop else None}))

if __name__ == "__main__":
    main()
