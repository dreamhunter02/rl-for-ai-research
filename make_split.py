#!/usr/bin/env python3
"""Make a train/eval split of the FinanceBench merged QA set.

Design:
- Questions are grouped by doc_name (the source filing) so ALL questions
  from the same filing go to the same split (no leakage).
- Within each company, filings are ordered by question count (largest first,
  ties broken deterministically by doc_name) and dealt alternately to
  train/eval, so each company's questions are split across train/eval
  proportionally where possible.
- ~72% of filings (by question count) go to train, ~28% to eval.
- Fixed random seed (42) is used for any randomized tie-breaking, so the
  split is reproducible.

Output: split.json -> {"train": [...], "eval": [...]}
Each row keeps all original fields (doc_name, company, question, answer,
justification, evidence, ...).
"""
import json
import os
import random
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
QA_PATH = os.path.join(BASE, "data", "financebench_merged.jsonl")
OUT_PATH = os.path.join(BASE, "split.json")
SEED = 42
TRAIN_FRAC = 0.72

KEEP_FIELDS = ["doc_name", "company", "question", "answer", "justification", "evidence"]


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    random.seed(SEED)
    rows = load_rows(QA_PATH)
    total = len(rows)

    # Group questions by doc_name; each doc belongs to exactly one company.
    doc_rows = defaultdict(list)
    doc_company = {}
    for r in rows:
        doc_rows[r["doc_name"]].append(r)
        doc_company[r["doc_name"]] = r["company"]

    # Per company: filings sorted by question count desc, then doc_name asc
    # (deterministic). Dealt alternately to train/eval.
    company_docs = defaultdict(list)
    for d, c in doc_company.items():
        company_docs[c].append(d)

    assign = {}  # doc_name -> "train" | "eval"
    for company, docs in company_docs.items():
        docs = sorted(docs, key=lambda d: (-len(doc_rows[d]), d))
        for i, d in enumerate(docs):
            assign[d] = "train" if i % 2 == 0 else "eval"

    # Adjust toward the target train fraction (~72%) by moving whole
    # doc_groups if it improves closeness to the target, never breaking
    # the doc-grouping or company-alternation guarantees too far.
    def split_counts(a):
        t = sum(len(doc_rows[d]) for d, s in a.items() if s == "train")
        return t, total - t

    target = round(TRAIN_FRAC * total)
    best = dict(assign)
    best_counts = split_counts(best)
    best_err = abs(best_counts[0] - target)

    # Single-doc moves that reduce error to target, deterministic order.
    docs_sorted = sorted(assign.keys(), key=lambda d: (-len(doc_rows[d]), d))
    for d in docs_sorted:
        cand = dict(assign)
        cand[d] = "eval" if assign[d] == "train" else "train"
        t, e = split_counts(cand)
        err = abs(t - target)
        # Keep the company-alternation spirit: only prefer moves that do not
        # leave a company with all its questions in one split when another
        # choice exists. Count companies that would become single-split.
        def single_split_companies(a):
            cs = Counter()
            for comp, docs in company_docs.items():
                s = Counter(a[x] for x in docs)
                if len(s) == 1:
                    cs[comp] = 1
            return len(cs)

        if err < best_err or (err == best_err and single_split_companies(cand) < single_split_companies(best)):
            best, best_err, best_counts = cand, err, (t, e)
            assign = cand

    train_rows = [r for d in sorted(assign) if assign[d] == "train" for r in doc_rows[d]]
    eval_rows = [r for d in sorted(assign) if assign[d] == "eval" for r in doc_rows[d]]

    # Preserve original row order within each split.
    order = {id(r): i for i, r in enumerate(rows)}
    train_rows.sort(key=lambda r: order[id(r)])
    eval_rows.sort(key=lambda r: order[id(r)])

    # Verify the minimum required fields are present in every row.
    for r in train_rows + eval_rows:
        for k in KEEP_FIELDS:
            assert k in r, f"missing field {k!r} in row: {r.get('doc_name')}"

    # No doc_name leakage.
    train_docs = {r["doc_name"] for r in train_rows}
    eval_docs = {r["doc_name"] for r in eval_rows}
    overlap = train_docs & eval_docs
    assert not overlap, f"doc_name overlap across splits: {sorted(overlap)}"

    out = {"train": train_rows, "eval": eval_rows}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Summary
    train_companies = Counter(r["company"] for r in train_rows)
    eval_companies = Counter(r["company"] for r in eval_rows)
    print(f"Total questions: {total}")
    print(f"Train: {len(train_rows)} questions, {len(train_docs)} filings "
          f"({len(train_rows)/total:.1%} of questions)")
    print(f"Eval:  {len(eval_rows)} questions, {len(eval_docs)} filings "
          f"({len(eval_rows)/total:.1%} of questions)")
    print(f"Filings in train: {len(train_docs)}/{len(train_docs)+len(eval_docs)} "
          f"({len(train_docs)/(len(train_docs)+len(eval_docs)):.1%})")
    print(f"Companies in train: {len(train_companies)} "
          f"({', '.join(sorted(train_companies))})")
    print(f"Companies in eval:  {len(eval_companies)} "
          f"({', '.join(sorted(eval_companies))})")
    print(f"doc_name overlap between train and eval: {len(overlap)} (must be 0)")

    # Verify it loads back
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert set(reloaded) == {"train", "eval"}
    assert len(reloaded["train"]) == len(train_rows)
    assert len(reloaded["eval"]) == len(eval_rows)
    assert len(reloaded["train"]) + len(reloaded["eval"]) == total
    print(f"Verification: split.json reloaded OK -> "
          f"train={len(reloaded['train'])}, eval={len(reloaded['eval'])}, "
          f"sum={len(reloaded['train'])+len(reloaded['eval'])} == total {total}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
