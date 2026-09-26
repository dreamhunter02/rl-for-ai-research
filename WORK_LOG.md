# RL-for-AI-Research — Work Log

**Date:** 2026-09-23
**Goal:** Reproduce/extend a GRPO-trained search agent (Jasper Lu's blog) on the
FinanceBench QA benchmark, with the paper claim that a small open model
(`Qwen3.5-9B-Base`) can match or exceed frontier-model retrieval on finance QA.
Target venue: RL4LLM-Agents @ ICAIF 2026 (deadline Oct 1, 2026 AoE).

---

## 1. Corpus assembly

Source: SEC filings + FinanceBench QA, pulled from the official repos (no bot-walled IR links, no EDGAR fallback needed).

- **PDFs:** 368 filings downloaded from `patronus-ai/financebench` repo `pdfs/` dir → `filings/`
- **QA pairs:** 150 rows from HuggingFace `PatronusAI/financebench` → `data/financebench_merged.jsonl` (canonical merged set; fields include `doc_name, company, question, answer, justification, evidence, gics_sector, doc_type, doc_period`)
- **Text extraction:** 362 of 368 PDFs → `text/*.txt` via pypdf (8 parallel workers). The 6 misses are INTEL 8-Ks not referenced by any QA pair.

### Why repo PDFs instead of EDGAR
The 21 originally-missing filings were bot-walled company-IR PDF links (WAF/Cloudflare challenge pages, timeouts, DNS failures). Rather than retry those or scrape EDGAR (which gave mismatched filing years for some companies), the repo's `pdfs/` dir turned out to contain the full 368-filing corpus with filenames matching our `doc_name` convention exactly. Strictly better, so we switched and deleted the EDGAR/IR fetch attempts.

---

## 2. File layout

```
~/Documents/Research/rl-for-ai-research/
├── filings/                      # 368 PDFs (repo source)
├── text/                         # 362 extracted .txt
├── data/financebench_merged.jsonl # 150 QA pairs (HF source)
├── make_split.py                 # train/eval split builder (reproducible, seed=42)
├── split.json                    # {"train": [...108...], "eval": [...42...]}
├── baseline_eval.jsonl           # zero-RL baseline results (per-question records)
├── WORK_LOG.md                   # this file
└── harness/
    ├── financebench_harness.py   # BM25 index + FinanceBench reward
    ├── finance_env.py            # Bm25Tool, agent env, Tinker wiring, dataset
    ├── train_financebench.py     # GRPO training entrypoint
    └── baseline_eval.py          # zero-RL baseline evaluator (untrained model)
```

Everything was moved out of `/tmp/financebench` into this durable location on
user request; all path constants in the harness were repointed accordingly.

---

## 3. Train/eval split

Built by `make_split.py` (deterministic, seed=42). Verified:

- **Total:** 150 questions
- **Train:** 108 questions (72.0%), 52 filings
- **Eval:** 42 questions (28.0%), 32 filings
- **Leakage:** 0 `doc_name` appears in both train and eval (whole filings kept together)
- **Companies:** 32 in train, 25 in eval (7 companies with only 1-2 questions can't be split, all land in train)
- **Reproducible:** ran twice → identical `split.json` sha256

Note: the "~72% of *filings*" and "stratify by *company*" constraints are in tension
(32 companies × ≥1 filing each can't all be split 72/28 at the filing level). The
split satisfies the stronger guarantees (no doc leakage, company-proportional where
possible, exact 72/28 question-level ratio), landing at 61.9% of filings in train.

---

## 4. Harness (model-agnostic)

- **`financebench_harness.py`** — BM25 index over the 362 filings (reads the
  pre-extracted `text/` cache), top-k passage retrieval, `TextAnswerReward`-style
  grading.
- **`finance_env.py`** — `Bm25Tool` (local BM25, no ChromaDB/Gemini), agent
  tool-calling env, `FinanceRLDataset`, company-name injection into questions
  (handles `3M` ticker, `LockheedMartin`→`Lockheed Martin`).
- **`train_financebench.py`** — GRPO training entrypoint
  (`Qwen/Qwen3.5-9B-Base`, LoRA rank 32, `qwen3_5_disable_thinking` renderer).

Adapted from `tinker_cookbook.recipes.search_tool` (Search-R1 replication),
swapping ChromaTool → Bm25Tool and the Wikipedia dataset → FinanceBench.

---

## 5. Training runs

Model: `Qwen/Qwen3.5-9B-Base` (LoRA r32) on **Tinker** (hosted GRPO).
Project: "RL for AIF" (ID `5485278b-9573-47cd-816c-9e380e84461f`),
API key at `~/.config/tinker/key`.

- **Run A (proof-of-concept, 10 steps, batch 4, group 8, LR 4e-5):** completed.
  Reward went 0 → 3/32 correct (iters 0-3 mean ~-0.07 to -0.10; iter 4 mean +0.031;
  iter 9 mean +0.037). Final: 4 correct (1.0), 15 format-correct (0.0), 61 no-answer
  (-0.1) across 80 trajectories. Checkpoint saved to Tinker.
  Metrics: `/tmp/tinker-examples/rl_finance/finbench_qwen-qwen3.5-9b-base_bs4_gs8_lr4e-05_20260923-1507/`
- **Run B (100 steps):** launched then **stopped on user request** before it
  produced meaningful results.

---

## 6. Zero-RL baseline (the "before" number)

`harness/baseline_eval.py` samples the **untrained** `Qwen3.5-9B-Base` via Tinker on
the held-out 42-question eval split, using the BM25 search tool for up to 5 turns,
N_SAMPLES=3 per question (counts correct if any sample is exact-match). This is the
baseline the paper's Figure 1 "before" row needs (vs FinSage ~50% SOTA).

### Bugs found & fixed in the eval script
1. **Corrupted Tinker key** — `~/.config/tinker/key` had been saved as a config line
   (value not recorded here); the API requires the `tml-` prefix, and the stored
   file had a wrong prefix plus a stray `key=` prefix. Restored the correct key value.
2. **Tool-call parser** — Qwen3.5 emits XML tool calls (a `<tool_call>` wrapper around a
   `<function=search>` block with a `<parameter>` query_list parameter`), not JSON. Rewrote
   `parse_tool_call` to handle both XML and JSON forms; verified it extracts queries
   correctly on a live model response.
3. **Tinker API calls** — `SamplingClient` must be created via
   `ServiceClient(project_id, api_key).create_sampling_client(base_model=MODEL)`
   (not constructed directly, which needs a `sampling_session_id`); pass the
   renderer's `build_generation_prompt(msgs)` `ModelInput` directly to `sample`;
   read `resp.sequences[0].tokens_np` (not `.completion`); and `.result()` the
   async `Future` if present.
4. **Grading crash** — the numeric-compare regex in `grade()` could match a bare
   `.` and crash on `float('.')`; tightened the regex to `-?\d+(?:\.\d+)?`.

### Current state
- Smoke test (2 questions) confirmed the full loop works: the model searches the
  filings and emits answers (e.g. Amazon 2019 → `$2,133 million`).
- Full 42-question eval was started in the background but **hit the grading bug at
  question 3** and stopped (3 of 42 questions logged before the crash).
- **Not yet re-run** (per user instruction: do not restart the baseline now).

---

## 7. What's next (results-perspective plan, agreed with user)

1. ✅ Train/eval split of the 150 (108 / 42, no leakage)
2. ⏳ Zero-RL baseline eval on the 42 held-out → the "before" number (script fixed; needs a clean full run)
3. GRPO train on the 108 train questions (~50 steps)
4. Eval the trained model on the 42 held-out → the "after" number
5. Paper Figure 1: 4-row table — untrained / vector-RAG (~19%) / us-GRPO / FinSage (~50%)

---

## 8. Environment & credentials

- **Hardware:** local DGX Spark `10.0.0.213` (GB10, aarch64, ~128 GB unified, GPU busy with SGLang);
  `sparkytwo` / `spark-a16b` `10.0.0.136` (GB10, idle GPU, ~114 GB RAM free).
- **Tinker client venv:** `/tmp/tinkvenv` (tinker 0.30.1, tinker_cookbook, chz, pypdf, chromadb, google-genai).
- **Tinker key:** `~/.config/tinker/key` (chmod 600, value not recorded here).
- **GitHub:** `gh` v2.101.0 at `~/.local/bin/gh`, authed as `dreamhunter02`.
- **LaTeX:** Tectonic 0.17.0 at `~/.local/bin/tectonic` (no system TeX).

## 10. Structured agentic harness, first training (2026-09-24)

- Reimplemented the harness as `harness/financebench_harness.py` (page-preserving pypdf extraction with `artifacts/page_cache`, BM25 prose and table indexes, `grep_document` with `grep_type` enum, `search_tables`, `read`, `read_table`, `calculate`, `finish`, `record_evidence`) and `harness/finance_env.py` (Tinker RL environment exposing the tools; `FinanceAnswerReward` grades the final `Answer:` line with tolerant deterministic matching).
- `harness/train_financebench.py` trains Qwen3.5-9B (not Base) with GRPO via Tinker; `harness/eval_agent.py` runs inference-only OpenAI and Tinker smoke evals and saves full trajectories.
- First GPT 5.6 terra (medium reasoning) smoke on 3 held-out questions: reward 0.3 (1 exact-or-numeric); Qwen 9B baseline on the same 3: reward 0.333 (1 exact-or-numeric). The two failures are statement-selection and final-answer-format issues, not extraction failures.
- GRPO baseline run `finbench_grpo_qwen35_9b_baseline_20260924-102032` (steps=50, batch=4, group=8, LR=2e-5, max_turns=8, split=train/108) was started and reached iteration 3; it was stopped before any checkpoint or in-training eval to keep the 42-question held-out split clean (final eval only, after training).
- Next: restart GRPO with `save_every=0`, `eval_every=0` by default; evaluate the final checkpoint on the full 42 held-out split; if 9B results are weak, extract GPT 5.6 teacher traces for distillation, otherwise continue with Dr. GRPO-style rollouts.

## 9. Paper and experiment artifact policy

No paper draft copy was found in this folder during the September 23 review; only `WORK_LOG.md` was present outside the corpus/code. Treat the folder as the source of truth for paper-ready artifacts: every run should retain its exact configuration, code revision, split hash, checkpoint identifier, training metrics, per-question trajectories, retrieval/tool statistics, and held-out evaluation outputs. Generate plots from those machine-readable files rather than screenshots.

The Tinker environment now exposes both `search` and bounded `read` tools, matching the intended Jasper-style search-then-inspect loop. The change was syntax-checked with both system Python and `/tmp/tinkvenv` Python; no training run has been started after this change.

## 10. Structured agentic harness, first training (2026-09-24)

- Reimplemented the harness as `harness/financebench_harness.py` (page-preserving pypdf extraction with `artifacts/page_cache`, BM25 prose and table indexes, `grep_document` with `grep_type` enum, `search_tables`, `read`, `read_table`, `calculate`, `finish`, `record_evidence`) and `harness/finance_env.py` (Tinker RL environment exposing the tools; `FinanceAnswerReward` grades the final `Answer:` line with tolerant deterministic matching).
- `harness/train_financebench.py` trains Qwen3.5-9B (not Base) with GRPO via Tinker; `harness/eval_agent.py` runs inference-only OpenAI and Tinker smoke evals and saves full trajectories.
- First GPT 5.6 terra (medium reasoning) smoke on 3 held-out questions: reward 0.3 (1 exact-or-numeric); Qwen 9B baseline on the same 3: reward 0.333 (1 exact-or-numeric). The two failures are statement-selection and final-answer-format issues, not extraction failures.
- GRPO baseline run `finbench_grpo_qwen35_9b_baseline_20260924-102032` (steps=50, batch=4, group=8, LR=2e-5, max_turns=8, split=train/108) was started and reached iteration 3; it was stopped before any checkpoint or in-training eval to keep the 42-question held-out split clean (final eval only, after training).
- Next: restart GRPO with `save_every=0`, `eval_every=0` by default; evaluate the final checkpoint on the full 42 held-out split; if 9B results are weak, extract GPT 5.6 teacher traces for distillation, otherwise continue with Dr. GRPO-style rollouts.

