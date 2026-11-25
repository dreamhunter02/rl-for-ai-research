# Muse Glimmer 30B — Uniform Financeharness v3 & Multi-Benchmark Serverless GRPO

**Base Model:** `accounts/fireworks/models/muse-glimmer-30b` (Tokenizer: `meta-models/Muse-Glimmer-30B`, Renderer: `muse_glimmer`)  
**Training Platform:** Fireworks Serverless Training API (`async_rl_loop_serverless`, LoRA `r=16`, `alpha=32`, DAPO Clip-Higher + MoE Router Replay R3)  
**Judge:** `accounts/fireworks/models/deepseek-v4p1-flash` (disk-cached)

---

## Promoted Models (Fireworks Custom Models — `READY`)

| Model ID | Training Stage | Resumable Training Checkpoint |
| :--- | :--- | :--- |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fb-grpo-v2` | Stage 1: FinanceBench GRPO (`7` steps) | `vineethkalluru/run-f38c03fe7bc7473f9619967aa8f6066e/fb-mg-v2-state-step-7` |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fin-multibench-v1` | Stage 2: Multi-Benchmark GRPO (`5` steps, init from Stage 1) | `vineethkalluru/run-064f38c62e9248eb976114073fc53ec1/mb-mg-v1-state-step-5` |
| **`accounts/vineethkalluru/models/muse-glimmer-30b-fin-multibench-v2`** | **Stage 3: Multi-Benchmark GRPO in `Financeharness v3` (`6` steps, init from Stage 2)** | `vineethkalluru/run-c60bd4380a564247bdfec1c22c1acc3a/mb-mg-v2-state-step-6` |

---

## Multi-Benchmark Scoreboard (All Untouched Holdouts, Deterministic `temperature=0.0`)

### Phase 2 & 3: FinRetrieval + FinAgentBench + FinSearchComp (`n=90`, 30 held-out eval questions each)

| Benchmark | Base Muse Glimmer 30B (`v1`) | Stage 2 (`multibench-v1`) | **Stage 3 (`multibench-v2` + `Financeharness v3`)** | **Total Gain vs. Base** |
| :--- | :---: | :---: | :---: | :---: |
| **FinAgentBench** (`n=30`) | `40.0%` (12/30) · `F=46.7%` | `46.7%` (14/30) · `F=56.7%` | **`70.0%` (21/30)** · **`F=76.7%`** | **`+30.0pp`** (`+30.0pp` finish) |
| **FinRetrieval** ([arXiv:2603.04403](https://arxiv.org/abs/2603.04403), `n=30`) | `70.0%` (21/30) · `9.37` calls | **`86.7%`** (26/30) · `10.23` calls | **`83.3%` (25/30)** · **`6.70` calls** | **`+13.3pp`** (`-28%` calls) |
| **FinSearchComp** (`Global T2+T3`, `n=30`) | `6.7%` (2/30) · `F=6.7%` | `16.7%` (DDG) → `20.0%` (BB) | **`30.0%` (9/30)** · **`F=46.7%`** | **`+23.3pp` (4.5×)** (`+40.0pp` finish) |
| **3-Benchmark Macro Average** (`n=90`) | `38.9%` (`R=0.3900`, `F=41.1%`) | `50.0%` (`R=0.5033`, `F=56.7%`) | **`61.1%` (`R=0.6155`, `F=70.0%`)** | **`+22.2pp`** (`+28.9pp` finish) |

### Phase 1: FinanceBench (`eval`, `n=42`, deterministic `temperature=0.0`)

| Arm | Model / Checkpoint | Harness | Scored | Mean `R` | Accuracy (`A=1, F=1`) | Grounded (`A=1, G=1, F=1`) | Finish Rate (`F`) | Mean Calls |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Unmodified Baseline** | Base Muse Glimmer 30B | Original (`v1`) | 42/42 | `0.6194` | `57.1%` (24/42) | `54.8%` (23/42) | `61.9%` (26/42) | `13.19` |
| **2. Harness Ablation** | Base Muse Glimmer 30B | Harness `v2` | 41/42 | `0.6573` | `65.9%` (27/41) | `56.1%` (23/41) | `66.7%` (28/42) | `12.69` |
| **3. GRPO v2 (Step 7)** | **`muse-glimmer-30b-fb-grpo-v2`** | Harness `v2` | 40/42 | **`0.7081`** | **`70.0%` (28/40)** | **`65.0%` (26/40)** | **`71.4%` (30/42)** | **`11.74`** |

---

## `Financeharness v3` Uniform Architecture (`harness/financeharness_v3.py`)

1. **Uniform `UniformResearchMemory` + Receipt Ledger:** Discovery tools never create receipts; inspection tools issue citable receipts (`r1, r2, ...`) packed automatically into every prompt (no memory tool).
2. **MintHarness-Style Structured Receipt Headers:** Deterministic 1-line metadata headers (`[r1 | AAPL | sec_company_facts | period:2024 | unit:USD]`) prepended to verbatim receipts.
3. **Sandboxed `run_python` Tool:** Isolated `python -I` subprocess (2s timeout, AST import/dunder guard, `math`/`statistics`/`Decimal` pre-imported).
4. **Dynamic Entity/Acronym Alias Normalization:** Automatically resolves `JnJ`/`J&J` ↔ `JOHNSON_JOHNSON`, `3M` ↔ `MMM`, `P&G`, `BofA`, etc., preserving zero-vector-DB BM25 lexical precision.
5. **Budget-Aware Browserbase Search + Fetch:** Normalized disk cache + free direct `urllib` fetch first, escalating to Browserbase `/v1/fetch` only on `403`/JS walls, with hard session cap (`BB_MAX_LIVE_CALLS`).
6. **Negative-Evidence Abstention & Action-Budget Cues:** Explicit abstention path for adversarial/unanswerable questions + budget cues on the final 3 calls.

---

## Next Phase: Sub-10B Transfer (`Qwen3.5-4B` on Local DGX Sparks)

Harvest verified winning trajectories from `muse-glimmer-30b-fin-multibench-v2` inside `Financeharness v3` → SFT/distillation into `Qwen3.5-4B` → local Unsloth/TRL GRPO across all 4 benchmarks.
