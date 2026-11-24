# Muse Glimmer 30B — Multi-Benchmark Finance Retrieval & Serverless GRPO

**Base Model:** `accounts/fireworks/models/muse-glimmer-30b` (Tokenizer: `meta-models/Muse-Glimmer-30B`, Renderer: `muse_glimmer`)  
**Training Platform:** Fireworks Serverless Training API (`async_rl_loop_serverless`, LoRA `r=16`, `alpha=32`, DAPO Clip-Higher + MoE Router Replay R3)  
**Judge:** `accounts/fireworks/models/deepseek-v4p1-flash` (disk-cached)

---

## Promoted Models (Fireworks Custom Models — `READY`)

| Model ID | Training Stage | Resumable Training Checkpoint |
| :--- | :--- | :--- |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fb-grpo-v2` | Stage 1: FinanceBench GRPO (`7` steps) | `vineethkalluru/run-f38c03fe7bc7473f9619967aa8f6066e/fb-mg-v2-state-step-7` |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fin-multibench-v1` | Stage 2: Multi-Benchmark GRPO (`5` steps, init from Stage 1) | `vineethkalluru/run-064f38c62e9248eb976114073fc53ec1/mb-mg-v1-state-step-5` |

---

## Multi-Benchmark Scoreboard (All Untouched Holdouts, Deterministic `temperature=0.0`)

### Phase 2: FinRetrieval + FinAgentBench + FinSearchComp (`n=90`, 30 held-out eval questions each)

| Benchmark | Base Muse Glimmer 30B | **`muse-glimmer-30b-fin-multibench-v1`** | Accuracy Delta | Finish Rate (`Base -> GRPO`) |
| :--- | :---: | :---: | :---: | :---: |
| **FinRetrieval** ([arXiv:2603.04403](https://arxiv.org/abs/2603.04403), `n=30`) | `70.0%` (21/30) | **`86.7%` (26/30)** | **`+16.7pp`** | `70.0%` → **`90.0%`** |
| **FinAgentBench** (`n=30`) | `40.0%` (12/30) | **`46.7%` (14/30)** | **`+6.7pp`** | `46.7%` → **`56.7%`** |
| **FinSearchComp** (`Global T2+T3`, `n=30`) | `6.7%` (2/30) | **`16.7%` (5/30)** | **`+10.0pp` (2.5×)** | `6.7%` → **`23.3%`** |
| **3-Benchmark Macro Average** (`n=90`) | `38.9%` (`R=0.3900`) | **`50.0%` (`R=0.5033`)** | **`+11.1pp`** | `41.1%` → **`56.7%`** |

### Phase 1: FinanceBench (`eval`, `n=42`, deterministic `temperature=0.0`)

| Arm | Model / Checkpoint | Harness | Scored | Mean `R` | Accuracy (`A=1, F=1`) | Grounded (`A=1, G=1, F=1`) | Finish Rate (`F`) | Mean Calls |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Unmodified Baseline** | Base Muse Glimmer 30B | Original (`v1`) | 42/42 | `0.6194` | `57.1%` (24/42) | `54.8%` (23/42) | `61.9%` (26/42) | `13.19` |
| **2. Harness Ablation** | Base Muse Glimmer 30B | Harness `v2` | 41/42 | `0.6573` | `65.9%` (27/41) | `56.1%` (23/41) | `66.7%` (28/42) | `12.69` |
| **3. GRPO v2 (Step 7)** | **`muse-glimmer-30b-fb-grpo-v2`** | Harness `v2` | 40/42 | **`0.7081`** | **`70.0%` (28/40)** | **`65.0%` (26/40)** | **`71.4%` (30/42)** | **`11.74`** |

---

## Unified Harness Architecture (`harness/`)

- **`harness/benchmarks/prepare_multibench_data.py`:** Builds the local FinRetrieval SQLite structured financial DB (`4,854` companies, `24,519` series, `2,447` fundamentals from `daloopa/finretrieval`), caches SEC EDGAR XBRL CompanyFacts + submissions for all 30 FinAgentBench tickers, and generates stratified zero-leakage train/eval splits (`FinRetrieval 400/50/50`, `FinAgentBench 93/40`, `FinSearchComp Global T2+T3 153/50`).
- **`harness/benchmarks/multibench_env.py`:** Unified stateful tool environments & graders:
  - **FinRetrieval:** `discover_companies`, `discover_company_series`, `get_company_fundamentals`, `calculate`, `finish`
  - **FinAgentBench:** `sec_company_facts`, `sec_list_filings`, `sec_search_filing`, `yfinance_info`, `calculate`, `finish` (with explicit `is_unanswerable` support for adversarial questions)
  - **FinSearchComp:** `web_search` (disk-cached multi-backend DuckDuckGo/Bing/Yahoo), `fetch_url`, `yfinance_history`, `yfinance_info`, `sec_company_facts`, `calculate`, `finish` (graded with FinSearchComp's official judge prompt)
- **`harness/eval_multibench.py`:** Unified deterministic evaluator via `FiretitanSamplingClient` + `MuseGlimmerRenderer` (exact TITO parity).
- **`harness/train_multibench_grpo.py`:** Interleaved multi-benchmark Fireworks Serverless Async GRPO trainer (DAPO Clip-Higher `eps_clip=0.2, eps_clip_high=0.28`, MoE Router Replay R3, dynamic zero-variance filtering, anti-loop repeat penalty `-0.03`, efficiency bonus `+0.05` for correct answers in `<= 6` calls, and budget-remaining cues on the final 3 calls).
