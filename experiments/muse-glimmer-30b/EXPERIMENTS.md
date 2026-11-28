# Muse Glimmer 30B — Uniform Financeharness v3.5 & Multi-Benchmark Serverless GRPO

**Base Model:** `accounts/fireworks/models/muse-glimmer-30b` (Tokenizer: `meta-models/Muse-Glimmer-30B`, Renderer: `muse_glimmer`)  
**Training Platform:** Fireworks Serverless Training API (`async_rl_loop_serverless`, LoRA `r=16`, `alpha=32`, DAPO Clip-Higher + MoE Router Replay R3)  
**Search & Data Stack:** Perplexity Search (NVIDIA Inference Hub) + Browserbase Proxy Fetch fallback + PyMuPDF PDF-URL extraction + SEC EDGAR XBRL + `yfinance` + Local FinRetrieval SQLite DB + Sandboxed `run_python`  
**Judge:** `accounts/fireworks/models/deepseek-v4p1-flash` (disk-cached)

---

## Promoted Models (Fireworks Custom Models — `READY`, Downloadable via `firectl download model`)

| Model ID | Training Stage | Resumable Training Checkpoint |
| :--- | :--- | :--- |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fb-grpo-v2` | Stage 1: FinanceBench GRPO (`7` steps) | `vineethkalluru/run-f38c03fe7bc7473f9619967aa8f6066e/fb-mg-v2-state-step-7` |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fin-multibench-v1` | Stage 2: Multi-Benchmark GRPO (`5` steps, init from Stage 1) | `vineethkalluru/run-064f38c62e9248eb976114073fc53ec1/mb-mg-v1-state-step-5` |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fin-multibench-v2` | Stage 3: Multi-Benchmark GRPO in `Financeharness v3` (`6` steps) | `vineethkalluru/run-c60bd4380a564247bdfec1c22c1acc3a/mb-mg-v2-state-step-6` |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fin-multibench-v3` | Stage 4: Multi-Benchmark GRPO in `Financeharness v3.5` (`8` steps) | `vineethkalluru/run-834107e63c4e4abe8946eede2d38d56c/mb-mg-v3-state-step-8` |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fin-sota-v4` | Stage 5: FinSearchComp-Focused GRPO w/ Perplexity Search (`5` steps) | `vineethkalluru/run-4a4ce00a9e704e8688c7f7e1ea1e1267/fsc-mg-v4-state-step-5` |

---

## Final 4-Benchmark Scoreboard (All Untouched Holdouts, Deterministic `temperature=0.0`)

| Benchmark | Split | Base Muse Glimmer 30B (`v1`) | **Best GRPO + Uniform `Financeharness v3.5`** | Total Gain | Frontier SOTA Reference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FinanceBench** | `eval` (`n=42`) | `57.1%` (`R=0.6194`) | **`80.5%`** (`R=0.8098`, `E=97.6%`, `33/33=100%` on finished) *(`multibench-v2` + `v3.5`)* | **`+23.4pp`** | `~81–85%` *(GLM 5.3 Flash `81%`)* |
| **FinAgentBench** | `eval` (`n=30`) | `40.0%` (`F=46.7%`) | **`70.0%`** (`21/30`, `F=76.7%`) *(`multibench-v2` + `v3`, `max_calls=15`)* | **`+30.0pp`** | `~69.5%` *(Vals AI FA v2.1)* |
| **FinRetrieval** ([arXiv:2603.04403](https://arxiv.org/abs/2603.04403)) | `eval` (`n=30`) | `70.0%` (`9.37` calls) | **`86.7%`** (`26/30` on `v1`; `83.3%` @ `6.70` calls on `v2`) | **`+16.7pp`** | `90.8%` *(Opus 4.5 + Daloopa)* |
| **FinSearchComp** *(Global T2+T3)* | `eval` (`n=30`) | `6.7%` (`F=6.7%`) | **`50.0%`** (`15/30`, `F=60.0%`, `15/18=83.3%` on finished, `13.87` calls) *(`fin-sota-v4` + Perplexity Search)* | **`+43.3pp` (7.5×)** | `~57–60%` on T2+T3 *(Grok 4 `68.9%` incl. T1)* |

### FinSearchComp Progression (`eval30`, Global T2+T3)
1. Base Muse Glimmer 30B + DuckDuckGo (`v1`): `6.7%` (`2/30`, `F=6.7%`)
2. Stage 2 GRPO (`multibench-v1`) + DuckDuckGo: `16.7%` (`5/30`, `F=23.3%`)
3. Stage 2 GRPO (`multibench-v1`) + Browserbase Search (empty snippets): `20.0%` (`6/30`, `F=30.0%`)
4. Stage 3/4 GRPO (`multibench-v2`/`v3`) + `Financeharness v3`: `30.0%` (`9/30`, `F=46.7%`)
5. Stage 4 GRPO (`multibench-v3`) + Perplexity Search (Inference Hub) + multi-year `yfinance` + 24-val `sec_company_facts`: `46.7%` (`14/30`, `F=60.0%`, T2 `56.2%`, T3 `35.7%`)
6. **Stage 5 GRPO (`fin-sota-v4`) + Perplexity Search (`5` steps, `26` groups):** **`50.0%` (`15/30`, `F=60.0%`, `83.3%` precision on finished, T2 `56.2%`, T3 `42.9%` — 7.5× baseline, zero regressions)**

---

## Stage 5.5 Harness Upgrade (`fred_search` + Forced Final Synthesis Turn)
- **FinSearchComp `eval30` (`fin-sota-v4`, `max_calls=24`, `temperature=0.0`):** **`53.3%` (`16/30`, `F=83.3%`, `R=0.5483` — 8.0× baseline)**

---

## Sub-10B Transfer Baselines: `Qwen/Qwen3.5-4B` (Untuned Base) in Uniform `Financeharness v3.5`
Served locally on `sparkyone` (DGX Spark GB10, vLLM `qwen3_coder` tool parser + `qwen3` reasoning parser, `temperature=0.0`):

| Benchmark | Split | `Qwen3.5-4B` Base (`v3.5`) | Best Muse Glimmer 30B GRPO (`v3.5`) | Teacher–Student Gap | Frontier SOTA Ref |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FinanceBench** | `eval` (`n=42`) | **`73.2%`** (`R=0.7698`, `E=85.4%`, `F=85.7%`, `6.31` calls) | **`80.5%`** (`R=0.8098`, `E=97.6%`) | `+7.3pp` | `~81–85%` |
| **FinAgentBench** | `eval` (`n=30`) | **`60.0%`** (`18/30`, `F=83.3%`, `6.50` calls) | **`70.0%`** (`21/30`, `F=76.7%`) | `+10.0pp` | `~69.5%` |
| **FinRetrieval** | `eval` (`n=30`) | **`66.7%`** (`20/30`, `F=80.0%`, `4.13` calls) | **`86.7%`** (`26/30`) | `+20.0pp` | `90.8%` |
| **FinSearchComp** *(Global T2+T3)* | `eval` (`n=30`) | **`40.0%`** (`12/30`, `F=80.0%`, `6.20` calls) | **`53.3%`** (`16/30`, `F=83.3%`) | `+13.3pp` | `~57–60%` |
| **4-Benchmark Macro Average** | — | **`60.0%`** | **`72.6%`** | **`+12.7pp`** | **`~75.5%`** |
