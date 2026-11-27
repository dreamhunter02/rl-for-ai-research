# Muse Glimmer 30B — Uniform Financeharness v3.5 & Multi-Benchmark Serverless GRPO

**Base Model:** `accounts/fireworks/models/muse-glimmer-30b` (Tokenizer: `meta-models/Muse-Glimmer-30B`, Renderer: `muse_glimmer`)  
**Training Platform:** Fireworks Serverless Training API (`async_rl_loop_serverless`, LoRA `r=16`, `alpha=32`, DAPO Clip-Higher + MoE Router Replay R3)  
**Judge:** `accounts/fireworks/models/deepseek-v4p1-flash` (disk-cached)

---

## Promoted Models (Fireworks Custom Models — `READY`, Downloadable via `firectl download model`)

| Model ID | Training Stage | Resumable Training Checkpoint |
| :--- | :--- | :--- |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fb-grpo-v2` | Stage 1: FinanceBench GRPO (`7` steps) | `vineethkalluru/run-f38c03fe7bc7473f9619967aa8f6066e/fb-mg-v2-state-step-7` |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fin-multibench-v1` | Stage 2: Multi-Benchmark GRPO (`5` steps, init from Stage 1) | `vineethkalluru/run-064f38c62e9248eb976114073fc53ec1/mb-mg-v1-state-step-5` |
| **`accounts/vineethkalluru/models/muse-glimmer-30b-fin-multibench-v2`** *(Overall Champion)* | **Stage 3: Multi-Benchmark GRPO in `Financeharness v3` (`6` steps)** | `vineethkalluru/run-c60bd4380a564247bdfec1c22c1acc3a/mb-mg-v2-state-step-6` |
| `accounts/vineethkalluru/models/muse-glimmer-30b-fin-multibench-v3` | Stage 4: Multi-Benchmark GRPO in `Financeharness v3.5` (`8` steps, `max_calls=20`) | `vineethkalluru/run-834107e63c4e4abe8946eede2d38d56c/mb-mg-v3-state-step-8` |

---

## Final 4-Benchmark Scoreboard (All Untouched Holdouts, Deterministic `temperature=0.0`)

| Benchmark | Split | Base Muse Glimmer 30B (`v1`) | **Best GRPO + Uniform `Financeharness`** | Total Gain | Frontier SOTA Reference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FinanceBench** | `eval` (`n=42`) | `57.1%` (`R=0.6194`, `F=61.9%`) | **`80.5%`** (`R=0.8098`, `E=97.6%`, `33/33=100%` on finished) *(`multibench-v2` + `v3.5`)* | **`+23.4pp`** | `~81–85%` *(GLM 5.3 Flash `81%`)* |
| **FinAgentBench** | `eval` (`n=30`) | `40.0%` (`F=46.7%`) | **`70.0%`** (`21/30`, `F=76.7%`) *(`multibench-v2` + `v3`, `max_calls=15`)* | **`+30.0pp`** | `~69.5%` *(Vals AI FA v2.1)* |
| **FinRetrieval** ([arXiv:2603.04403](https://arxiv.org/abs/2603.04403)) | `eval` (`n=30`) | `70.0%` (`9.37` calls) | **`86.7%`** (`26/30`, `multibench-v1`; `83.3%` @ `6.70` calls on `v2`) | **`+16.7pp`** | `90.8%` *(Opus 4.5 + Daloopa)* |
| **FinSearchComp** *(Global T2+T3)* | `eval` (`n=30`) | `6.7%` (`F=6.7%`) | **`30.0%`** (`9/30`, `F=46.7%`, `multibench-v2`/`v3` + Browserbase) | **`+23.3pp` (4.5×)** | `~57–60%` on T2+T3 *(Grok 4 `68.9%` incl. T1)* |

### Key Experimental Finding (`max_tool_calls=15` vs. `20`)
On identical `Financeharness v3.5` (`max_tool_calls=20`), `multibench-v3` beat `multibench-v2` (`57.8%` vs. `55.6%` macro; `30.0%` vs. `23.3%` FinSearchComp). However, tighter action-budget pacing (`max_tool_calls=15`, triggering the final-3-call budget cue at call 12) produced higher overall macro accuracy (`61.1%` vs. `57.8%`) and `FinAgentBench` accuracy (`70.0%` vs. `60.0%`) by preventing late-turn over-searching.
