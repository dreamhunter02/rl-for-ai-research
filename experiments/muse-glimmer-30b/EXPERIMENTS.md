# Muse Glimmer 30B — FinanceBench Retrieval & Serverless GRPO Experiments

**Base Model:** `accounts/fireworks/models/muse-glimmer-30b` (Tokenizer: `meta-models/Muse-Glimmer-30B`, Renderer: `muse_glimmer`)  
**Training Platform:** Fireworks Serverless Training API (`async_rl_loop_serverless`, LoRA + MoE Router Replay R3)  
**Judge:** `LiveAdditiveJudge` backed by `accounts/fireworks/models/deepseek-v4p1-flash` (`max_tokens=16000`, disk-cached)  
**Reward Formula:** `R = 0.15E + F(0.20G + 0.45A + 0.15B + 0.05)`  
*(E: retrieved evidence sufficiency, G: cited factual grounding, A: core answer correctness, B: full-reference completeness, F: accepted finish)*

---

## Scoreboard (FinanceBench — 96 train / 12 dev / 42 eval holdout)

| Run ID | Model / Checkpoint | Harness | Split | Scored | Mean `R` | `E` | `G` | `A=1, F=1` | Grounded (`A=1, G=1, F=1`) | Finish Rate (`F`) | Mean Calls |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_dev12` | Base Muse Glimmer 30B | Original (`v1`) | `dev` (12) | 12/12 | `0.6067` | `1.0000` | `0.6167` | `50.0%` (6/12) | `41.7%` (5/12) | `66.7%` (8/12) | `14.83` |
| `baseline_eval42` | Base Muse Glimmer 30B | Original (`v1`) | `eval` (42) | 42/42 | `0.6194` | `0.9048` | `0.6032` | `57.1%` (24/42) | `54.8%` (23/42) | `61.9%` (26/42) | `13.19` |
| `grpo_v2_step0` | Base Muse Glimmer 30B | Harness `v2` | `dev` (12) | 12/12 | **`0.8811`** | — | — | **`91.7%`** (11/12) | **`75.0%`** (9/12) | **`91.7%`** (11/12) | **`7.50`** |
| `grpo_v2` *(in progress)* | Serverless GRPO LoRA (`r=16`) | Harness `v2` | `dev` / `eval` | *running* | *pending* | — | — | *pending* | *pending* | *pending* | *pending* |

---

## Baseline Failure-Mode Diagnostics

1. **High intrinsic accuracy once finishing:** On `baseline_eval42`, when Muse Glimmer 30B called `finish`, accuracy was **`92.3%` (24/26)** (`23/26` fully grounded).
2. **Duplicate tool-call looping despite `E = 1.0`:** On `baseline_eval42`, 15/42 episodes hit `tool_call_limit` (25 calls), and on 10 of those 15 it had already retrieved sufficient evidence (`E = 1.0`). On `dev12`, episodes like `00438` (`7/25` unique calls) and `00499` (`10/25` unique calls) repeated identical `read_table`/`bm25_search` calls after older turns aged out of `ResearchMemory`.
3. **Dimensionless ratio `finish` rejections:** On `04735`, the model found the exact answer (`0.66`) on turn 6, but passed `unit=""` and was rejected 3 times without clear guidance to pass `unit="ratio"`.
4. **Judge reasoning truncation:** `ComponentJudge._request` originally had `max_tokens=6000`, causing 2/12 `dev` episodes to return `judge output incomplete`. Bumping to `16000` resolved `100%` of both `dev12` (12/12) and `eval42` (42/42).

---

## Harness v2 Improvements (`harness/harness_v2_patches.py`)

Model-agnostic fixes diagnosed from `dev12`:
1. Clearer numeric `finish` validation error explicitly noting `unit="ratio"` / `"number"` for dimensionless ratios.
2. Exact-duplicate tool-call loop-breaker note appended when a non-`finish` `(tool, args)` call is repeated, reminding the agent of its existing receipts.
3. Concise system-prompt instruction to compute standard financial ratios via `calculate` from retrieved financial statement line items when not explicitly tabulated.

---

## Active RL Run: `grpo_v2` (`harness/train_muse_grpo.py`)

- **Algorithm:** Fireworks Serverless Async GRPO (`async_rl_loop_serverless`) with **DAPO Clip-Higher** (`eps_clip=0.2`, `eps_clip_high=0.28`), **dynamic zero-variance group filtering**, and **MoE Router Replay (R3)**.
- **LoRA / Optimizer:** `lora_rank=16`, `lora_alpha=32`, `lr=2e-5`, `temperature=1.0`.
- **Shaped Reward:** `R_shaped = max(0, R_additive - min(0.10, 0.03 * exact_repeats) + 0.05 * efficiency_bonus)` (where `efficiency_bonus` is awarded strictly when `F=1, A=1, G=1` and `tool_calls <= 10`, tapering linearly to `20`).
- **Batching:** `48` FinanceBench `train` prompts, `prompt_groups_per_step=6`, `completions_per_prompt=4`, `max_concurrency_rollout_sample=24`, `max_tool_calls=18`, `evaluation_interval=4`.
- **Health (Steps 1–4):** `ppo_ratio_mean=0.9998–1.0001`, `raw_inference_logprob_coverage=1.0`, `ppo_kl=0.0004–0.0007`, `grad_norm=0.57→0.18`, `truncated_ratio≈0.0`.

---

## Phase 2 Roadmap (Next Benchmarks)

After completing FinanceBench holdout evaluation (`eval42`) and promoting the best Muse Glimmer 30B checkpoint:
- **FinAgentBench**
- **FinSearchComp**
- **FinRetrieval** ([arXiv:2603.04403](https://arxiv.org/abs/2603.04403))
