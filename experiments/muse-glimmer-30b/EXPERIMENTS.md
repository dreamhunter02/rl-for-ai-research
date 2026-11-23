# Muse Glimmer 30B — FinanceBench Retrieval & Serverless GRPO Experiments

**Base Model:** `accounts/fireworks/models/muse-glimmer-30b` (Tokenizer: `meta-models/Muse-Glimmer-30B`, Renderer: `muse_glimmer`)  
**Training Platform:** Fireworks Serverless Training API (`async_rl_loop_serverless`, LoRA + MoE Router Replay R3)  
**Judge:** `LiveAdditiveJudge` backed by `accounts/fireworks/models/deepseek-v4p1-flash` (`max_tokens=16000`, disk-cached)  
**Reward Formula:** `R = 0.15E + F(0.20G + 0.45A + 0.15B + 0.05)`  
*(E: retrieved evidence sufficiency, G: cited factual grounding, A: core answer correctness, B: full-reference completeness, F: accepted finish)*

---

## Scoreboard (FinanceBench — 96 train / 12 dev / 42 eval holdout)

### Untouched Holdout (`eval`, 42 questions, deterministic `temperature=0.0`)

| Arm | Model / Checkpoint | Harness | Scored | Mean `R` | `E` | `G` | Accuracy (`A=1, F=1`) | Grounded (`A=1, G=1, F=1`) | Finish Rate (`F`) | Mean Calls |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Unmodified Baseline** | Base Muse Glimmer 30B | Original (`v1`) | 42/42 | `0.6194` | `0.9048` | `0.6032` | `57.1%` (24/42) | `54.8%` (23/42) | `61.9%` (26/42) | `13.19` |
| **2. Harness Ablation** | Base Muse Glimmer 30B | Harness `v2` | 41/42 | `0.6573` | `0.8293` | `0.6341` | `65.9%` (27/41) | `56.1%` (23/41) | `66.7%` (28/42) | `12.69` |
| **3. Post-Trained GRPO v2** | **`fb-mg-v2-step-7` LoRA (`r=16`)** | Harness `v2` | 40/42 | **`0.7081`** | `0.8750` | **`0.6844`** | **`70.0%`** (28/40) | **`65.0%`** (26/40) | **`71.4%`** (30/42) | **`11.74`** |

**Causal Attribution on `eval` (42):**
- **Total Gain (GRPO v2 + Harness v2 vs. Unmodified Baseline):** `+0.0887` Mean `R`, **`+12.9pp` Accuracy**, **`+10.2pp` Grounded Accuracy**, **`+9.5pp` Finish Rate**, `-11%` tool calls (`-20%` wall time).
- **Pure GRPO Post-Training Gain (Arm 3 vs. Arm 2, identical Harness v2):** `+0.0508` Mean `R`, **`+4.1pp` Accuracy**, **`+8.9pp` Grounded Accuracy**, **`+4.8pp` Finish Rate**, `-0.95` tool calls/question.
- **Precision on Completed Finishes:** **`28 / 28 = 100.0%`** accuracy on every scored `finish` in Arm 3!

### Dev Split (`dev`, 12 questions)

| Run ID | Model / Checkpoint | Harness | Temp | Scored | Mean `R` | Accuracy (`A=1, F=1`) | Grounded (`A=1, G=1, F=1`) | Finish Rate (`F`) | Mean Calls |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_dev12` | Base Muse Glimmer 30B | Original (`v1`) | `0.0` | 12/12 | `0.6067` | `50.0%` (6/12) | `41.7%` (5/12) | `66.7%` (8/12) | `14.83` |
| `grpo_v2_step0` | Base Muse Glimmer 30B | Harness `v2` | `1.0` | 12/12 | `0.8811` | `91.7%` (11/12) | `75.0%` (9/12) | `91.7%` (11/12) | `7.50` |
| `grpo_v2_step7` | `fb-mg-v2-step-7` LoRA | Harness `v2` | `1.0` | 11/12 | `0.7792` | `72.7%` (8/11) | `63.6%` (7/11) | `90.9%` (10/11) | `6.73` |

---

## Completed RL Run: `grpo_v2` (`harness/train_muse_grpo.py`)

- **Run ID:** `accounts/vineethkalluru/trainingRuns/run-f38c03fe7bc7473f9619967aa8f6066e`
- **Session:** `accounts/vineethkalluru/trainingSessions/ts-008e603455c44d49863c072db7875048`
- **Final Sampler Snapshot:** `vineethkalluru/run-f38c03fe7bc7473f9619967aa8f6066e/fb-mg-v2-step-7-ef2b7444`
- **Resumable Training Checkpoint:** `vineethkalluru/run-f38c03fe7bc7473f9619967aa8f6066e/fb-mg-v2-state-step-7`
- **Algorithm:** Fireworks Serverless Async GRPO (`async_rl_loop_serverless`) with **DAPO Clip-Higher** (`eps_clip=0.2`, `eps_clip_high=0.28`), **dynamic zero-variance group filtering**, and **MoE Router Replay (R3)**.
- **Hyperparameters:** `48` train rows (first half of `train96`), `7` optimizer steps, `37` accepted prompt groups, `prompt_groups_per_step=6`, `completions_per_prompt=4`, `lora_rank=16`, `lora_alpha=32`, `lr=2e-5`, `max_tool_calls=18`.
- **Shaped Reward:** `R_shaped = max(0, R_additive - min(0.10, 0.03 * exact_repeats) + 0.05 * efficiency_bonus)`.

---

## Remaining Failure Modes on `eval42` (Next Iteration Targets)

Out of 12 non-finish episodes in `grpo_v2_step7_eval42`:
1. **Company-name / ticker alias mismatch (4/12):** 4 Johnson & Johnson (`JnJ`) questions hit `tool_call_limit` (`01487`, `01488`, `01490`, `01491`, three with `E=0.0`).
2. **"None / Not applicable" negative-answer loops (2/12):** Ulta Beauty `00746` and `00521` had `E=1.0` (filing confirms none), yet kept searching for 25 calls instead of concluding "None".
3. **Mental-math token overflow (1/12):** General Mills CCC (`04103`) retrieved `E=1.0` in 4 calls, then hit `generation_token_limit` (4096 tokens) instead of calling `calculate`.

---

## Phase 2 Roadmap (Next Benchmarks)

- **FinAgentBench**
- **FinSearchComp**
- **FinRetrieval** ([arXiv:2603.04403](https://arxiv.org/abs/2603.04403))
