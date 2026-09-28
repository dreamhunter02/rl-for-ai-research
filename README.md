# FinanceBench agentic GRPO

This repository contains the reproducible FinanceBench retrieval harness, teacher-trace artifacts, baseline evaluations, Unsloth/TRL smoke tests, and the minimal FinanceBench-to-LoRA optimizer-step bridge.

The corpus PDFs, extracted text, page-cache index, large Tinker run logs, and generated adapter weights are intentionally excluded from GitHub. They are local or regenerable artifacts; the tracked JSONL traces and metrics preserve the experiment evidence without publishing the corpus or a large binary checkpoint.

## Current status: Reward-v2 FinanceBench GRPO review (2026-09-27)

The reward-v2 implementation, audit harness, teacher-trace scoring, Tinker integration, and full-run artifact capture are complete and available in this repository. The full run completed technically, but it is a diagnostic/training regression rather than evidence of model improvement; the 42-question holdout evaluation is still pending.

### Implementation

- `harness/reward_calculation.py` — deterministic finish, contradiction, direction, numeric/unit/date/arithmetic checks; DeepSeek residual semantic judging through DeepInfra; strict JSON parsing; confidence thresholding; asynchronous cache writes; ephemeral GNOME Keyring retrieval; audit metadata.
- `harness/finance_env.py` — Tinker reward bridge, finish handling, grounded-reward metrics, and numeric-only reducer-safe metrics.
- `harness/train_financebench.py` — Tinker training entry point and run metadata.
- `harness/audit_deepseek_reward.py` — known-case reward audit, including qualitative paraphrases and hard negatives.
- `harness/score_teacher_traces.py` — scoring harness for the 91 saved teacher traces.
- `harness/test_reward_redesign.py`, `harness/test_reward_design.py`, and `harness/test_tinker_reward.py` — regression coverage; latest suite result was 20 tests passing.
- `configs/nemotron35_lightning_grpo_reward_v2.json` — full Reward-v2 configuration.
- `REWARD_REDESIGN.md` — reward contract and stability notes.

### Trace and audit results

- Teacher set: `results/teacher_traces/sft_judged_strict.jsonl` contains 91 traces.
- Teacher audit: `results/reward_audits/teacher91_reward_v2_summary_final.json` and `teacher91_reward_v2_scores_final.json`.
- Teacher audit outcome: mean answer quality `0.9893`, mean evidence quality `0.5545`, mean grounded reward without the finish gate `0.7699`, `83/83` strict Opus-correct cases retained, and `0/91` teacher traces containing a terminal finish call.
- Live semantic audit: `results/reward_audits/deepseek_v41_reward_audit_small_live.json` and `deepseek_v41_reward_v2_summary.json`; the 20-case audit recovered the Amcor qualitative paraphrase and preserved the Verizon directional hard negative.
- Judge cache: `results/reward_judgments/deepseek_v41_flash_audit_live_cache.json` and `deepseek_v41_flash_teacher91_final_cache.json`; credential values are not stored.

### Full Reward-v2 run

Run identity: `nemotron35_financebench_grpo_reward_v2_full_seed0_20260927`

- Configuration: 108 questions, 27 optimizer steps, batch size 4, group size 8, 864 trajectories, maximum 8 turns, learning rate `1e-5`, seed `0`.
- Outcome: 220/864 valid finishes (`25.5%`), 187 positive trajectories, 616 zero-reward trajectories, 61 trajectories at `-0.1`, mean authoritative total reward `0.1475`, and 38 all-zero groups out of 108.
- Trace volume: 38,760 low-level trace events, six checkpoints, and 32 DeepSeek cache records with zero observed judge errors.
- Comparison: the prior grounded v11 run had 213/864 valid finishes, 319 positive trajectories, and mean reward `0.2573`; Reward-v2 slightly improved finishing but produced a weaker learning signal.
- Interpretation: the dominant issue is sparse reward and a mismatch between the required terminal finish action and the demonstrations; rollout count alone is not the immediate fix.

The canonical report is `results/tinker_runs/nemotron35_lightning_grpo_reward_v2_full_20260927/EXPERIMENT_REPORT.md`. The large local Tinker logs, rollout summaries, checkpoints, and trace files remain in that run directory and are intentionally ignored by GitHub; the report records their exact paths and checkpoint URIs.

### Proposed resolution under review

`REWARD_V2_PROBLEM_AND_RESOLUTION.md` contains the complete problem statement and proposed resolution, including the DeepSeek review. The current recommendation is:

1. Create 91–200 finish-annotated SFT examples; use them for format/termination behavior, not as a claim of broad finance-reasoning supervision.
2. Run short 10–15-step controlled ablations before another full run: gated factorized reward, gated reward plus verified trajectory shaping capped at `0.05`, label-noise control, and an unfinished-credit comparison.
3. Keep answer correctness, evidence grounding, finish behavior, and trajectory/process reward as separate metrics; treat numeric zero as valid; retain hard contradiction vetoes for answer quality.
4. Add trajectory reward only for verifiable behaviors such as evidence actually cited, evidence carried across turns, no repeated identical tool calls, and a valid finish; do not reward tool volume or termination alone.
5. Require improvement on the untouched 42-question holdout against the untrained base-model baseline before claiming success.

DeepSeek's detailed review is tracked at `results/reward_audits/deepseek_v41_reward_design_review_final_20260927.md`.

Key files:

- `harness/financebench_harness.py` — page-aware BM25/table retrieval, reads, calculations, provenance, and answer/evidence scoring.
- `harness/finance_env.py` — typed FinanceBench environment and reward integration.
- `harness/generate_teacher_traces.py` — multi-turn trace generation with finish handling, seeds, and grounded rewards.
- `harness/financebench_grpo_trial.py` — minimal harness-to-Unsloth optimizer-step bridge.
- `harness/unsloth_qwen4b_grpo_smoke.py` — Unsloth/TRL toy GRPO smoke test.
- `results/teacher_traces/` — teacher candidates, judgments, and strict selected traces.
- `results/local_eval/` — baseline and rollout evaluation JSONL artifacts.
- `GRPO_PROGRESS.md` — experiment record and next steps.
- `SFT_PLAN.md` — supervised warm-start plan; it is now the recommended next diagnostic because the teacher traces lack finish calls.

The current project status is documented in `GRPO_PROGRESS.md`; no claim of FinanceBench model improvement should be made from the toy smoke test or the one-step bridge proof alone.
