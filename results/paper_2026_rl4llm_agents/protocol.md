# ICAIF 2026 RL4LLM-Agents FinanceBench protocol

Protocol version: `2026-09-27-repaired-v1`

## Research question

How much improvement comes from repairing the FinanceBench agent harness, and how much additional improvement comes from GRPO?

## Conditions

- B0: base Nemotron checkpoint with the original harness, only if it can be reproduced under a frozen compatible evaluator; otherwise report B0 as diagnostic and exclude it from paired claims.
- B1: the same base checkpoint with the repaired harness and frozen evaluator.
- R1: GRPO from the same base checkpoint and repaired harness, with the selected pilot recipe.
- Optional arms: finish-only SFT and a matched retrieval-shaping ablation, clearly separated from the main B0/B1/R1 comparison.

## Model and renderer

- Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
- Renderer: `nemotron3_ultra`, verified against the pinned installed Tinker SDK before training.
- Training seed and evaluation sampling seed are recorded separately.

## Splits

- `train96`: 96 questions from the original 108-question training split.
- `dev12`: 12 stable, document-diverse questions reserved from the original training split.
- `eval42`: the original 42-question evaluation split; it has already been inspected and is therefore reported as previously inspected, not untouched.
- Exact IDs are in `split_manifest.json`.
- No gold answers, gold evidence, or evaluator-only targets are inserted into student prompts.

## Budgets and schedule

- Default starting budget: 8 turns and 1,024 generation tokens.
- Increase to 2,048 tokens or 12 turns only when dev failure slices justify it; freeze the choice afterward.
- Pilot: 8 prompt groups × 8 rollouts, up to 10 actual updates per recipe.
- Final candidate: at most 3 epochs over 96 training questions; batch 8/group 8 gives 36 nominal updates and 2,304 rollouts per seed before skipped updates.
- Checkpoint and evaluation cadence are recorded by actual optimizer update, not nominal step count.

## Frozen reward contract

Baseline task reward:

```text
R = F * A * (0.5 + 0.5 * G)
```

- `F`: accepted terminal submission.
- `A`: verified answer correctness.
- `G`: validated citation/derivation support.
- Numeric zero is a valid value.
- Numeric/directional contradictions veto answer correctness.
- Missing or invalid finish receives zero task reward; ungated answer quality is logged diagnostically.
- Infrastructure failures are invalid samples, not incorrect answers.
- DeepSeek resolves only unresolved semantic cases; ambiguity, insufficient evidence, and provider failure are unresolved and logged.
- Optional retrieval shaping is tested only as a matched ablation after the baseline, with fixed `0.1` weight and evidence-linked support.

## Required metrics

Every condition reports valid finish, correct-and-finished, grounded success, mean A/G, unresolved counts, infrastructure failures, stop reasons, visible evidence support, tool mix, calculator use on derived tasks, repeated calls, turns, raw/visible tool lengths, truncation, judge coverage/errors, group variance, retained groups, trained tokens, KL, checkpoints, wall time, and observed cost where available.

All scheduled questions remain in task-success denominators. Repeated rollouts are aggregated at question level for evaluation. Paired bootstrap uses 10,000 question-level resamples with a fixed seed; document-cluster sensitivity is also reported.

## Claims policy

Do not compare historical reward means across incompatible evaluators. Do not claim causality from gold-page recall or reward correlation. If R1 does not improve B1, report no GRPO improvement and frame the result as a reproducible systems/negative-result study.
