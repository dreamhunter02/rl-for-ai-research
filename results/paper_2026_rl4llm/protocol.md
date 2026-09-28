# Frozen workshop protocol

Code commit: `61cd93a8048cb8d944d1da0ac1d27aaa592547fc`
Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
Renderer: `nemotron3_ultra`
Scorer: `workshop-v1`
Evaluation status: **previously inspected** 42-question split, not an untouched test.

## Conditions

- B0: original harness reference at `361d3f2337ed55576c241866353e4a9b72a9e9b6`; diagnostic unless common receipts/adapter exist.
- B1: corrected harness, base model, frozen prompts/tools/budget.
- R1: corrected harness with GRPO, restarted from base, selected only on dev.

## Frozen questions and documents

- Train: 96 IDs (`train96`)
- Dev: 12 IDs (`dev12`)
- Evaluation: 42 IDs (`eval42`)
- Train/dev document overlap: 8 documents: 3M_2018_10K, 3M_2022_10K, 3M_2023Q2_10Q, ACTIVISIONBLIZZARD_2019_10K, ADOBE_2022_10K, AES_2022_10K, AMAZON_2017_10K, AMCOR_2023_10K
- Train/eval document overlap: 0
- Dev/eval document overlap: 0

The split manifest is authoritative and is preserved without reallocation. The 42-question evaluation split is never used in prompts, reward weights, LR selection or checkpoint selection.

## Frozen budgets and selection

- Training: three epochs, batch 8, group 8, nominal pilot cap 10 batches per LR, final cap 36 nominal batches for 96 train questions; actual optimizer updates come only from `optimizer_audit.jsonl`.
- Rollouts: max 8 turns, max 1,024 generated tokens, train temperature 1.0, evaluation temperature 0.2, LoRA rank 32.
- Selection: update-zero B1 and matched dev checkpoints; grounded success first, correctness second, lower cost third; base is a candidate. Final R1 restarts from base.
- Evaluation: one rollout per scheduled question, fixed policy, no best-of-k; SDK sampling seed support is recorded rather than assumed.

## Scoring

`R = F * A * (0.5 + 0.5 * G)`. F is accepted terminal finish, A is fully correct typed answer, G is visible receipt-backed evidence and reviewed derivation support. Correct-and-finished and grounded-success binaries remain separate from mean A/G. All scheduled questions stay in denominators; failed/unresolved exits contribute zero to success and unresolved status is reported separately.

## Labels and evidence

`targets.draft.json` is not used. `targets.reviewed.json` has 150 entries and 189 exact indexed-page support spans produced by the automated source-span audit. This is not claimed as human semantic review; target values/types/precision inherit the recorded FinanceBench answers and remain a limitation. Credentials are inherited ephemerally and never written.

## Conflicts

See `protocol_conflicts.md` for the patch-base, split-count, overlap, SDK, judge, B0 and target-review conflicts and resolutions.
