# Nemotron 3.5 Lightning Tinker GRPO — Final Report

Date: 2026-09-27

## Run

- Base model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
- Renderer: `nemotron3_ultra`
- Backend: Tinker GRPO
- Split: FinanceBench train (108 questions)
- Configuration: 27 batches, batch size 4, group size 8, learning rate `1e-5`, maximum 8 turns, seed 0
- Rollouts: 864 trajectories (27 × 4 × 8)
- Final sampler checkpoint: `tinker://d4f45079-50ca-56ce-8ed5-dc4fc9986405:train:0/sampler_weights/final`
- Final state checkpoint: `tinker://d4f45079-50ca-56ce-8ed5-dc4fc9986405:train:0/weights/final`

The run exited normally with code 0. Full console output, metrics, checkpoint manifest, rollout summaries, log trees, trace events, timing spans, and HTML rollout views are in this directory's `tinker_log/` tree.

## Reward validation

The reward is a deterministic FinanceBench grader, not a learned neural reward model. It scores answer correctness and evidence grounding while leaving finish/formatting out of answer quality. The reward and Tinker integration tests passed; the exact test commands and source are retained in the repository.

A prior `0.2131` evaluation number was invalid: 27 of 42 rows were zeroed by an inference-evaluation context-overflow bug. The evaluator was corrected to apply the same 4,000-character observation cap used during training and to terminate cleanly on `finish`; the corrected 42-question run had zero harness errors.

## Held-out evaluation

Both checkpoints were evaluated with the corrected evaluator, the same renderer, the same 42-question eval split, and the same 8-turn limit.

| System | Mean reward | Strong answers (>=0.9) | Nonzero answers | Harness errors |
|---|---:|---:|---:|---:|
| Base Nemotron | 0.3976 | 17/42 | 17/42 | 0 |
| Trained final sampler | 0.4452 | 19/42 | 19/42 | 0 |

Paired against the same question IDs, training improved 7 cases, regressed 5, and left 30 unchanged; mean paired gain was `+0.0476`.

## Conclusion

The full Tinker GRPO run is complete, reproducible, and the final checkpoint is usable. It produced a modest but positive held-out gain under the corrected protocol; this is a successful feasibility run, not evidence that the training recipe is solved. The five regressions and the remaining 23 zero-reward cases should be addressed before scaling the run or treating the checkpoint as a production model.

## Independent Opus verification

Claude Opus 4.6, called through NVIDIA Inference Hub, independently recomputed the aggregates and returned `PASS_WITH_CAVEATS`. It confirmed the arithmetic, checkpoint/trajectory counts, stable KL range, and lack of silent truncation. It also noted that 7 improvements versus 5 regressions is not statistically significant (two-sided paired sign test approximately `p=0.39`) and that 23/42 held-out cases still receive zero reward. Full audit: `results/reward_audits/opus_grpo_verification_20260927.md`.
