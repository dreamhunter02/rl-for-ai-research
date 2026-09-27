# Claude Opus verification — Nemotron GRPO results

Date: 2026-09-27

- Endpoint: NVIDIA Inference Hub (`https://inference-api.nvidia.com/v1`)
- Model: `azure/anthropic/claude-opus-4-6`
- Inputs: `results/tinker_runs/nemotron35_lightning_grpo_grounded_full_v11_20260927/FINAL_REPORT.md`, the corrected base and trained 42-row evaluation JSON files, checkpoint manifest, training metrics, and reward-logic summary.

## Verdict

**PASS_WITH_CAVEATS.** Opus independently recomputed the reported aggregates and found them correct:

- Base: `16.70 / 42 = 0.397619`, 17 strong answers.
- Trained: `18.70 / 42 = 0.445238`, 19 strong answers.
- Paired gain: `+0.047619`; 7 improvements, 5 regressions, 30 unchanged.
- Training: 27 logged steps, 864 trajectories, six expected checkpoints including final, stable sampled KL approximately `0.005–0.009`, and no evidence of truncation.

## Caveats added by the audit

- The gain is not statistically significant by a paired sign test: 7 improvements versus 5 regressions gives approximately `p = 0.39` two-sided.
- 23/42 held-out cases received zero reward (`54.8%`), so the model remains weak despite the modest aggregate gain.
- Groups with constant rewards contribute no GRPO advantage signal; this is wasted compute, not a reward-correctness bug.
- The grader is deterministic rather than learned; the audit found no obvious leakage or reward-logic defect from the supplied evidence.
