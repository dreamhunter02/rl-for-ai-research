# Experiment record: Nemotron 3.5 Lightning FinanceBench GRPO Reward v2

- Canonical experiment name: `nemotron35_financebench_grpo_reward_v2_full_seed0_20260927`
- Display name: Nemotron 3.5 Lightning — FinanceBench GRPO Reward v2 full run
- Status: completed successfully
- Date: 2026-09-27
- Repository commit at launch: `94df6b0876f756c4ea4fb9de697e262ee41c21d7`
- Run directory: `results/tinker_runs/nemotron35_lightning_grpo_reward_v2_full_20260927/`

## Training configuration

- Student model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
- Tinker renderer: `nemotron3_ultra`
- Dataset split: `train`
- Unique questions: 108
- Optimizer steps: 27
- Questions per batch: 4
- Rollouts per question: 8
- Total rollout trajectories: 864
- Maximum turns per rollout: 8
- Learning rate: `1e-5`
- Seed: `0`
- Checkpoint interval: every 5 steps plus final
- Constant-reward group removal: enabled
- Evaluation during training: disabled (`EVAL_EVERY=0`)
- Semantic judge: `deepseek-ai/DeepSeek-V4.1-Flash` through DeepInfra
- Judge credential source: GNOME Keyring service label `DEEPINFRA_API_KEY`; the secret was retrieved ephemerally and was not written to this record
- Judge confidence threshold: `0.85`
- Finish gate: required
- Grounded reward: `clip(answer_quality * (0.5 + 0.5 * evidence_quality), 0, 1)`

## Verified outcome

- Process exit: `0`
- Tinker completion marker: present
- Traceback in console log: none
- Metric batches: 27
- Rollout summaries: 864
- Valid finish trajectories: 220/864 (`25.5%`)
- Zero-total-reward trajectories: 616/864
- Negative process-penalty trajectories: 61/864 at `-0.1`
- Positive-total-reward trajectories: 187/864
- Mean authoritative total reward: `0.1475228666`
- Sum of authoritative total rewards: `127.4597568`
- DeepSeek cache records: 32
- Judge-error metric: zero observed
- Low-level trace events: 38,760
- Checkpoint records: 6

The rollout total-reward distribution and all per-step metrics are authoritative. The known Tinker serialization defect remains: `final_reward` is zero in the rollout summaries, so use each trajectory's `total_reward`, final-step metrics, and `metrics.jsonl` instead.

## Checkpoints

- Step 5: `tinker://b68fd838-fc7b-581a-b1b9-91744fa09763:train:0/sampler_weights/000005`
- Step 10: `tinker://b68fd838-fc7b-581a-b1b9-91744fa09763:train:0/sampler_weights/000010`
- Step 15: `tinker://b68fd838-fc7b-581a-b1b9-91744fa09763:train:0/sampler_weights/000015`
- Step 20: `tinker://b68fd838-fc7b-581a-b1b9-91744fa09763:train:0/sampler_weights/000020`
- Step 25: `tinker://b68fd838-fc7b-581a-b1b9-91744fa09763:train:0/sampler_weights/000025`
- Final: `tinker://b68fd838-fc7b-581a-b1b9-91744fa09763:train:0/sampler_weights/final`

## Complete log and artifact index

- `console.log` — complete training console output
- `tinker_log/logs.log` — Tinker structured log stream
- `tinker_log/metrics.jsonl` — 27 per-step metric records
- `tinker_log/trace_events.jsonl` — 38,760 Chrome/Perfetto trace events
- `tinker_log/timing_spans.jsonl` — timing spans
- `tinker_log/experiment_manifest.json` — run identity, reward contract, and judge configuration
- `tinker_log/config.json` — complete Tinker recipe configuration
- `tinker_log/checkpoints.jsonl` — checkpoint paths and step numbers
- `tinker_log/code.diff` — captured code-diff metadata
- `tinker_log/iteration_000000/` through `tinker_log/iteration_000026/` — per-step rollout summaries, HTML viewers, and log trees
- `deepseek_judge_cache.json` — cached judge decisions; contains no credential

## Interpretation and next evaluation

This run validates the full reward-v2 training path and artifact capture, but its rollout signal is sparse: only 25.5% of trajectories reached a valid finish and the mean total reward was 0.1475. The 108-question dataset provides limited question diversity, but the immediate issue is reward/termination sparsity rather than a missing log or provider failure. The official 42-question holdout evaluation has not yet been run; it must be evaluated against the base model and the final checkpoint before claiming improvement.
