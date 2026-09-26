# Opus Review of the Teacher-Guided GRPO Plan

Reviewer: Claude Opus 4.6 via `azure/anthropic/claude-opus-4-6`
Date: 2026-09-26
Verdict: Conditional GO; revise the plan before training begins.

## Overall assessment

Opus agrees that the proposed method is sound if Qwen generates every rollout and the 91 teacher traces remain secondary references rather than replayed training data. It explicitly distinguishes this from SFT, offline RL, and behavior cloning.

The existing four-trajectory SGLang-to-Unsloth experiment should be described only as a reward/integration proof. It did not validate on-policy training dynamics because the rollout actions were reconstructed and were not accompanied by exact token IDs and old log-probabilities from the same local policy checkpoint.

## Required changes before the real run

- Rename the current Stage 0 concept to de novo local-policy verification; do not imply that the SGLang trial transferred on-policy training validity.
- Freeze the retrieval corpus, page cache/index, chunking configuration, tool schemas, and environment version for the entire run.
- Require the same local Qwen checkpoint to generate each action and provide its exact per-token old log-probability.
- Record sampling parameters per trajectory: temperature, top-p, top-k, seed, and any decoding constraints.
- Record per-token log-probabilities and per-trajectory KL from the initial/reference policy, not only sequence aggregates.
- Define MLflow as a derived view of the append-only JSONL artifacts, upload those JSONL files to MLflow, and validate MLflow aggregates against the JSONL after the run.

## Teacher-trace guidance

Opus recommends using the 91 traces first for reward validation only. Full trajectory similarity, token overlap, a fixed teacher tool-call path, and KL toward one teacher trace are unsafe because one trajectory is not a teacher policy distribution and may collapse exploration.

If teacher guidance is later enabled, measure how often primary reward ties occur first, then use a small tie-breaker only within equal-primary-reward groups. Ablate at least three tie-breaker magnitudes, verify that it never overrides answer/evidence correctness, and track tool-path diversity and per-question disagreement with the teacher.

The 91 traces are filtered from 349 candidates, so the run should report filtering statistics and check whether the 17 questions without strict traces are systematically harder or structurally different.

## Budget and evaluation changes

- A 20–30-question pilot is reasonable for mechanics, but sample it stratified across question types rather than randomly.
- Increase the monitoring set from 10 to approximately 15–20 questions, or report confidence intervals explicitly.
- Group size 4 is acceptable initially, but measure zero-variance groups and consider an ablation at group sizes 8 and 16 if memory allows.
- Clarify optimizer steps versus groups processed and include a GB10 memory budget estimate for 16k contexts.
- Retain the 1–3 pass cap and early stopping; also monitor KL from the initial checkpoint and stop if policy drift or diversity collapse becomes excessive.

## Recommended decision

Proceed only after the exact same-policy local rollout/log-probability gate, retrieval-index freeze, and trace-schema additions are implemented. Start with a teacher-free answer/evidence GRPO pilot using the existing 91-question coverage for reward validation; enable the teacher tie-breaker only after measuring ties and diversity, and report pure-GRPO versus teacher-guided results under identical seeds and questions.
