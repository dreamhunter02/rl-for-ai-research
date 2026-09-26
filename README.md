# FinanceBench agentic GRPO

This repository contains the reproducible FinanceBench retrieval harness, teacher-trace artifacts, baseline evaluations, Unsloth/TRL smoke tests, and the minimal FinanceBench-to-LoRA optimizer-step bridge.

The corpus PDFs, extracted text, page-cache index, large Tinker run logs, and generated adapter weights are intentionally excluded from GitHub. They are local or regenerable artifacts; the tracked JSONL traces and metrics preserve the experiment evidence without publishing the corpus or a large binary checkpoint.

Key files:

- `harness/financebench_harness.py` — page-aware BM25/table retrieval, reads, calculations, provenance, and answer/evidence scoring.
- `harness/finance_env.py` — typed FinanceBench environment and reward integration.
- `harness/generate_teacher_traces.py` — multi-turn trace generation with finish handling, seeds, and grounded rewards.
- `harness/financebench_grpo_trial.py` — minimal harness-to-Unsloth optimizer-step bridge.
- `harness/unsloth_qwen4b_grpo_smoke.py` — Unsloth/TRL toy GRPO smoke test.
- `results/teacher_traces/` — teacher candidates, judgments, and strict selected traces.
- `results/local_eval/` — baseline and rollout evaluation JSONL artifacts.
- `GRPO_PROGRESS.md` — experiment record and next steps.
- `SFT_PLAN.md` — supervised warm-start plan retained for comparison, not a requirement for the current direct-GRPO study.

The current project status is documented in `GRPO_PROGRESS.md`; no claim of FinanceBench model improvement should be made from the toy smoke test or the one-step bridge proof alone.
