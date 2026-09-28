# FinanceBench/Nemotron workshop evidence package

This package records the corrected protocol through the first scientific-validity gate. It does not claim a GRPO improvement.

## Reproduce frozen preparation

```bash
export FINANCEBENCH_ROOT="$PWD"
export FINANCEBENCH_SPLIT="$PWD/results/paper_2026_rl4llm/split_frozen.json"
export FINANCEBENCH_TARGETS="$PWD/results/paper_2026_rl4llm/targets.reviewed.json"
export FINANCEBENCH_PDF_PAGES=1
PYTHONPATH=harness python harness/workshop_prepare.py preflight --split "$FINANCEBENCH_SPLIT" --targets "$FINANCEBENCH_TARGETS" --corpus --out results/paper_2026_rl4llm/preflight.json
PYTHONPATH=harness python harness/workshop_scorer_audit.py --out results/paper_2026_rl4llm
PYTHONPATH=harness python harness/capability_tests.py
```

The B1 development smoke command and the one-update provider-smoke command are recorded in `run_manifest.json`; credentials are inherited ephemerally and never printed. Raw traces and audits are under `B1-dev-smoke/` and `provider_smoke_lr1e5_v3/`.

## Interpretation

The 12-question dev smoke is not the final 42-question evaluation. The provider smoke made zero actual optimizer updates because all eight sampled training groups were zero-reward/all-equal; the corrected gate therefore blocks the paid pilot and R1. See `completion_report.md`, `protocol_conflicts.md`, `measured_summary.json`, `table1_main.csv`, `table2_failures.csv`, `table3_scorer.csv`, `table4_paired.md`, and `figure1_learning.blocked.md`.
