# FinanceBench review repairs implementation plan

> Execute inline with systematic debugging, regression tests and one independent final review.

**Goal:** deliver an applicable patch against f29df1a that closes the September 27 review findings.
**Architecture:** reuse the previously tested workshop evaluator with per-episode visible evidence receipts, typed targets and one environment for training/evaluation. Preserve historical artifacts and the existing question split; explicitly supersede unsupported gate/accuracy claims.
**Tech stack:** Python, unittest, pinned Tinker cookbook, Decimal.
**Spec:** user-requested fixes and NEMOTRON_FINANCEBENCH_GRPO_EXECUTION_BRIEF-3.md.

## Constraints and review focus
- Gold targets stay evaluator-only and require source verification before paid runs.
- Never count tool echoes or failed calls as retrieved evidence or accepted finish.
- Preserve all scheduled questions in reporting denominators, including parse/length exits.
- Hold evaluation questions out of checkpoint selection and preserve frozen split IDs.
- No live API run is authorized by this patch request; distinguish offline tests from SDK validation.

## Tasks
1. Snapshot exact upstream bytes; reproduce numeric errors, empty-call-ID grounding and pagination failures in tests/test_review_regressions.py. Record RED output.
2. Integrate existing tested workshop reward, receipts, observation budgets, epochs, strict group filtering and matched evaluation; retain active-document loading and existing split IDs. Run tests to GREEN.
3. Repair compatibility entrypoints/configs, require reviewed targets and final dev-checkpoint evaluation, correct historical denominator/claim documentation. Add tests for each newly found defect.
4. Run all available tests, inspect unavailable dependency/corpus failures, independent code review, verify clean application to upstream bytes, package patch and concise handoff.

## Ledger
- Baseline: a289116 is an exact-byte snapshot of relevant f29df1a files; hashes verified against GitHub blob IDs.
- Ruling: reuse the already implemented and reviewed workshop modules rather than rewrite their validated logic; rebase against the user's newer files and preserve their split IDs and active-document optimization.
- Independent review found nonobject tool arguments, malformed citation IDs, target/submission scale mismatch and initial-overflow denominator loss. Each was reproduced in a failing regression and fixed; the SDK-specific malformed-call check remains explicitly skipped until the pinned runtime is installed.
- Root suite: 54 tests, 46 passed and 8 SDK skips. Legacy suite: 28 tests, 19 passed and 9 skips. Compilation and whitespace checks passed. Actual upstream split preparation preserved 96/12/42 IDs.
- Scheduling decision: bound nominal batches and report actual optimizer calls explicitly. Checkpoint cadence remains nominal; this deviation from actual-update scheduling is documented in the runbook, and skipped batches never count as optimizer updates.
- Delivery: one patch against exact upstream bytes, with handoff and paper runbook. No live provider evaluation, training or GitHub publication performed.
