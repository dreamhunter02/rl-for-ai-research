# Calculator-Free Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove calculator dependence, apply the approved component reward, rescore 24 dev records, and migrate teacher data without fabricating behavior.

**Architecture:** Keep the legacy scorer for historical interpretation. Add a shared versioned rubric/judge/scoring module used by live runners and offline rescoring. Produce immutable-source migrations and score sidecars before exporting usable SFT data.

**Tech Stack:** Existing Python harness, pytest, configured OpenAI-compatible judge transport, JSONL, standalone HTML.

**Spec:** `docs/superpowers/specs/2026-09-30-calculator-free-reward-design.md` (approved by user).

## Global Constraints

## Execution status (final, 2026-09-30)

This checklist supersedes the original per-task command/file suggestions below;
those remain as the design history, not claims that every suggested command ran.

- [x] Shared component judge and calculator-free teacher/RL tool interfaces implemented.
- [x] Retrieval metadata, statement classification, aliases, resume identity and tool diagnostics repaired.
- [x] All 24 saved dev traces rescored; two Inference Hub judges compared; standalone HTML structurally verified.
- [x] Teacher inventory migrated non-destructively; 429 balanced candidates judged; identified-teacher SFT export verified.
- [x] Independent code review fixes and regression tests completed: 145 passed, 5 opt-in skips; 5 live controls passed separately.
- [ ] Git integration: intentionally pending user choice; overlapping pre-existing edits are not staged, committed, merged or pushed.

Amendments: HTML rendering and comparison tests live in `rescore_traces.py` and
`test_rescore_traces.py`, not a second comparison module. Final report is
`results/trace_comparisons/reward_v4_20260930/COMPLETION.md`; HTML is `comparison.html`
in that directory. Historical observations are preserved, not regenerated.
Migration inventories all inputs but judges a bounded one-per-question/teacher
shortlist; 1,133 other candidates remain explicitly unjudged. Four provider-rate-
limited judgments remain unresolved and are excluded from SFT. Unknown-teacher
and synthetic gold-recovery records are excluded from the final identified-teacher
export. No claim of 500 clean traces or complete train96 coverage is made.

## Constraints retained

- R = F * A * (0.5 + 0.5 * G); mixed A = 0.6 * N + 0.4 * S; numeric-only A = N; semantic-only A = S.
- Non-applicable components are null, not zero. Weights are a starting protocol, not a validated optimum.
- No training, new model evaluation, eval42 tuning, artifact deletion, or paid Tinker runs are in scope.
- Preserve original evidence, the frozen 96/12/42 split, and existing adapters.
- Write new artifacts only. Preserve existing dirty/untracked work; use explicit-path staging and exclude pre-existing changes from commits.
- Endpoint and credential configuration stays in environment variables; no secrets or private deployment identifiers in public artifacts.
- Do not change thinking mode, turn limits, or token budgets as part of rescoring.

## Review Focus

Execution authorized by user after tool review. Implement natively in the existing
feature checkout to preserve its required untracked harness/data. Keep all six
non-calculator tools; explicitly describe read_table as a page-window convenience.
Task 3 additionally produces tool-use diagnostics (counts, errors, duplicate calls,
delivered characters) without adding unvalidated process-reward weights. Judge
through the user's Inference Hub endpoint using environment configuration.

- A candidate omits its semantic conclusion: applicability must remain mixed (Task 1).
- Evidence payload is truncated or judge output malformed: unresolved, not zero or unsupported success (Task 1).
- A historical finish was rejected: rescoring must not repair F (Task 3).
- Calculator and retrieval share one assistant message: preserve the retrieval call and matching observation (Task 5).
- Resume sees the same ID under different judge settings: reject stale cache/config reuse (Tasks 1 and 3).

## Task 1: Shared component scoring

**Files:** Create `harness/component_reward.py`, `tests/test_component_reward.py`; modify `harness/reward_calculation.py` only to reuse/extract provider transport without changing legacy semantics.

**Interfaces:** `aggregate_reward(*, F: int, N: float | None, S: float | None, G: float, mode: str) -> dict`; `async score_episode(*, question: str, rubric: dict, submission: dict | None, receipts: dict, F: int, judge: object) -> dict`. Judge protocol: `async judge_components(payload: dict) -> dict`. New scorer version: `workshop-rubric-v4-components`.

- [ ] Write tests asserting mixed F=1,N=1,S=0,G=1 yields A=.6,R=.6; numeric-only N=1,G=0 yields .5; F=0 yields 0; missing applicable components, NaN/out-of-range values, malformed judgments, and incomplete evidence produce unresolved/null reward. Candidate omission cannot change rubric mode. Add wrong-final-number and fabricated-citation controls.
- [ ] Run `PYTHONPATH=harness pytest tests/test_component_reward.py -q`; confirm failures demonstrate missing implementation.
- [ ] Implement fixed question/reference rubrics, claim-level aggregation, authentic-receipt evidence assembly, structured judge transport and versioned cache keyed by complete inputs/configuration. Numeric and semantic claims are graded separately; unsupported grounding claims receive no credit. Store raw judgments/reasons. Preserve separate strict correctness and unresolved fields.
- [ ] Run the test file with a fake judge; assert model/prompt/evidence changes invalidate cache and an API failure never yields a valid zero-reward example.
- [ ] Commit only task-owned changes after inspecting the diff; record any edits left uncommitted to avoid including prior user changes.

## Task 2: Calculator-free live interface

**Files:** Modify `harness/finance_env.py`, `harness/teacher_runtime.py`, `harness/generate_teacher_traces.py`, `harness/prepare_sft_dataset.py`; add `tests/test_calculator_free_harness.py`; extend `tests/test_teacher_runtime.py`.

**Interfaces:** Live tool set is `bm25_search, grep_document, search_tables, read, read_table, finish`. Existing runner entry points stay stable and delegate rewards to Task 1. Numeric finish permits optional `derivation: str`; `calc_id` is not required or exposed in the new schema.

- [ ] Write tests asserting every active tool list excludes calculate, unknown calculate calls fail explicitly, numeric finish accepts concise derivation without calc_id, and teacher/RL consumers return identical Task 1 scores for the same fixture. Test Responses output remains a list after executing a tool.
- [ ] Run `PYTHONPATH=harness pytest tests/test_calculator_free_harness.py tests/test_teacher_runtime.py -q` and inspect expected failing assertions.
- [ ] Remove calculator exposure/instructions in all registrations; retain legacy interpretation code. Connect both scorers to Task 1, keep unresolved-group exclusion explicit, align core prompts and record remaining runner-specific behavior in the manifest. Fix the Responses variable collision without changing generation budgets.
- [ ] Rerun both test files and existing preparation tests. Verify tool schemas and runner prompts contain no calculator requirement.
- [ ] Review/stage only task-owned changes; preserve pre-existing finance_env/generator edits.

## Task 3: First delivery — 24 dev rescored records and HTML

**Files:** Create `harness/rescore_traces.py`, `tests/test_rescore_traces.py`, `tests/test_dev12_comparison.py`; modify `harness/make_dev12_comparison.py`; create versioned artifacts under `results/trace_comparisons/` and `results/reward_judgments/`.

**Interfaces:** `rescore_record(record: dict, *, rubric: dict, judge: object) -> dict` is async and delegates to Task 1. Sidecar fields include source SHA256, financebench_id, model/phase, original_score, new_score, scorer/judge configuration, and raw judgment reference. CLI accepts explicit input/output, split, rubric and judge config; HTML accepts before/after plus matching sidecars.

- [ ] Write tests asserting no source mutation, F=0 remains 0, exact 12 unique frozen dev IDs per model, sidecars reject source/config hash mismatches, and HTML contains F/N/S/A/G, old/new rewards, reasons, unresolved flags and escaped complete traces. Test IDs 00807,00684,02987,04735,00499,06655 using evidence-based fixtures rather than expected live-judge scores.
- [ ] Run `PYTHONPATH=harness pytest tests/test_rescore_traces.py tests/test_dev12_comparison.py -q`; confirm red.
- [ ] Implement receipt reconstruction from actual delivered observations, historical accepted-finish preservation, bounded judge retries/cache, configuration-safe resumption, and versioned HTML generation. Original calculator calls stay visible as historical behavior; they are not source receipts.
- [ ] Rerun tests. Rescore the two saved dev12 JSONL files through the configured endpoint, verify 24 records and report unresolved cases. Generate `results/trace_comparisons/qwen35_4b_base_vs_sft_dev12_reward_v4.html`; inspect rendered or structurally validated output, labeling verification level. Show the HTML to the user before corpus-wide migration.
- [ ] Commit owned implementation files and safe manifests; never stage secrets/raw deployment configuration. Report this as post-hoc scoring, not a new model evaluation.

## Task 4: Bounded retrieval and diagnostics repairs

**Files:** Modify `harness/financebench_harness.py`, `harness/observations.py`, `harness/finance_env.py`, `harness/eval_current_harness.py`; create `tests/test_harness_retrieval_repairs.py`; extend `tests/test_eval_current_harness.py`.

**Interfaces:** Preserve existing retrieval function signatures. Observation metadata must truthfully expose truncation and continuation; alias matching uses an explicit mapping. Resume identity includes scorer/prompt/tool/budget/model settings, not question ID alone.

- [ ] Write fixtures asserting plural income-statement recognition, incidental notes headings do not win classification, AES alias matching without unrelated-company matches, grep truncation visibility/continuation, and neighbor metadata does not claim unseen text. Add config-mismatch resume rejection.
- [ ] Run `PYTHONPATH=harness pytest tests/test_harness_retrieval_repairs.py tests/test_eval_current_harness.py -q`; confirm specific failures.
- [ ] Implement bounded classifier/alias corrections, truthful observation metadata and actionable field errors. Add explicit configuration fingerprints and generation stop/usage metadata where returned by the provider. Do not invent missing fiscal metadata or replace retrieval ranking wholesale.
- [ ] Rerun tests plus affected existing retrieval/observation tests. Confirm Task 3 historical source observations remain unchanged.
- [ ] Review and commit only owned changes, documenting deferred broader tool redesign.

## Task 5: Provenance-preserving teacher migration and clean export

**Files:** Create `harness/migrate_teacher_traces.py`, `tests/test_migrate_teacher_traces.py`; modify `harness/curate_sft_traces.py`, `harness/prepare_sft_dataset.py`; extend their tests. New outputs under a versioned directory in `results/teacher_traces/`.

**Interfaces:** `migrate_record(record: dict) -> dict` returns `{status, record, edits, reasons}`; status is `candidate` or `quarantined`. `build_manifest(inputs: list[dict], outcomes: list[dict]) -> dict` reconciles hashes, counts, lineage, duplicates, and train-ID coverage. Candidate replay and scoring consume Tasks 1/2/4.

- [ ] Write tests for mixed tool-call turns, orphaned messages, missing finish, later reliance on removed calculator results, residual tool errors, duplicate lineage, unchanged originals, and dev/eval ID exclusion. Assert no output claims calculator-derived reasoning was model-generated; unrepairable records are quarantined rather than silently dropped.
- [ ] Run `PYTHONPATH=harness pytest tests/test_migrate_teacher_traces.py tests/test_curate_sft_traces.py tests/test_prepare_sft_dataset.py -q`; confirm targeted failures.
- [ ] Inventory local/remote teacher artifacts and actual SFT input with hashes. Implement paired removal, schema/prompt migration and edit logs; quarantine dependence/ambiguity. Replay available source receipts, validate finishes, rejudge eligible candidates, and select clean high-quality candidates rather than first matches. Preserve all source files and reconcile every input outcome.
- [ ] Rerun targeted tests and `PYTHONPATH=harness pytest tests -q` in the configured environment, reporting dependency blockers accurately. Execute migration only on frozen training IDs. Verify usable export has no calculate schemas/calls, unresolved rewards or failed-tool actions; report coverage, duplicates, quarantine and judge failures. No training launch.
- [ ] Review final diff and publish manifest, clean-export paths, test results and the updated HTML link. Commit only owned changes and update execution checkboxes to actual state.

## Execution handoff

Authorized: native execution in this session, with an independent final review.
Tasks share scorer and trace contracts heavily; sequential implementation avoids
conflicting interface changes. User requested implementation while away; do not
pause for intermediate approval of these requested plan amendments.
