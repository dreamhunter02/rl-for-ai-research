# Calculator-free harness and reward repair — 2026-09-30

Implemented in the local feature checkout; tested using an isolated copy on
SparkyOne. The production remote checkout was not overwritten. Changes remain
uncommitted because the checkout contains overlapping pre-existing user work.
No training, new policy evaluation, eval42 tuning, checkpoint deletion or Tinker
run occurred. All results below are post-hoc rescoring of existing trajectories.

## Tools and reward

Active tools: bm25_search, grep_document, search_tables, read, read_table, finish.
Calculator is absent from active schemas, prompts and final SFT examples. Legacy
calculator interpretation remains for historical diagnostics. Original dev traces
still display their calculator events; calculator results are not source evidence.
Read-table is explicitly a page-window convenience, not independent table parsing.
Grep truncation has continuation metadata; statement classification and explicit
company aliases are repaired. Tool errors, redundant calls and output volume are
diagnostics, not new reward bonuses. No tool-diversity incentive was added.

Reward: R = F × A × (0.5 + 0.5G). Mixed A = 0.6N + 0.4S;
numeric-only A = N; semantic-only A = S. Applicability is frozen from the question
and reference before grading a candidate. Source receipts authenticate evidence;
the judge assesses numeric result, semantic conclusion and claim support.
Malformed judgments/provider failures remain unresolved with null reward, not
valid negative examples. Configuration identity prevents stale-run resumption.

## Saved dev12 comparison

| Metric | Base old → repaired | SFT old → repaired |
|---|---:|---:|
| Mean reward | 0.291667 → 0.344444 | 0.375000 → 0.590377 |
| Correct | 4/12 → 4/12 | 6/12 → 7/12 |
| Grounded success | 3/12 → 3/12 | 3/12 → 4/12 |
| Unresolved, repaired | 0/12 | 0/12 |

All 12 unique frozen dev IDs appear once per phase. Primary judge:
openai/openai/gpt-5.6-terra. Independent second judgment:
nvidia/zai-org/glm-5.3-flash. Both were called through Inference Hub with the same
candidate-blind rubrics. N/S/A agreement is 24/24; exact G agreement is only 15/24.
This is agreement evidence, not proof that either judge is correct. Fractional
grounding remains insufficiently calibrated for an unattended GRPO run.

Examples: SFT 00807 now receives N=1, S=0, A=.6, G=.5, R=.45:
the .96 ratio is correct but the healthy-liquidity conclusion is wrong.
SFT 02987 and 04735 recover full credit for valid answers/evidence without a
calculator requirement. Wrong scale in the base 04735 remains wrong. SFT 00499
falls from .5 to .342857 for missing metrics/unsupported facts. Thus the revision
does not simply increase every reward. Wrong 07507 and 00540 remain zero.

`comparison.html` contains all original events, old/new components, judge reasons
and second-judge differences. Structural validation found 12 question articles and
24 trace panels; no browser-rendering verification is claimed. JSONL sidecars and
`summary.json` preserve the machine-readable scores and source hashes.

## Teacher migration and export

Artifacts: `../../teacher_traces/calculator_free_v4_20260930/`.

7,783 input rows reconcile to 1,562 migrated candidates, 1,545 quarantined,
4,469 duplicates and 207 outside the training split. Original files are untouched;
input hashes and per-line outcomes are in manifest.json and lineage.jsonl.
Ambiguous calculator dependencies, invalid finishes, unresolved targets and
unrepairable tool actions are quarantined; finishes and answers are not fabricated.
Delivered source text for judged candidates was checked against indexed pages.
These remain edited historical trajectories, not native new-harness rollouts.

A balanced 429-candidate shortlist was judged; 1,133 alternatives remain retained
but unjudged and are not SFT-ready. 272 candidates across 78 questions received
F=A=G=1. Four judgments remained HTTP 429 unresolved even after explicit GLM
fallback: 00651, 01244, 01981, 01328. They are excluded from exports.

Use **sft_identified_teachers.jsonl**, not the preliminary sft_train.jsonl:
the final export has **247 examples across 73/96 training questions**, excluding
10 explicitly synthetic gold-recovery rows and 15 unknown-teacher rows. Identity
labels are not independently authenticated model provenance. Existing repair
lineage is retained in clean.jsonl; do not describe these as untouched teacher
rollouts. There are not yet 500 clean examples or full train96 coverage.

## Verification and remaining boundary

Final isolated SDK test suite: **145 passed, 5 skipped, 43 subtests passed**.
The five skips are opt-in live controls; all five passed against Inference Hub,
then passed from cache using the final scorer. Local focused tests also passed;
the full SDK suite required the remote configured environment.

Independent code review found four important issues, all fixed with regressions:
accepted-finish mismatch, skipped malformed evidence, incomplete grounding output
validation, and incomplete judge/config resume identity. Export verification checks
paired tool calls, no calculator schema/calls, F=A=G=1, resolved scores and train-only
IDs. No claim is made that all legacy teacher data can be safely repaired.

Before GRPO: calibrate the nine grounding disagreements and validate native
new-harness rollouts. This implementation does not establish a training gain.
