# Calculator-free FinanceBench harness and reward migration

Status: approved; user authorized tool review, plan amendments, and implementation.

## Tool-review amendment

Keep six tools: bm25_search discovers documents/passages; grep_document locates
terms inside a known document; search_tables prioritizes table-like pages; read
delivers source windows; read_table resolves table IDs to page windows with
metadata (not a parsed table); finish submits the answer. Keep table shortcuts
until removal ablations justify merging them. No reward for tool diversity.

Record call validity, errors, repeated identical calls, total calls and delivered
characters separately from outcome reward. These are diagnostics, not an added
efficiency reward yet: a bounded search miss is not a model error, and successful
grounded task completion must precede cost optimization. Use the same-question
successful trajectories for future efficiency comparisons. Provider failures must
not be charged as malformed model calls. The authorized judge is an Inference
Hub model configured through environment variables, never a hard-coded secret.

## Outcome and constraints

Remove the calculator from the active agent interface and curated teacher data;
reward final numerical correctness, semantic correctness, and grounding separately.
First produce a post-hoc comparison for all 24 existing dev trajectories (12 base,
12 SFT), then finish migrating the teacher corpus. Preserve original evidence,
the frozen 96/12/42 split, and existing adapters. No training, new model evaluation,
eval42 tuning, artifact deletion, or paid Tinker runs are in scope.

Existing audit: `results/trace_comparisons/dev12_harness_audit_20260930.md`.
Existing modified and untracked files belong to the user and must be preserved.

## Approach

Recommended: version the new scoring contract, use one shared judge path across
RL, teacher generation, and offline rescoring, and migrate data into new files.
This prevents different runners from rewarding different behavior.

Alternatives: patch only the existing deterministic scorer (less code but retains
the extraction/provenance failure modes); regenerate all teachers immediately
(clean native trajectories but additional inference cost before validating rewards).
Neither is the selected approach.

## Reward contract

For resolved judgments:

    R = F * A * (0.5 + 0.5 * G)
    A = 0.6 * N + 0.4 * S   # mixed numeric and semantic questions
    A = N                   # numeric-only questions
    A = S                   # semantic-only questions

F is binary: an accepted, structurally valid finish. N, S, G are in [0,1].
Question/reference rubrics determine which dimensions apply before the candidate
is judged; a candidate cannot evade semantic grading by omitting its conclusion.
Non-applicable components are null, not zero. Weights are a starting protocol,
not a validated optimum.

- N evaluates requested final numerical results, including units, scale, sign,
  and reference-appropriate rounding. Correct inputs without the requested final
  result do not earn final-number credit. Credit across multiple requested results
  is the fraction correct under the fixed rubric.
- S evaluates required claims and conclusions. For mixed questions this includes
  conclusions such as whether a ratio is healthy. Material contradictions count
  against the corresponding claim; verbosity earns no credit.
- G evaluates whether evidence actually available to the agent supports the final
  claims and any necessary derivation. Equivalent evidence on another page is
  acceptable. A calculator call, exact gold quote, exact metric string, or exact
  expression syntax is not required.

Use structured LLM judgments with claim-level decisions, reasons, and cited
receipts. Aggregate scores from those decisions, rather than accepting an
unexplained floating-point rating. Judge confidence is diagnostic, not reward.
Treat question text, candidate text, and document evidence as untrusted inputs.

Code checks schema validity and receipt authenticity; it does not determine
numerical correctness through brittle answer extraction. Invalid or invented
citations cannot support grounding. They do not secretly redefine F or A.
API errors, malformed judgments, and unresolved ambiguity are explicit unresolved
outcomes with null reward, never silently incorrect answers. Training excludes
unresolved groups and reports counts. Strict benchmark correctness remains
separate from fractional training reward.

Store the rubric, model, prompt/scorer versions, input hashes, raw judgment,
component scores, and reasons. Cache keys include the complete scoring inputs
and configuration. Share this scoring implementation across all consumers.

## Active harness changes

Remove `calculate` from exposed tools, prompts, registrations, and SFT tool schemas.
Do not require `calc_id` for finish. Permit concise derivation text in a numeric
submission without requiring it for direct extraction. Preserve legacy artifacts
and legacy scoring code sufficiently to interpret old runs.

Bounded retrieval repairs from the audit:

1. Recognize plural income-statement headings and avoid classifying a notes page
   solely from an incidental balance-sheet mention.
2. Support explicit, tested company aliases such as AES/AES Corporation; avoid
   unrestricted fuzzy matching that could mix companies.
3. Make truncated grep/read observations truthful, with usable offsets or a
   continuation instruction; do not claim neighbor content was delivered when
   only page numbers were returned.
4. Make errors identify the failed field and correction. Align teacher/evaluation
   and RL task instructions, and record configuration fingerprints for resumption.

Do not introduce a new retrieval engine, invent fiscal-period metadata, or change
turn/token budgets in this repair. Defer broader tool redesign for the requested
follow-up discussion. Fix the Responses-output variable collision if touching
that generation path; cover it with a regression test.

## First deliverable: 24 rescored dev traces and HTML

Use the saved base and SFT files already consumed by
`harness/make_dev12_comparison.py`. Verify exactly the same 12 frozen dev IDs in
each file. Preserve all original messages, calculator calls, submissions, and
scores in the historical records: rescoring cannot retroactively change behavior.

For old submissions, adapt the record for judging without changing the answer or
fabricating a finish. Preserve historical F; diagnostics may judge an unaccepted
answer, but its reward remains zero when F=0. Calculator output alone is not
source evidence; underlying retrieved receipts may support the derivation.

Write versioned rescore sidecars and a new standalone comparison HTML with old
and new rewards, F/N/S/A/G, judge reasons, unresolved flags, and complete traces.
Keep all 12 questions in each model's denominator. Label this as post-hoc reward
repair, not improved model performance. Retain the historical baseline settings
caveat. Use an already configured authorized judge endpoint; do not print keys.

Regression examples include 00807 (correct 0.96, incorrect conclusion), 00684
(equivalent evidence on a different page), 02987/04735 (dimensionless ratios),
00499 (missing facts/unsupported figures), and 06655 (correct answer without
adequate final citations). Expected outcomes follow evidence, not a desire to
raise scores. Include fabricated-evidence and wrong-final-number controls.

## Teacher migration

Inventory local and remote teacher sources and the actual SFT input before
changing data. Record file hashes, source lineage, duplicate policy, and frozen
training-ID membership. Do not mix dev or eval trajectories into teacher training.

Write new artifacts only. Remove calculator tool calls and their paired tool
messages; preserve other calls in multi-call turns and validate pairing. Remove
obsolete calculator instructions and references from active schemas. Preserve
the original record and a machine-readable edit log for every migrated record.

Removing a tool result can invalidate later reasoning. Do not relabel a calculator
result as model-authored reasoning, fabricate source evidence, append invented
successful finishes, or replace wrong answers with gold answers. Mark edited
records as migrated, not native new-harness rollouts. Quarantine records whose
meaning or evidence depends on removed content and report why they need new
generation or explicit review. Preserve those originals; no trace is silently lost.

Replay/check remaining retrieval receipts where the corpus permits, validate
the finish under the new schema, rejudge eligible candidates, and reject unresolved
or invalid trajectories from the usable SFT export. Exclude residual failed-tool
actions from the clean export rather than teaching those mistakes. Rank clean
candidates before selecting per-question/per-teacher examples; do not simply
take the first eligible record. Report coverage and quarantined counts honestly.

## Verification and completion criteria

Use test-first changes for scoring, schemas, retrieval repairs, migration, and
HTML generation. Keep legacy-score regression tests separate from the new version.
Run applicable existing tests as well as the new tests; report failures rather
than suppressing them.

Completion requires: no calculator exposed by active runners or usable SFT data;
consistent scoring across consumers; all 24 dev records accounted for with
versioned judgments or explicit unresolved statuses; a downloadable standalone
HTML; and a teacher migration manifest reconciling every input with an output or
quarantine reason. Unresolved judge failures block claiming complete rescoring.

The design does not promise every old teacher trace can safely become training
data. Regeneration of quarantined traces is a separate, costed next action.
