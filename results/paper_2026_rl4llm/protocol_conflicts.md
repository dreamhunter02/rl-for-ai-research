# Protocol conflicts and resolutions

Recorded before any new provider evaluation or training run. The corrected implementation and the review runbook take precedence over stale historical claims.

1. **Patch base versus frozen working tree.** The review document describes applying to `f29df1a8776d63ad5aa21d40f4088869753f5978`; this checkout already contains that patch and the follow-up archived-test commit. The frozen corrected implementation is the current commit recorded in `environment/environment.json` (commits `5ded6ac` and `61cd93a` are its immediate review history). Upstream `361d3f2337ed55576c241866353e4a9b72a9e9b6` is retained only as the original-harness B0 reference; it is not merged into corrected B1/R1.

2. **Historical split counts versus corrected split.** The original execution brief discusses 108 training questions and reserving approximately 12 for development. The corrected protocol explicitly preserves the recorded manifest: 96 train, 12 dev, and 42 eval question IDs. No new split is generated and no evaluation question is used in prompts, reward weights, learning-rate selection, or checkpoint selection.

3. **Document overlap.** The brief/runbook requires disclosure rather than pretending train/dev are document-disjoint. The frozen manifest has 8 train/dev overlapping documents, with zero train/eval and zero dev/eval document overlap; this is a limitation and not corrected by resampling.

4. **Evaluation split status.** The 42-question evaluation split was previously inspected during diagnostics. It is therefore labeled a previously inspected evaluation split, not an untouched test set. The historical `b1_base_eval42` artifact was produced under the predecessor scorer and is not used for current checkpoint or LR selection.

5. **SDK gate versus offline tests.** The runbook says skipped SDK contract tests are a failed runtime gate. The root capability suite is the authoritative count; its 54 tests repeat the root suite and are not added to the independent test total. The legacy 28-test suite is retained diagnostically and its one skip is not treated as an SDK pass.

6. **Pilot/final nominal batches versus actual updates.** `STEPS`/36 batches are nominal scheduling caps. Constant/unresolved groups may skip optimizer calls. We will report `optimizer_audit.jsonl` actual completed and skipped updates and will not describe a nominal cap as an achieved update count.

7. **Judge availability.** The frozen policy is DeepInfra `deepseek-ai/DeepSeek-V4.1-Flash` for residual semantic cases only. Provider failures and unresolved semantic cases remain unresolved; with no usable judge credential, non-exact text groups are excluded from training and retained in the evaluation denominator. Credentials are inherited ephemerally and never printed or persisted.

8. **B0 comparability.** B0 is an original-harness diagnostic unless rerun with a deterministic adapter and sufficient trace information. Missing historical receipts cannot be fabricated; B0 grounding is marked unmeasurable when necessary and excluded from strict paired B1/R1 tables.

9. **Target review gate.** `targets.draft.json` is not valid for experiments. Before preflight, targets must have source support, typed fields, exact one-based page spans, and explicit reviewer/method/date metadata. Any automated source audit is labeled automated and is not described as human review.
