# Apply the September 28 FinanceBench review fixes

This patch targets upstream **f29df1a8776d63ad5aa21d40f4088869753f5978**. It supersedes the earlier `financebench_workshop.patch`; apply only this patch to that upstream revision. No training or provider evaluation was run while producing the patch.

```bash
git switch -c fixes/financebench-review f29df1a8776d63ad5aa21d40f4088869753f5978
git apply --check /path/to/financebench_review_fixes.patch
git apply /path/to/financebench_review_fixes.patch
python -m pip install -r requirements-workshop.txt
PYTHONPATH=harness python -m unittest discover -s tests -v
PYTHONPATH=harness python -m unittest discover -s harness -p 'test_*.py'
python harness/capability_tests.py
```

The capability command deliberately fails when SDK tests are skipped. Do not claim E1 passed from offline tests alone. One legacy test demanding high lexical scores for 46 historical teacher answers is explicitly retired: it contradicts the new reviewed-target rubric. Teacher traces are not used for training by this patch.

## What changed

| Review finding | Implemented correction |
|---|---|
| Percent/decimal/sign/scale errors and number dumps receive credit | Decimal scalar grading, explicit unit/scale and reviewed rounding precision; prose cannot override numeric fields |
| `finish` echoes validate fabricated citations | Per-episode receipts created only from visible read/grep/table observations, checked before terminal submission |
| Live grounding is always zero | Fix empty-call-ID tool-name overwrite; authoritative grading uses receipts directly |
| Typed fields/calc ID ignored | Validated accepted submission; derived answers require actual calculation, source operands, reviewed expression and matching result |
| Attempted/multiple finish counted | Invalid calls receive recoverable validation errors; accepted finish stops; multiple submissions invalidate acceptance |
| Three epochs were only one pass | Epoch-aware dataset, reshuffle and remainder batches; actual updates and zero-signal skips logged |
| Evaluation diverged from training | `eval_agent.py` delegates to the same environment/renderer/scorer used by `workshop_eval.py` |
| Test split used for checkpoint choice | Training rejects checkpoint evaluation on any split other than dev |
| Final optimizer step was unevaluated | Final saved sampler checkpoint always receives a separate matched dev evaluation |
| Pagination stalled and oversized JSON hid evidence | Moving default windows, complete structured JSON, accurate delivered offsets, strict PDF page cache |
| Ambiguous/provider-failed judging became labels | Unresolved status retained; entire training group excluded; evaluation keeps denominator and bounds |
| Finish-rate denominator dropped failed episodes | Terminal metrics explicitly include zero on failed exits; paper exporter rejects missing/duplicate question rows |
| `American Express` filter missed `AMERICANEXPRESS` | Normalize spacing/punctuation in company filters |
| Malformed tool arguments or citation IDs crash training | Delegate nonobject arguments to SDK validation; reject nonstring identifiers before scoring |
| Equivalent scales lose calculation credit | Interpret reviewed calculation results in target units/scale before comparing submissions |
| Initial context overflow disappears from dev averages | Add terminal zero metrics to the initial-overflow path as well as step exits |

## Prepare without changing the recorded splits

```bash
python harness/workshop_prepare.py prepare --split split.json \
  --split-manifest results/paper_2026_rl4llm_agents/split_manifest.json \
  --out artifacts/workshop-reviewed
export FINANCEBENCH_ROOT="$PWD"
export FINANCEBENCH_SPLIT="$PWD/artifacts/workshop-reviewed/split.json"
export FINANCEBENCH_TARGETS="$PWD/artifacts/workshop-reviewed/targets.reviewed.json"
export FINANCEBENCH_PDF_PAGES=1
```

Create `targets.reviewed.json` from the draft only after checking the requested value, units supplied by the question, rounding, decision/text facts, and exact one-based PDF support spans. A draft is not a valid training target. Record reviewer/method/date. Derived targets also require expression and source-backed operand specifications. See `docs/WORKSHOP_AGENT_RUNBOOK.md` for a schema example. Alternate correct evidence must be added to the rubric before freezing it; strict literal support is deliberately conservative and needs a source audit.

```bash
python harness/preflight_repaired.py
python harness/workshop_eval.py --condition B1 --run-id B1-dev --split dev \
  --out results/paper_2026_rl4llm/B1-dev
python harness/run_workshop.py --config configs/nemotron35_lightning_grpo_repaired_pilot.json \
  --log-path results/tinker_runs/reviewed-pilot-lr1e5 --dry-run
```

After SDK/corpus/label checks and a one-update provider smoke, remove `--dry-run` for the authorized pilot. Repeat with `--lr 5e-5` into a different directory. The config loader applies declared settings instead of assuming JSON files are automatically consumed. Final config requires explicit `--lr` chosen on dev and one `--seed` per run. Existing nonempty run directories are rejected. Environment credentials are inherited, never printed by `--dry-run`.

The nominal pilot cap is 10 batches; the final cap is 36 batches for the preserved 96 questions, batch 8, three epochs. All-constant and unresolved groups can reduce actual optimizer updates. Do not claim 36 actual updates from `STEPS=36`. Archive `optimizer_audit.jsonl`, `group_audit.jsonl`, checkpoints and the automatic `final_dev/` predictions. Use a measured update-zero B1 and matched dev checkpoints for LR/weight selection; run the 42-question evaluation only after selection.

## Results status and paper handoff

Historical files in `results/paper_2026_rl4llm_agents/` remain diagnostics from the old scorer. The old E0/E1 passing claims and 5/42 grounded-accuracy classification are superseded by the reproduced defects. The old update-1 dev finish rate is **4/12 = 33.33%**, not 4/11. All three recorded pilot steps had zero extracted strong/weak evidence; that run does not test the intended grounding reward. Do not infer GRPO ineffectiveness from it.

Generate new B1/R1 predictions with `workshop_eval.py` and the tables/paired confidence intervals with `workshop_report.py`; the runbook gives exact commands. Finish the source-label/human audit, B0 compatibility work and measured provider costs before marking the paper package complete. No script invents these measurements. A corrected patch is not evidence of improved benchmark performance.

## Verification boundary

Offline tests cover scalar regressions, fake citations including terminal echoes, accepted-submission gating, empty IDs, pagination, source calculations, 36-batch epochs, uncertainty handling, fixed denominators and report generation. The pinned SDK could not be installed in the editing workspace, so real SDK contracts, renderer round trips, provider optimizer/checkpoint behavior and end-to-end corpus runs remain required on the training machine. `requirements-workshop.txt` pins cookbook; archive a resolved `pip freeze` after installation.

Validation on this patch: root suite **54 tests, 46 passed, 8 SDK skips**; legacy suite **28 tests, 19 passed, 9 skips** (the same 8 SDK checks plus the retired lexical assertion). Compilation and whitespace checks passed. Preparation against the actual upstream split and recorded manifest preserved **96 train / 12 dev / 42 eval**. These are implementation checks, not benchmark results.
