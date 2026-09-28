# Paper-writing handoff: measured evidence to the first validity gate

## Research question and protocol

The corrected FinanceBench/Nemotron 3.5 Lightning protocol asks whether GRPO changes correct-and-finished grounded success after repairing terminal acceptance, typed numeric scoring, citation receipts, pagination, epoch handling, group filtering and matched dev evaluation. The frozen protocol is `protocol.md`; conflicts with the older execution brief are in `protocol_conflicts.md`. The 96/12/42 question IDs are fixed, and the 42-question evaluation split is previously inspected rather than untouched.

## Measured development result

B1 base on the 12-question dev split completed all 12 scheduled one-rollout evaluations. It accepted 2/12 terminal finishes (16.67%), produced 0/12 correct-and-finished answers and 0/12 grounded successes, with mean A 0.0000 and mean G 0.0833. There was 1 unresolved episode, 4 length exits and 6 turn-limit exits; no infrastructure failures occurred. Ten episodes had zero citations, and 75% read an annotated gold page. The denominator remains 12. These are development measurements, not final eval results.

## Training-signal gate

The pinned SDK completed a one-nominal-batch provider smoke at LR 1e-5. It sampled 8 groups of 8 trajectories (64 trajectories), all groups were all-zero/all-equal, 4 groups were unresolved, none were retained, and the trainer made 0 actual optimizer updates. It saved a final sampler checkpoint and performed the required matched 12-question dev diagnostic, but that checkpoint is explicitly labeled zero-update and is not R1. The gate blocks the paid LR pilots and final GRPO.

## Scorer evidence

`table3_scorer.csv` contains 15 numeric regression/adversarial cases and passes all expected decisions, including sign, percent-point, accounting-negative, scale, currency, zero, rounding and number-dump attacks. The provenance and citation protections are covered by the 54-test authoritative suite; the suite is not counted a second time in test totals.

## Effects and uncertainty

No B1−B0 or R1−B1 paired effect is reported. B0 lacks a settings-matched corrected-rubric rerun with common receipts, and R1 has no actual optimizer update. Consequently no paired bootstrap interval, learning curve or final 42-question corrected evaluation is scientifically estimable from this run.

## Compute and data budget

The provider smoke used model `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, renderer `nemotron3_ultra`, training temperature 1.0, evaluation temperature 0.2, 8 turns, 1,024 generated tokens, batch 8, group 8, LoRA rank 32, seed 0 and LR 1e-5. Tinker and cookbook versions, hashes and commands are in `environment/` and `run_manifest.json`. Monetary cost is unmeasured.

## Limitations

The source target audit is automated and exact against indexed pages after normalized alignment; no human semantic review is claimed. Train/dev share 8 documents, while train/eval and dev/eval have zero document overlap. The evaluation split was previously inspected. The sample is small, there is one training seed, provider generation is not treated as bitwise seeded, the one residual semantic judge call failed with a RuntimeError and remained unresolved, public-data pretraining contamination is unknown, and no generalization claim is justified. A zero-signal gate may reflect model capability and rubric strictness; it is not evidence that GRPO cannot learn.

## Four-page outline

1. Problem and failure mechanisms: show how terminal echoes, numeric shortcuts, missing provenance and dropped failure denominators invalidate naive FinanceBench RL measurements.
2. Corrected harness and reward: define receipts, typed targets, reviewed derivations, explicit F/A/G/Ret, pagination, epoch-aware scheduling and unresolved-group handling.
3. Controlled results: report the complete B1 dev smoke and zero-update training gate, then state that B0/R1/final eval42 are blocked rather than filling expected cells.
4. Error analysis and limitations: analyze turn-limit/length failures, zero-signal groups, target-audit limits, document overlap, inspected evaluation split and the need for a successful capability/update gate before causal claims.

## Artifact map

- `protocol.md`, `protocol_conflicts.md`, `run_manifest.json`: frozen design and conflicts.
- `B1-dev-smoke/`, `B1-dev-report/`: raw dev predictions, rollout traces, tables and failure figure.
- `provider_smoke_lr1e5_v3/`: raw training traces, group/optimizer audits, checkpoint metadata and matched zero-update dev.
- `table1_main.*`, `table2_failures.*`, `table3_scorer.*`, `table4_paired.*`: measured and explicitly blocked tables.
- `completion_report.md`, `measured_summary.json`, `case_studies.md`: status and interpretation.
