# Workshop evidence completion report

## Complete

- Corrected implementation frozen at commit `61cd93a8048cb8d944d1da0ac1d27aaa592547fc` with the review history `5ded6ac` and `61cd93a`.
- Protocol conflicts documented before new experiments in `protocol_conflicts.md`.
- Frozen 96/12/42 question IDs prepared in `artifacts/workshop/split.json`; train/dev overlap is disclosed in `protocol.md`.
- Automated exact indexed-page source audit verified 189/189 support spans for 150 targets; no human semantic review is claimed.
- Corpus/target preflight passed; scorer regression table passed.
- Authoritative capability suite passed 54 tests with zero skips; the legacy 28-test suite passed with its one diagnostic skip. The capability suite repeats the root suite and is not counted as an additional independent test set.
- Matched B1 development smoke completed on all 12 dev questions and is preserved under `B1-dev-smoke/`, with report tables and failure figure.
- One Tinker provider training smoke completed and preserved under `provider_smoke_lr1e5_v3/`, including raw rollouts, group/optimizer audits, checkpoints and matched zero-update dev evaluation.

## Scientific gate and blocked work

The actual-optimizer-update smoke gate failed: one nominal train batch sampled 8 groups × 8 trajectories, all groups were zero-reward/all-equal, four were unresolved, zero groups were retained, and actual optimizer updates were `0`. This is a valid measured null/blocked condition, not a reason to relax filtering or train on constant reward. Therefore the 10-batch LR pilots, final R1 training, checkpoint selection, 42-question corrected evaluation, and strict B0/R1 paired effects were not launched.

The B1 dev smoke had 2/12 accepted finishes, 0/12 correct-and-finished answers, 0/12 grounded successes, mean A `0.0000`, mean G `0.0833`, 1 unresolved episode, 1 judge error retained as unresolved, 4 length exits and 6 turn-limit exits. These figures are development evidence only.

## Remaining measurements

B0 matched rerun, successful R1 with at least one actual update, LR selection, final corrected eval42, paired bootstrap effects, learning curve and final paper tables across B0/B1/R1 remain blocked. No positive GRPO claim is made.
