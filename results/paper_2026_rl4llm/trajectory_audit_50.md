# Rubric-v3 training trajectory audit

Status: agent-assisted review of a deterministic 50-row prefix of `replay_v3/trajectory_replay.jsonl`; **not human review** and no new inference compute.

## Sample

- Trajectories reviewed: 50
- Accepted finishes: 9
- Turn-limit exits: 25
- Token-length exits: 15
- Rejected finish: 1
- No accepted finish: 41

The sample includes all accepted finishes for `00216`, `01290`, and `00711`, plus one of the two accepted finishes for `00585`. It also covers derived-numeric attempts for `10420` and `10499` and failed attempts for `00799`.

## Agreement with source evidence

| Outcome | Count | Audit decision |
|---|---:|---|
| No accepted finish | 41 | Zero answer reward is correct; there is no scoreable terminal answer. |
| Accepted but incorrect or materially incomplete | 4 | Withholding answer credit is correct. This includes the wrong FY2021 tax-rate sign, an unsupported/wrong J&J turnover answer, one mixed Verizon conclusion, and one incomplete Boeing customer answer. |
| Accepted and semantically correct for the question | 5 | Conservatively unresolved because the semantic judge was unavailable; these were not converted into negative training labels. |

Two correct Verizon answers state the source-backed quick ratio of approximately 0.54 and conclude that liquidity is not healthy under the conventional threshold. Three Boeing answers correctly identify commercial airlines and the U.S. government as the primary customer groups.

## Rubric findings

1. Exact-alias grading alone misses valid explanatory answers; a frozen semantic judge remains necessary.
2. Boeing's recorded 40% U.S.-government revenue share is supporting context, not required to answer *who* the primary customers are. Rubric v3 now treats only the two customer groups as required.
3. Valid Boeing evidence appears on alternate pages 3 and 40 in the saved trajectories. Those exact spans are now encoded as alternatives to pages 8 and 10.
4. Missing `finish` calls and calculation-input failures—not answer-label disagreement—dominate the batch.
5. The audit supports conservative masking of unresolved text groups; it does not justify LR pilots until the judge credential works.

