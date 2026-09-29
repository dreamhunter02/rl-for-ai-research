# Offline diagnosis of the zero-update provider smoke

This report analyzes the committed `provider_smoke_lr1e5_v3` evidence at
commit `36492f7`. It does not rescore trajectories, change targets, or claim a
model improvement.

## Measured outcome

- 64 trajectories were sampled: eight FinanceBench questions with group size
  eight.
- Rewards were zero before constant-group removal. All eight groups were
  all-zero/all-equal, four groups contained an unresolved trajectory, no group
  was retained, and the optimizer correctly made zero updates.
- 35 trajectories stopped at the turn limit, 15 at the token limit, 10 at an
  accepted `finish`, and four completed without an accepted `finish`.
- The model attempted `finish` 13 times. Ten text submissions were accepted;
  two numeric submissions were rejected because `value` was a JSON number
  instead of the required decimal string; one text attempt first failed
  citation validation.
- Every accepted text submission missed the sole exact alias and became
  unresolved. The DeepInfra fallback failed with `RuntimeError` on all ten.
  Four of these submissions had `G=1` and `Ret=1`, demonstrating that valid
  retrieval/grounding occurred but could not produce reward without a resolved
  correct answer.
- Tool use was substantial: 151 BM25 searches, 165 document greps, 49 reads,
  50 calculation attempts, 13 table searches, and 13 finish attempts.

## Per-question failure table

| Question | Accepted / 8 | Finish attempts | Stop reasons | Grounded | Principal observed failure |
|---|---:|---:|---|---:|---|
| `financebench_id_00216` | 3 | 3 | 3 finish, 2 turn, 3 token | 3 | All accepted text answers unresolved; judge failed. Calculation provenance also produced 16 validation errors. |
| `financebench_id_00585` | 2 | 2 | 2 finish, 3 turn, 3 completed without finish | 0 | Exact-alias text grading plus judge failure; seven calculation-provenance errors. |
| `financebench_id_00603` | 0 | 0 | 7 turn, 1 completed without finish | 0 | No finish attempt despite a qualitative question. |
| `financebench_id_00711` | 1 | 2 | 1 finish, 7 turn | 0 | A derived numeric question is typed as text; one citation validation error and one unresolved accepted text answer. |
| `financebench_id_00799` | 0 | 0 | 4 turn, 4 token | 0 | No finish; nine calculation/provenance errors while answering a derived comparison typed as text. |
| `financebench_id_01290` | 4 | 4 | 4 finish, 3 turn, 1 token | 1 | Four plausible text answers all required judge fallback; judge failed. Only one cited all reviewed claims. |
| `financebench_id_10420` | 0 | 2 | 3 turn, 5 token | 0 | Both finishes rejected because numeric `value` was a float; 12 calculation/provenance errors. Target is marked non-derived. |
| `financebench_id_10499` | 0 | 0 | 6 turn, 2 token | 0 | No finish attempt. Target is a calculated inventory-turnover value but is marked non-derived. |

## Root-cause chain

1. The rollout budget and search behavior yielded no finish attempt in 51 of 64
   trajectories (including four nominally completed trajectories).
2. The structured calculation contract was difficult for the model to satisfy:
   48 recorded errors involved operand structure, source quotes, metric/period
   matching, expression variables, or malformed nested arguments.
3. The smoke targets are not semantically ready for this contract. All eight
   are marked `derived: false`; at least the inventory-turnover and quick-ratio
   questions require calculations. Six are text targets with one exact alias
   and empty `required_facts`.
4. Consequently, the ten accepted text finishes depended on the semantic
   judge. Its runtime failure left all ten unresolved, and unresolved groups
   are intentionally excluded from GRPO.
5. With retrieval shaping disabled, correct retrieval alone earns no reward.
   Because no accepted submission received `A=1`, every group was constant
   zero and the optimizer had no learning signal.

## Minimal hypothesis to test next

The first paid rerun should not change the model, reward formula, split, LR, or
group filtering. First repair and independently verify the same eight targets,
verify the judge credential with one cached/non-cached residual, and replay the
saved accepted submissions through the corrected targets. A sampling-only rerun
is justified only if this offline replay produces both resolved labels and
nonconstant rewards.
