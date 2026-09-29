# Derived numeric target audit

Status: agent-assisted source and formula audit; **not human review**. The frozen 96/12/42 split is unchanged.

## Result

All 36 derived numeric targets were checked for:

- correspondence between the question's stated metric and the encoded expression;
- operand period, metric, sign, unit, and scale;
- presence of every operand in an exact cited source span;
- recomputed result against the frozen target at the requested rounding precision.

The expressions and operands are semantically consistent with their questions, and all recomputed results match. Two precision metadata errors were corrected in rubric v3:

| ID | Question requirement | Recomputed value | Prior rubric | Rubric v3 |
|---|---|---:|---:|---:|
| `06272` | Dividend payout ratio, two decimals | 0.798156… | 0.8 at precision 1 | 0.80 at precision 2 |
| `04103` | Cash conversion cycle, two decimals | -3.70061… | -3.7 at precision 1 | -3.70 at precision 2 |

These are metadata corrections, not changes to the formulas, operands, split membership, or cited evidence. Regression tests verify that the full-precision calculated values match the two-decimal targets.

## Remaining target work

The numeric rubric now has complete agent-assisted review, except for the two unresolved source/answer conflicts documented in `direct_numeric_target_audit.md` (`04672` and `00283`). The 98 text targets still require semantic review of their `required_facts`; DeepInfra judge availability remains necessary for non-exact paraphrases.

