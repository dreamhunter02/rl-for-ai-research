# Direct numeric target audit

Status: agent-assisted source audit; **not human review**. The frozen 96/12/42 split is unchanged.

The corrected target artifact is versioned as `workshop-rubric-v3-source-rounding`; prior v2 runs remain identified by their recorded target hash.

## Result

The 16 non-derived numeric targets divide into three classes:

- 8 source-exact or conditionally exact targets.
- 6 valid rounded benchmark answers whose stored `precision` is too strict.
- 2 source/answer conflicts that must remain unresolved pending adjudication.

The precision issue is scientifically material. The scorer currently derives precision from display formatting such as `382.00`, then requires equality at two decimals. A filing-grounded answer such as `381.603` therefore fails against the intended whole-million benchmark answer `382.00`.

## Review table

| ID | Split | Source value in requested units | Frozen target | Finding | Proposed disposition |
|---|---|---:|---:|---|---|
| `04672` | train | 8.738 USD billion | 8.70, precision 2 | Conflict: the 3M balance sheet reports 8,738 USD million; 8.70 is not a standard rounding of that value. | Keep unresolved until adjudicated; do not train on it. |
| `01319` | train | 0 | 0, precision 0 | Valid conditional zero because restructuring cost is not explicitly listed. | Keep. |
| `05718` | train | 0.389 USD billion | 0.40, precision 2 | Benchmark rounds to one decimal, but the rubric requires two. | Set value 0.4 and precision 1. |
| `07661` | train | 381.603 USD million | 382.00, precision 2 | Benchmark rounds to a whole million, but the rubric requires two decimals. | Set value 382 and precision 0. |
| `04209` | train | 59,268 USD million | 59,268.00, precision 2 | Source-exact. | Keep. |
| `04700` | train | 32,780 USD million | 32,780.00, precision 2 | Source-exact. | Keep. |
| `03029` | dev | 1,577 USD million expenditure | 1,577.00, precision 2 | Source-exact magnitude; parentheses denote cash outflow. | Keep. |
| `03882` | dev | 1,615.9 USD million | 1,616.00, precision 2 | Benchmark rounds to a whole million, but the rubric requires two decimals. | Set value 1616 and precision 0. |
| `08286` | eval | 11,588 USD million | 11,588.00, precision 2 | Source-exact. | Keep. |
| `04417` | eval | 5,409 USD million | 5,409.00, precision 2 | Source-exact. | Keep. |
| `10285` | eval | 12,645 USD million | 12,645.00, precision 2 | Source-exact. | Keep. |
| `04171` | eval | 302.578 USD million | 303.00, precision 2 | Benchmark rounds to a whole million, but the rubric requires two decimals. | Set value 303 and precision 0. |
| `03282` | eval | 5,466.312 USD million | 5,466.00, precision 2 | Benchmark rounds to a whole million, but the rubric requires two decimals. | Set value 5466 and precision 0. |
| `03531` | eval | 16,525 USD million | 16,525.00, precision 2 | Source-exact. | Keep. |
| `04980` | eval | 4.625 USD billion | 4.60, precision 2 | Benchmark expresses a one-decimal rounded answer with a trailing zero. | Set value 4.6 and precision 1. |
| `00283` | eval | Approximately 70 USD million remaining (10% of approximately 700) | 77.78, unit `number` | Conflict: the cited text says 700 million total and 90% already incurred. The frozen value and unit are not supported by that statement. | Keep unresolved until adjudicated; do not score as negative. |

## Required before LR pilots

1. Apply the six precision-only corrections and add regression cases proving that both the rounded benchmark answer and the exact filing value receive credit.
2. Mark `04672` and `00283` unresolved without removing them from the frozen split.
3. Complete the same semantic check for the 36 derived numeric targets and the text-target fact rubrics.
4. Restore and test the DeepInfra semantic judge before scoring non-exact text answers.
