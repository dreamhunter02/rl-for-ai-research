# Additive reward report

User approved scoring the saved 24 traces with
R=.15E+F(.20G+.45A+.15B+.05), and an HTML showing latest scores only.
This is an offline candidate-protocol evaluation, not deployment to training.

- [x] Add and run failing regressions, implement strict E/G/A/B/F aggregation.
- [x] Build a latest-only HTML renderer; preserve all previous artifacts.
- [x] Judge all 24 frozen dev records via Inference Hub; verify provenance/counts.
  22 scored; both 00684 records unresolved because gold says .8% while cited
  19.4% and 18.5% imply .9 percentage points. No gold correction was imposed.
- [x] Full suite: 162 passed, 14 live skips, 43 subtests. Four current additive
  live controls passed separately. Inspect 00499 and 00807; preserve judge limits.
- [x] Publish latest-only HTML at results/trace_comparisons/reward_v6_additive_latest/comparison.html.

E uses source receipt sufficiency: none=0, partial=.5, full=1. It is independent
of final citation choice and correctness. G is the fraction of material factual
claims supported by valid final citations; pure subjective conclusions belong in A.
A retains question-led numeric/semantic requirements. B is binary coverage of all
material reference claims, requires A=1 and no material contradictions. F preserves
the historical accepted-finish status. Provider failures remain unresolved/null.

The live v5 training scorer and v4 teacher exports are not changed by this task.
