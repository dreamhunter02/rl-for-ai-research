# Question-led reward correction

Authorized: repair rubric generation and rescore the same 24 saved dev traces.
No policy inference, training, teacher remigration, remote production deployment,
or modification to original traces/results is included.

- [x] Add regressions demonstrating reference-derived requirement expansion.
- [x] Implement two-stage question-only requirements then reference interpretation;
  enforce identical applicability and claim IDs. Preserve the approved formula.
- [x] Version scorer/prompt cache identity; retain candidate-claim grounding and
  check volunteered numbers even for semantic questions.
- [x] Run full configured suite and live classification/numeric controls: 149
  passed, 10 opt-in skips; all 10 live controls passed separately.
- [x] Rescore 24 originals, inspect 00499 and numeric cases, publish new HTML:
  `results/trace_comparisons/reward_v5_question_led/comparison.html`.

Versions: workshop-rubric-v5-question-led / component-judge-v4-question-led.
The v4 teacher export retains historical v4 rewards; it is not silently promoted
to the new scoring protocol. All new results must be labeled post-hoc rescoring.

Final 24 scores use Inference Hub openai/openai/gpt-5.6-terra, with zero
unresolved. Original hashes and frozen dev IDs verified. Full live controls caught
and drove repair of optional arithmetic incorrectly contaminating semantic credit,
then of a partly false bundled claim being marked grounded. Grounding remains
LLM-assessed, not independently calibrated by these controls. Previous draft runs
and all original v4 evidence remain preserved. No commit or deployment performed.
