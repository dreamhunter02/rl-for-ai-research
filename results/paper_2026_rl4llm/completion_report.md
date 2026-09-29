# Workshop evidence completion report

## Complete

- Corrected implementation frozen at commit `61cd93a8048cb8d944d1da0ac1d27aaa592547fc` with review history `5ded6ac` and `61cd93a`.
- Protocol conflicts are documented in `protocol_conflicts.md`. The frozen 96/12/42 question IDs remain unchanged in `artifacts/workshop/split.json`; train/dev document overlap is disclosed in `protocol.md`.
- The original automated audit found all 189 indexed support spans for 150 targets. This establishes source-span presence only; no human semantic review is claimed.
- Corpus/target preflight and scorer regressions passed. The authoritative suite passed 54 tests with zero skips; its repeated root tests are not counted as an independent suite.
- The B1 development smoke completed all 12 questions. It produced 2/12 accepted finishes, 0/12 correct-and-finished answers, 0/12 grounded successes, mean answer correctness `0.0000`, and mean grounding `0.0833`.
- Offline diagnosis of the initial 64 saved training trajectories is preserved in `provider_smoke_lr1e5_v3/offline_diagnosis.md`: 35 hit the turn limit, 15 hit the token limit, 13 attempted `finish`, and the batch retained no groups or updates.
- Focused sampling probes then established that Nemotron 3.5 Lightning can produce mixed, correct, source-grounded outcomes on the source-verified direct-numeric training question `q04209`.
- The bounded group-size-8 capability smoke in `provider_smoke_costco_update_v2_g8/` passed the optimizer signal gate. Its rewards were `[1, 0, 1, 1, 0, 1, 0, 0]`, one group was retained, and exactly one real optimizer update completed with finite metrics. The final sampler checkpoint is `tinker://fc94753e-3098-5d77-9238-c9ed88c46994:train:0/sampler_weights/final`.
- A fresh 12-question dev run from that checkpoint completed and produced 1/12 verified-correct answers and 0/12 grounded successes. One update is a capability result, not evidence of improvement.

## Validity gates still open

- The DeepInfra semantic judge is unavailable because no `DEEPINFRA_API_KEY` credential can be loaded on SparkyOne. `judge_credential_test.json` records that the test failed before making a network request. Text-answer residuals must remain unresolved until this is fixed.
- Target-v2 generation and replay improve typing, argument coercion, numeric parsing, and calculation handling, but the resulting target audit is automated. It must not be described as human or independent semantic review.
- At least one official target conflict was found: `q04672` cites 3M PP&E of 8,738 million while the stored target is 8.70 billion. The target/rubric set therefore needs a question-by-question semantic review before paper-scale training or evaluation.

## Work deliberately withheld

The optimizer capability gate is no longer the blocker. The 10-batch learning-rate pilots, final R1 training, checkpoint selection, corrected eval42, paired bootstrap effects, learning curve, and final B0/B1/R1 tables remain paused until the judge credential works and the target/rubric review is complete. No positive GRPO-learning claim is made from the one-update smoke.
