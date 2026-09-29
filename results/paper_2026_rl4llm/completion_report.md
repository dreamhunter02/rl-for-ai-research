# Workshop evidence completion report

## Complete

- Corrected implementation frozen at commit `61cd93a8048cb8d944d1da0ac1d27aaa592547fc` with review history `5ded6ac` and `61cd93a`.
- Protocol conflicts are documented in `protocol_conflicts.md`. The frozen 96/12/42 question IDs remain unchanged in `split_frozen.json`; train/dev document overlap is disclosed in `protocol.md`.
- The original automated audit found all 189 indexed support spans for 150 targets. This establishes source-span presence only; no human semantic review is claimed.
- Rubric-v3 corpus/target preflight passed for all 150 questions and 12,013 indexed pages. The provider-capable suite now passes 67 tests with zero skips; its repeated root tests are not counted as an independent suite.
- The B1 development smoke completed all 12 questions. It produced 2/12 accepted finishes, 0/12 correct-and-finished answers, 0/12 grounded successes, mean answer correctness `0.0000`, and mean grounding `0.0833`.
- Offline diagnosis of the initial 64 saved training trajectories is preserved in `provider_smoke_lr1e5_v3/offline_diagnosis.md`: 35 hit the turn limit, 15 hit the token limit, 13 attempted `finish`, and the batch retained no groups or updates.
- Focused sampling probes then established that Nemotron 3.5 Lightning can produce mixed, correct, source-grounded outcomes on the source-verified direct-numeric training question `q04209`.
- The bounded group-size-8 capability smoke in `provider_smoke_costco_update_v2_g8/` passed the optimizer signal gate. Its rewards were `[1, 0, 1, 1, 0, 1, 0, 0]`, one group was retained, and exactly one real optimizer update completed with finite metrics. The final sampler checkpoint is `tinker://fc94753e-3098-5d77-9238-c9ed88c46994:train:0/sampler_weights/final`.
- A fresh 12-question dev run from that checkpoint completed and produced 1/12 verified-correct answers and 0/12 grounded successes. One update is a capability result, not evidence of improvement.
- The direct and derived numeric audits corrected eight precision metadata errors. All 36 derived expressions recompute successfully against their source-backed operands.
- All 117 text support spans now fit a single read receipt (maximum 2,387 characters), and multi-span answers use distinct required claim IDs. This repairs evidence visibility but is not human semantic review.
- The two source/answer conflicts, `q04672` and `q00283`, are explicitly marked unresolved. Training removes their groups; evaluation keeps them in the frozen denominator as unresolved.
- Offline rubric-v3 replay of the original 64 trajectories still retains 0/8 groups. This confirms that the old batch cannot be rescued by label repair alone.
- A deterministic 50-trajectory audit found 41 episodes without an accepted finish, 4 incorrect/incomplete finishes, and 5 semantically correct finishes conservatively left unresolved. It also identified and repaired one overly restrictive Boeing fact and two valid alternate evidence spans.
- SparkyOne now supports a permission-checked headless judge credential file. A 0600 placeholder exists at `~/.config/financebench/deepinfra.key`; the loader was verified without retaining the diagnostic secret.

## Validity gates still open

- The DeepInfra semantic judge is unavailable because no `DEEPINFRA_API_KEY` credential can be loaded on SparkyOne. `judge_credential_test.json` records that the test failed before making a network request. Text-answer residuals must remain unresolved until this is fixed.
- The 98 text `required_facts` were produced by deterministic answer segmentation. They still need semantic adjudication or a working frozen judge; no human review is claimed.
- The 50 saved-trajectory audit is complete, but fresh text decisions cannot be calibrated until the semantic judge works.

## Work deliberately withheld

The optimizer capability gate is no longer the blocker. The 10-batch learning-rate pilots, final R1 training, checkpoint selection, corrected eval42, paired bootstrap effects, learning curve, and final B0/B1/R1 tables remain paused until the judge credential works and the text-rubric/trajectory audit is complete. No positive GRPO-learning claim is made from the one-update smoke.
