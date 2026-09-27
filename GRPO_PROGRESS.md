# FinanceBench GRPO Progress

Date: 2026-09-26

## Current status

The project has a functioning FinanceBench retrieval harness, a validated Unsloth Qwen3.5-4B GRPO smoke test, and a 30-episode multi-turn FinanceBench rollout pilot. We do not yet have a trained FinanceBench GRPO checkpoint: the current Unsloth run is a toy verifier experiment, while the FinanceBench run exercised the harness through a Qwen3.5-4B SGLang endpoint.

## Harness completed

The existing FinanceBench harness provides:

- BM25 prose and table search.
- Document-scoped grep.
- Page-aware reads and table reads.
- Safe arithmetic calculation.
- Evidence provenance recording.
- Typed `finish(answer, evidence_document, evidence_page)` termination.
- Full trajectory, tool-call, reward, timing, and termination logging.
- Frozen split of 108 training questions and 42 held-out evaluation questions.

The active answer scorer extracts `finish.answer` first, handles `Answer:` responses, supports numeric/currency/scale/percentage/unit/rounding/sign/period checks, rejects contradictions, and keeps finish/format/tool-count/latency as separate diagnostics. The known 91-trace audit has mean revised reward `0.9841`, but that audit is selected-trace calibration and does not validate the scorer on exploratory rollouts.

## Unsloth setup on Spark Two

Host: `spark-a16b`.

The isolated environment is inside container `qwen-train` at `/workspace/unsloth-qwen4b-venv` and contains:

- Unsloth `2026.9.11`.
- Transformers `5.5.0`.
- TRL `0.24.0`.
- PyTorch `2.12.1+cu130`.
- bf16 support on NVIDIA GB10.

The Qwen3.5 recipe uses `FastLanguageModel`, bf16/16-bit LoRA, `fast_inference=False`, rank 32, LoRA alpha 32, and target modules:

`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

`fast_inference=False` is intentional because the current Unsloth Qwen3.5 guidance uses the non-vLLM path for this model family.

The smoke script is maintained at:

`harness/unsloth_qwen4b_grpo_smoke.py`

A copy used on Spark Two is at:

`/home/dreamhunter/unsloth_qwen4b_grpo_smoke.py`

## Toy GRPO experiment

The first rank-32/16k smoke test used two arithmetic prompts and two GRPO steps. It completed successfully and saved an adapter, proving that model loading, LoRA injection, GRPO generation, backpropagation, and checkpoint saving work.

The first toy verifier was binary exact match, which produced zero within-group reward variance. It was replaced with a mixed verifier:

- Exact answer: approximately `1.0`.
- Formatted but wrong numeric answer: approximately `0.25`.
- Numeric output without the requested answer marker: approximately `0.10`.
- Empty/non-numeric output: `0.0`.
- Tiny bounded length tie-breaker for otherwise different outputs.

The mixed experiment used 15 prompts, 2 generations per prompt, and 15 steps: 30 total rollouts. It completed in `73.13` seconds with rank 32 and a 16k context cap. It produced nonzero reward variance in multiple groups, reward standard deviations up to approximately `0.6293`, nonzero gradient norms, low KL drift around `0.0007`, and saved:

`/home/dreamhunter/unsloth_grpo_smoke_qwen35_4b_r32_16k_mixed30/checkpoint-15`

This passes the Unsloth/TRL stack smoke gate only; it is not evidence of FinanceBench improvement.

## FinanceBench rollout pilot

The existing multi-turn harness was run on 30 unique FinanceBench training questions using Qwen3.5-4B through the local SGLang endpoint:

- Maximum turns: 6.
- Maximum output tokens per turn: 512.
- Temperature: 0.
- Successful episodes: `30/30`.
- Mean revised reward: `0.4425`.
- Median reward: `0.0`.
- Strong answers at least `0.9`: `13/30`.
- Zero-reward episodes: `16/30`.
- Nonempty answers: `30/30`.
- Valid `finish` termination: `29/30`.
- Plain assistant termination: `1/30`.
- Average tool calls: `5.77`.
- Average wall time: `48.76` seconds.

Artifact:

`results/local_eval/qwen35_4b_harness_30_rollouts.jsonl`

This run validated the current harness and scorer on real trajectories, but it was not an Unsloth GRPO training run. Standard TRL `GRPOTrainer` generates ordinary completions and does not natively execute our multi-turn typed tools.

## Opus audit findings

Claude Opus 4.6 reviewed the toy result, the 30-episode summary, and five representative traces. Its conclusion was:

- The toy Unsloth/TRL GRPO stack passes the smoke gate.
- The FinanceBench harness is not ready for agentic GRPO yet.
- A correct-looking answer that terminates without `finish` can receive zero because answer extraction/termination handling is still too strict.
- A directionally correct answer can receive substantial partial credit even when its retrieved evidence contradicts the claim; this indicates a grounding-score weakness.
- More baseline validation is needed before building the full local agentic GRPO bridge.

Representative issues:

1. One 3M response had detailed, substantively relevant content but ended with plain assistant text rather than `finish` and received `0.0`.
2. One AMD response answered the yes/no surface question correctly but cited diversification evidence rather than the required `16%` customer-concentration fact and received `0.625`.

## What the custom multi-turn GRPO bridge means

The current harness can run multi-turn episodes, and Unsloth/TRL can train on single-turn completions, but those pieces are not connected for policy updates.

The missing bridge must:

1. Sample several assistant continuations from the local Unsloth policy.
2. Parse typed tool calls.
3. Execute `Bm25Tool`, `grep`, `read`, `read_table`, `calculate`, and `finish` against the existing corpus.
4. Append tool observations to the conversation and continue generation.
5. Preserve assistant token IDs, per-token log probabilities, turn boundaries, tool calls, observations, and termination metadata.
6. Compute one final reward per completed trajectory and group-relative advantages.
7. Apply the GRPO clipped objective and KL control to the assistant-generated tokens only; tool observations are context, not trainable actions.

This is an adapter/training integration problem, not a replacement for GRPO or a new reward formula. The existing Tinker environment already contains much of the environment-side logic, but Tinker training is currently blocked by billing, so a local bridge is required for local GRPO.

## Next steps

### 1. Fix termination and answer extraction before GRPO

Score the last nonempty assistant answer even when `finish` is missing, while recording missing finish as a separate operational failure or applying only a bounded cap. Preserve the strict `finish` requirement for training targets, but do not let a correct substantive answer become indistinguishable from an empty answer solely because of parser mechanics. Add regression tests for plain assistant answers, truncated answers, forced finalization, and malformed finish arguments.

### 2. Add evidence-grounding checks

Implement claim-level evidence support using the retrieved/read/grep provenance. Require numerical claims such as `16%` to be supported by evidence, penalize evidence that contradicts the conclusion, and distinguish search snippets from document/page reads. Keep trajectory evidence discovery separate from final cited-claim support.

### 3. Run a larger baseline audit

Run at least 50 additional episodes, preferably with two deterministic seeds or temperatures, and report:

- Mean and confidence interval.
- Reward distribution and within-group variance.
- Finish and nonempty rates.
- Evidence-document/page validity.
- Read/grep coverage versus search-only episodes.
- Tool count, latency, and context length.

Do not start a long GRPO run until the scorer passes manually reviewed termination and grounding cases.

### 4. Build a minimal local rollout bridge

Start with one FinanceBench question and two sampled trajectories. Confirm that both trajectories can complete tool interactions, that assistant token log-probabilities are retained across turns, that rewards differ, and that one optimizer step changes the LoRA adapter without an OOM or context corruption.

### 5. Scale the bridge cautiously

Use a small pilot such as 2–4 questions, 2–4 generations per question, one or two training steps, rank 32, 16k context, and conservative learning rate/KL settings. Compare the post-step policy against the untouched zero-RL baseline on the same questions and on the untouched 42-question evaluation split.

### 6. Only then run the 30-rollout FinanceBench GRPO pilot

The first meaningful local GRPO experiment should use the existing harness, frozen prompts, the validated answer/evidence reward, complete trajectory logging, and a separate SFT-only or zero-RL control. Report both answer quality and operational metrics; never use reward alone as the success claim.

## Blocker fixes and two-seed audit

The two Opus blockers were addressed before the follow-up baseline:

### Termination and answer-quality separation

- The answer scorer now evaluates the last nonempty assistant response even when `finish` is missing.
- Missing `finish` remains a separate `finish_ok` diagnostic rather than turning a substantive answer into an empty-answer reward.
- The direction parser now ignores table labels such as `increase/(decrease)` when determining the narrative conclusion.
- Percentage matching tolerates presentation signs such as `1.7%` versus `-1.7%` when the surrounding conclusion supplies the direction.
- Finish metadata is retained when available.

### Evidence-grounding signal

- Added trajectory evidence extraction from tool outputs.
- Bounded reads, table reads, and grep receive stronger provenance than search snippets.
- Required numerical claims must appear in both the submitted answer and evidence encountered in the trajectory for full grounding credit.
- Unsupported claims reduce grounding, while answer correctness remains the primary term.
- The combined grounded reward is `answer_quality * (0.5 + 0.5 * evidence_quality)`; answer reward and evidence reward are logged separately.
- Added a regression case showing that an AMD answer claiming diversification without the required `16%` concentration fact receives zero evidence credit.

Regression status: `5/5` reward tests pass.

The 30-episode artifact was used to sanity-check the revised logic: the formerly zero-scored 3M narrative now receives answer-quality `1.0` because the scorer no longer mistakes `increase/(decrease)` for the conclusion; the AMD diversification trace receives evidence `0.0` and grounded reward `0.3125` rather than answer-only reward `0.625`.

### 50-episode, two-seed rerun

The same 25 training questions were sampled at temperature `0.2` with seeds `0` and `1`, producing 50 episodes and 25 paired questions. These are evaluation rollouts through the Qwen3.5-4B SGLang endpoint, not GRPO updates.

Seed `0`, 25 episodes:

- Mean answer reward: `0.7040`.
- Mean evidence reward: `0.2461`.
- Mean grounded reward: `0.4686`.
- Strong grounded reward at least `0.9`: `2/25`.
- Zero grounded reward: `7/25`.
- Finish rate: `24/25`.
- Nonempty answer rate: `25/25`.
- Average tool calls: `5.76`.
- Average latency: `49.27` seconds.

Seed `1`, 25 episodes:

- Mean answer reward: `0.7080`.
- Mean evidence reward: `0.3402`.
- Mean grounded reward: `0.5163`.
- Strong grounded reward at least `0.9`: `2/25`.
- Zero grounded reward: `7/25`.
- Finish rate: `24/25`.
- Nonempty answer rate: `25/25`.
- Average tool calls: `5.68`.
- Average latency: `47.64` seconds.

Paired seed comparison:

- Mean grounded difference, seed 1 minus seed 0: `+0.0477`.
- Identical grounded rewards: `13/25`.
- Mean answer reward: `0.7040` versus `0.7080`.
- Mean evidence reward: `0.2461` versus `0.3402`.
- All 50 episodes completed without generator failures.

Combined artifact:

`results/local_eval/qwen35_4b_grounded_50_two_seeds.jsonl`

Per-seed artifacts:

- `results/local_eval/qwen35_4b_grounded_seed0_25.jsonl`
- `results/local_eval/qwen35_4b_grounded_seed1_25.jsonl`

Interpretation: the termination blocker is materially improved because non-finish responses are now scored for answer quality and logged separately; the grounding blocker is exposed rather than hidden, with low evidence scores on many trajectories. The two seeds are reasonably close on answer quality, but the grounded mean is not yet strong enough to justify GRPO training. The next gate is a manual review of the low-evidence/high-answer cases and a small calibration of the evidence component before implementing the local multi-turn GRPO bridge.

## Current go/no-go status after the two-seed audit

Toy Unsloth/TRL stack: GO for further engineering.

Termination handling: PASS for baseline experimentation; retain `finish_ok` as a diagnostic.

Evidence-grounded scoring: IMPLEMENTED, but calibration remains required before using it as the sole GRPO reward.

50-episode baseline stability: PASS as an initial two-seed audit; answer means differ by only `0.0040`, while evidence variance remains substantial.

Full FinanceBench agentic GRPO training: NO-GO until the exact local-policy bridge is validated across multiple questions/seeds with reference KL and finish/evidence diagnostics.

## Minimal FinanceBench-to-LoRA optimizer-step trial

The first bridge trial has now passed the two-trajectory optimizer-step gate.

Implementation:

`harness/financebench_grpo_trial.py`

The trial used the existing FinanceBench harness to generate and score four complete tool-interacting trajectories: two generations for `financebench_id_01226` and two for `financebench_id_03029`. The records preserved questions, tool calls, tool observations, evidence traces, finish metadata, answer reward, evidence reward, grounded reward, and termination metadata.

The bridge then:

1. Grouped trajectories by question.
2. Computed group-relative advantages from grounded rewards.
3. Reconstructed the assistant tool-call/final-answer action spans while retaining tool observations as context.
4. Loaded Qwen3.5-4B with Unsloth bf16 LoRA rank 32 and a 16k context cap.
5. Computed policy log-probabilities over 2,053 assistant action tokens.
6. Completed one clipped-gradient-free first-step policy-gradient update using the group-relative advantages.
7. Saved the adapter checkpoint.

Trial result:

- Rollout trajectories: `4`.
- Valid action sequences: `4/4`.
- Question groups with nonzero reward variance: `2/2`.
- Group rewards for `financebench_id_01226`: `0.8333`, `0.9231`; standard deviation `0.0449`.
- Group rewards for `financebench_id_03029`: `0.8000`, `0.7500`; standard deviation `0.0250`.
- Mean grounded reward: `0.8266`.
- Mean answer reward: `1.0000`.
- Mean evidence reward: `0.6532`.
- Loss contribution: `0.1035`.
- Gradient norm before clipping: `3.8884`.
- Trainable-parameter checksum delta: `24.6860`.
- LoRA rank: `32`.
- Context cap: `16,384` tokens.
- Learning rate: `1e-6`.
- Optimizer steps completed: `1`.
- Adapter save completed successfully.

Artifacts:

- `results/local_eval/financebench_grpo_trial_qwen35_4b_r32_16k/trial_metrics.json`
- `results/local_eval/financebench_grpo_trial_qwen35_4b_r32_16k/rollouts.jsonl`
- `results/local_eval/financebench_grpo_trial_qwen35_4b_r32_16k/adapter-step-1/`

This is a bridge proof-of-life, not yet the final production trainer. Rollouts were generated by the existing Qwen3.5-4B SGLang endpoint and the optimizer update was performed by a local Unsloth/PEFT model; the trial validates harness-to-reward-to-LoRA-update connectivity. The exact local-policy implementation below supersedes it for on-policy testing.

## Exact local-policy FinanceBench optimizer-step smoke test

The new implementation is `harness/financebench_onpolicy_grpo.py`. It uses Unsloth Qwen3.5-4B for both generation and optimization, inserts local tool observations between actions, parses the Qwen3.5 XML tool-call format, and recomputes the generation policy's exact per-token old log-probabilities with a forward pass before applying the clipped GRPO objective. Generation-time warped sampling scores are retained only as diagnostics and are not used as policy log-probabilities.

Run artifact: `results/grpo_runs/exact_local_smoke_01226_g4_v5/`

Configuration and result:

- One FinanceBench training question: `financebench_id_01226`.
- Four fresh local-policy generations in one group.
- Three rollout records used multiple local tool calls; one stopped with a plain assistant answer.
- Rewards: `0.9643`, `0.9063`, `0.0000`, and `0.5000`; group standard deviation was nonzero.
- Mean grounded reward: `0.5926`; mean answer reward: `0.7500`; mean evidence reward: `0.4353`.
- Exact old/new ratio before the update: mean/min/max `1.0/1.0/1.0`; clipped fraction `0.0`.
- Gradient norm: `0.9528`; action tokens: `2,414`; peak GPU allocation: approximately `47.7 GB`.
- Adapter save completed successfully with an SHA-256 checksum.
- Teacher tie-breaker: disabled (`0.0`); reference-KL load: disabled for this mechanics smoke test.

This passes the exact same-policy token/log-probability and nonzero-variance optimizer connectivity gate for one question, but it is not a model-improvement result: the sample is tiny, the model often failed to call `finish`, reference KL was disabled, and the run used three turns/256 generated tokens rather than the full pilot budget. The next gate is a multi-question run with reference KL enabled, full trace capture, and fixed pilot/monitoring IDs.
