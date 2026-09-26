# FinanceBench Agentic Retrieval Journal

Sensitive values, credentials, tokens, and connection strings are intentionally omitted.

## Project purpose

This journal records the work done to build and evaluate a FinanceBench agentic retrieval system, establish reproducible model baselines, generate and judge teacher traces, and prepare for evidence-consistent fine-tuning. The project prioritizes simple, auditable baselines before new reward terms or more complicated retrieval components.

## Timeline

### 2026-09-24 — Harness foundation and corpus indexing

- Located the project at `/home/dreamhunter/Documents/Research/rl-for-ai-research` after the initially assumed lowercase path was not present.
- Reworked the FinanceBench harness around a page-preserving document index.
- Added BM25 prose and table retrieval, filename metadata, page caching, document grep, table search, page-aware reads, table reads, calculation, evidence recording, and answer completion support.
- Added typed tool registration and Tinker-compatible tool handling.
- Preserved the frozen FinanceBench split: 108 training questions and 42 evaluation questions, with one evidence document per question.
- Built the full corpus index: 7,765 tables and 102,518 passages.
- Added page caching under `artifacts/page_cache/`; cold index construction was slow, while warm rebuilds were substantially faster.
- Fixed several integration problems:
  - Tinker `FunctionTool` rejected ordinary dictionaries, so tool-input coercion and correct `ToolCall`/`ToolInput` handling were added.
  - A `return outside function` syntax error in `finance_env.py` was corrected.
  - OpenAI environment/key discovery was repaired without preserving credential values.
  - Responses API calls were updated to the supported interface: removed invalid `max_tool_calls` and changed reasoning configuration to the accepted form.
  - Multi-turn tool execution was changed to preserve assistant messages, previous outputs, call IDs, tool outputs, and state.
- Improved retrieval quality by:
  - Increasing passage length to 2,200 characters with 300-character overlap.
  - Increasing the maximum read size to 8,000 characters.
  - Adding statement-aware table scoring for cash-flow, income, balance-sheet, equity, comprehensive-income, and related pages.
  - Improving numeric extraction and stop-word handling.
- Verified the deterministic answer reward on a formatting variant: `$11588.00` versus `Answer: $11,588.00` returned reward 1.0.

### 2026-09-24 — Initial GRPO attempt and billing block

- Started a preliminary Qwen3.5-9B GRPO run with approximately:
  - 50 requested steps
  - Batch size 4
  - Group size 8
  - Learning rate `2e-5`
  - Eight maximum turns
- Stopped after roughly three iterations before producing a usable checkpoint or contaminating later evaluation.
- Tinker subsequently returned HTTP 402 billing errors, blocking further GRPO work.
- Recorded the decision not to start another expensive Tinker run while billing remains blocked.
- Found six persistent checkpoints from earlier Qwen3.5-9B Base LoRA experiments in Tinker metadata. Their identifiers were not retained because they are connection-like values.
- Corrected a logging-directory issue encountered during training: future runs need the output directory created before using `tee`.

### 2026-09-24 — Teacher-trace generation and judging pipeline

- Created and repaired the teacher-trace generation pipeline.
- Added support for full multi-turn trajectories, tool calls and observations, answers, deterministic rewards, evidence, timing, retries, resume support, custom models, providers, base URLs, and complete conversation-state preservation.
- A Qwen 3.8 Flash Next generation exceeded the provider context limit because retrieved observations were too large; capped observations at 4,000 characters.
- Generated candidate traces from GPT 5.6, Qwen 3.8 Flash Next, DeepSeek V4.1-Flash, GLM-5.3, Inference Hub models, and targeted fallback runs.
- Created filtering, judging, and HTML trace-viewer tools.
- Used Claude Opus 4.6 to judge 349 candidate traces for answer correctness and evidence grounding.
- Selected at most one accepted candidate per question:
  - 349 candidate judgments
  - 100 unique questions selected
  - 91 conservative evidence-consistent traces accepted for strict SFT
- Created `results/teacher_traces/judged_traces.jsonl`, `sft_judged.jsonl`, and `sft_judged_strict.jsonl`.
- Generated targeted GPT 5.6 traces for 13 unresolved IDs; all completed. One previously unresolved case reached reward 1.0, and two others reached 0.7.
- Decided to train first on the 91-trace strict set, while retaining the 100-trace set for ablation.

### 2026-09-25 — Container preparation and Qwen3.5-9B tests

- Safely stopped the unrelated Qwen 3.8 Flash serving container.
- Created the GPU-enabled `qwen-train` container with shared Hugging Face and project mounts.
- Downloaded Qwen3.5-9B into `/hf` and verified the completed snapshot.
- Verified the Sparky Two software stack, including Transformers, PyTorch, NVIDIA GB10, vLLM, Accelerate, and PEFT.
- Tested local Qwen3.5-9B BF16 serving at approximately 12 tokens/second; it was too slow for a full 42-question baseline.
- Ran a small local Qwen smoke evaluation. The tested questions exhausted the turn budget and returned reward 0.0.
- Reduced the output budget from 2,048 to 512 tokens; this did not materially improve latency. A one-question probe still took about 70 seconds and failed to finalize.
- Added `harness/local_eval_qwen35.py` and quarantined stale-DNS rows where reward was unavailable.
- Checked the official Unsloth Qwen3.5 recipe:
  - DGX Spark/Blackwell and `sm_121` support were confirmed.
  - The recipe estimated roughly 22 GB for Qwen3.5-9B BF16 LoRA.
  - An older Unsloth Dockerfile pinned Transformers 4.56.2, conflicting with the current Qwen3.5 Transformers v5 requirement.
  - No official Unsloth Qwen3.5-9B NVFP4 SFT recipe was found.

### 2026-09-25 — Nemotron 3.5 Lightning baseline

- Confirmed Inference Hub availability for `nvidia/nvidia/nemotron-3.5-lightning`.
- Measured smoke-test serving speed at approximately 278 tokens/second, substantially faster than local Qwen BF16.
- Created and fixed `harness/eval_nemotron35.py`.
- Completed the six-turn evaluation:
  - 42 rows, 41 valid and 1 timeout.
  - Mean deterministic reward: approximately 0.1024.
  - Three results at or above 0.9 and two at 0.7.
- Increased the maximum turn budget to ten and reran the complete evaluation:
  - 42/42 valid.
  - Mean deterministic reward: approximately 0.2119.
  - Eight results at or above 0.9.
- Generated a trace viewer for inspection.
- Established Nemotron 3.5 Lightning as the strongest current engineering candidate and primary baseline, while retaining Qwen3.5-9B as a required secondary comparison.
- Kept the smoke-tested questions out of any final paper metric unless a replacement split is created.

### 2026-09-25 — Corrected Qwen Inference Hub evaluation

- Found that Qwen3.5-9B emitted XML-style tool calls inside `reasoning_content`, which the first parser did not recognize.
- Added parsing for those embedded tool calls.
- Quarantined the obsolete preliminary artifact as `results/local_eval/qwen35_9b_inference_hub_eval_10turn_preparser_invalid.jsonl`.
- Reran the corrected evaluation. It still showed excessive searching, including 34 parsed tool calls on one question, one approximately 138.67-second timeout, and mean reward of approximately 0.1220 across the observed run.
- Conclusion: Qwen remains a required comparison, but is a weaker engineering candidate than Nemotron for the current retrieval harness.

### 2026-09-25 — LFM2.5 BF16 investigation

- Identified `LiquidAI/LFM2.5-8B-A1B` as the requested LFM model.
- Searched available hosted endpoints and found no usable LFM/Liquid endpoint in the checked Inference Hub or DeepInfra catalogs.
- Downloaded the native LFM2.5-8B-A1B snapshot to `/hf`.
- Verified that vLLM contained LFM-related model files.
- Built `harness/lfm25_server.py`, a FastAPI/Transformers OpenAI-compatible local server.
- Fixed LFM message-history handling so assistant content was preserved alongside parsed tool calls.
- Started a temporary native BF16 server and waited through delayed startup until it became healthy.
- Ran a one-question BF16 probe: approximately 150.9 seconds, 15 tool calls, long reasoning, inefficient repeated searching, and a computed value around 824.96. The route was too slow for a 42-question baseline.
- Stopped the BF16 Transformers server.

### 2026-09-25 — Abandoned LFM NVFP4 route

- Downloaded `sakamakismile/LFM2.5-8B-A1B-NVFP4` and attempted a vLLM/ModelOpt deployment.
- The service did not return a confirmed `/v1/models` response.
- The route was considered insufficiently trustworthy for the experiment.
- After the user requested cleanup before redeployment:
  - Removed the failed LFM and unrelated experiment containers.
  - Deleted only the unused LFM NVFP4 model directory.
  - Verified that no stray SGLang, vLLM, or LFM serving process remained.
  - Reclaimed GPU memory before deploying the replacement.
- The native BF16 LFM target was retained because it was still needed as the speculative-decoding target.

### 2026-09-25 — LFM2.5 DSpark and SGLang deployment

- Downloaded LiquidAI LFM2.5-8B-A1B-DSpark and used SGLang because DSpark is a target-plus-drafter speculative-decoding configuration.
- Created the isolated lfm-sglang container and installed SGLang 0.5.18, then upgraded to 0.5.20.
- Initial launches failed or did not reach a confirmed endpoint. Transformers lacked the expected LFM draft registration, so SGLang source and the native DSpark path were inspected.
- Investigated FlashInfer compatibility and installed flashinfer-python 0.6.18.
- Patched a local DSpark configuration copy so its architecture used the available SGLang-compatible Qwen3DSparkModel path.
- Relaunched after cleanup; target and drafter loaded and GB10 CUDA-graph compilation completed.
- Verified /v1/models, application startup, warmup, and readiness on port 18350. A non-fatal warning reported no tuned GB10 MoE kernel configuration.

### 2026-09-25 — DSpark probe and native tool-call parsing

- The first one-question probe returned reasoning but zero structured calls because LFM emitted native calls as raw delimiter text. Increasing the output budget did not fix this.
- Patched the harness to parse LFM native tool-call text safely.
- The retry completed in about 60.9 seconds with 11 parsed tool calls, answer 0.83, and deterministic reward 1.0.

### 2026-09-25 — Full LFM DSpark evaluation

- Ran all 42 evaluation questions with 12 turns, 4096 maximum generation tokens, temperature 0.0, one attempt per question, full trajectory logging, and a separate failure output.
- All 42 executions completed technically; no rows entered the failure file.
- Mean deterministic reward was 0.0833: 38 at 0.0, one at 0.7, two at 0.9, and one at 1.0.
- Three results were strong at or above 0.9.
- 32 of 42 traces had empty final answers. Average tool calls were about 10.69 and average wall time was about 41.1 seconds per question.
- The primary failure was control flow: repeated searching, turn-budget exhaustion, and failure to emit a final answer, not server reliability.
- Full artifact: results/local_eval/lfm25_dspark_eval.jsonl

### 2026-09-25 — Opus 4.6 judging of LFM traces

- Extended the judge to accept custom JSONL input and include empty-answer traces.
- Claude Opus 4.6 judged all 42 LFM trajectories using the question, gold answer, candidate answer, and bounded retrieved evidence.
- Five traces passed as semantically correct and grounded; 37 failed. The 32 empty answers failed as expected.
- Correctness scores: 34 at 0, 3 at 1, 1 at 3, and 4 at 4. Grounding scores: 35 at 0, 4 at 1, and 3 at 2. Three gold-answer issues were flagged.
- One deterministic-reward-0 trace was accepted semantically, showing that the deterministic metric is stricter than semantic review.
- Judgments: results/local_eval/lfm25_dspark_opus_judged.jsonl
- Accepted traces: results/local_eval/lfm25_dspark_opus_selected.jsonl
- Conclusion: LFM shows limited competence, but current performance is dominated by failure to finalize rather than total retrieval failure.

## Implementation findings

- The harness did not expose a strongly terminating finish or submit tool to LFM. The model could continue searching until the turn budget ended.
- A reminder alone is unlikely to solve this because the model repeatedly selects retrieval tools.
- The proposed structural fix is a typed submit_answer(answer, evidence) terminal tool, immediate termination when called, duplicate-search guards, a retrieval-turn budget, and a forced final-answer phase with tools disabled after retrieval.
- Raw model output and parser diagnostics should remain preserved for audit.

## Current artifacts

- Harness: financebench_harness.py, finance_env.py, eval_agent.py, generate_teacher_traces.py, generate_targeted_traces.py, filter_teacher_traces.py, judge_teacher_traces.py, eval_nemotron35.py, eval_lfm25_local.py, and lfm25_server.py.
- Teacher data: results/teacher_traces/judged_traces.jsonl, sft_judged.jsonl, and sft_judged_strict.jsonl.
- Baselines: results/local_eval/nemotron35_lightning_eval_10turn.jsonl, qwen35_9b_inference_hub_eval_10turn.jsonl, and lfm25_dspark_eval.jsonl.
- LFM analysis: results/local_eval/lfm25_dspark_opus_judged.jsonl and lfm25_dspark_opus_selected.jsonl.
- Corpus: data/financebench_merged.jsonl, split.json, filings/, and artifacts/page_cache/.
- Runtime: qwen-train remains available; lfm-sglang is the active LFM container; the uncertain NVFP4 checkpoint and failed serving containers were removed.

## Baseline snapshot

- Nemotron 3.5 Lightning, ten turns: 42/42 valid, mean deterministic reward about 0.2119, and 8 strong results at or above 0.9.
- Qwen3.5-9B Inference Hub: mean reward about 0.1220 in the corrected observed run, with one long timeout and substantial over-searching.
- LFM2.5-8B-A1B DSpark: 42/42 technically completed, mean deterministic reward 0.0833, 5/42 semantically accepted by Opus 4.6, and 32/42 empty final answers.
- The strongest immediate LFM opportunity is finalization and control-flow repair, not another model or quantization change.

## Not yet completed

- The submit_answer/finalization redesign has not yet been implemented.
- No complete post-redesign LFM pilot exists.
- No local BF16 LoRA/SFT run has started.
- Nemotron SFT format conversion remains incomplete.
- GRPO remains blocked by Tinker billing HTTP 402.
- The final paper evaluation split still needs to exclude the three smoke-tested questions or replace them.
- FinAgentBench adaptation remains deferred.

## Recommended next sequence

1. Preserve the current LFM run for audit, but do not treat it as clean SFT data.
2. Add submit_answer as a typed terminal tool with evidence fields.
3. Add repeated-tool and retrieval-budget guards, followed by a forced final-answer call.
4. Run a five-question LFM pilot and inspect raw traces plus Opus judgments.
5. If finalization improves materially, rerun the 42-question LFM baseline.
6. Compare repaired LFM against the ten-turn Nemotron baseline before selecting an SFT base.
7. Train first on the 91 strict Opus-filtered traces, preferably with BF16 LoRA on the selected non-draft target.
8. Keep the 100-trace set as an ablation and preserve rejected and candidate traces for audit.


## 2026-09-25 — Calibrated reward redesign audit

Replaced the brittle FinanceAnswerReward path with a calibrated answer-quality verifier in `harness/financebench_harness.py`. The new scorer preserves verbose answers after the final `Answer:` marker, handles markdown answer markers, parses yes/no conclusions, normalizes numeric units and percentages, tolerates rounding and scale, checks sign and directional contradictions, and removes finish/format bonuses from quality reward. `FinanceAnswerReward` now reports quality, conclusion, detail, format, finish usage, and nonempty-answer metrics separately; the typed `finish` tool remains registered in `finance_env.py`.

On the 91 Opus-selected strict traces, old rewards were 52 zero, 12 at 0.7, 9 at 0.9, and 18 at 1.0 (mean 0.3791). New rewards were 10 zero, 22 at 0.95, 1 at 0.975, and 58 at 1.0 (mean 0.8777). All 46 traces previously scoring zero but judged correctness=4 and grounding=2 now score at least 0.95; minimum target score is 0.95. Added `harness/test_reward_design.py` with recovery and adversarial tests; all three tests pass. Audit artifacts: `results/teacher_traces/sft_judged_strict_reward_audit.jsonl` and `results/teacher_traces/sft_judged_strict_reward_audit_summary.json`.


### Follow-up reward audit

Reviewed the 10 traces that remained at zero after the first pass. Eight were valid financial numeric answers expressed in millions/billions or with rounding differences, one was a valid concise categorical `None` answer, and one (`financebench_id_00606`) was directionally correct but only partially grounded; the verifier now gives it `0.625` rather than treating it as correct. After unit-scale variants, rounded numeric matching, concise categorical handling, and scalar conclusion handling, all 91 traces are nonzero: 68 at 1.0, 21 at 0.95, one at 0.975, and the partially grounded trace at 0.625. Final mean quality reward is `0.9840659341`; the 46 fully correct-and-grounded known traces have minimum `0.95`. Regression and adversarial tests pass.
