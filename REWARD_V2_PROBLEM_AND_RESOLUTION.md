# FinanceBench Reward-v2: Problem Statement and Proposed Resolution

**Date:** 2026-09-27  
**Student model:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`  
**Training method:** Tinker GRPO with `nemotron3_ultra`  
**Status:** Review document; no new training launched from this proposal

## Executive summary

The full Reward-v2 run completed technically, but it did not produce a successful learning result. The main issue is not simply that there were too few rollouts: the reward is sparse and poorly aligned with the policy's demonstrated behavior. The policy was required to emit a terminal `finish` call, yet none of the 91 teacher traces contained one, so the training objective demanded behavior absent from the supervision. Reward-v2 slightly improved finishing but sharply reduced the number of trajectories carrying positive learning signal.

The recommended next step is not another blind 27-step GRPO run. First teach the required finish/answer format with a small, finish-annotated SFT set, then run short controlled reward ablations with dense diagnostics before committing to a new full run.

## Problem sent to DeepSeek

We asked DeepSeek V4.1 Flash through DeepInfra to review the following experiment and proposed reward redesign:

- Full run: 108 training questions, 27 optimizer steps, 4 questions per batch, 8 rollouts per question, 864 trajectories.
- Model: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.
- Maximum 8 turns, valid terminal finish required, learning rate `1e-5`, seed `0`.
- Reward-v2 used deterministic answer gates, evidence grounding, a residual DeepSeek semantic judge, and constant-reward-group removal.
- Proposed redesign: independent `F=valid_finish`, `A=answer correctness`, and `E=evidence grounding`, initially `R=0.10F + 0.65A + 0.25E`; numeric zero should be valid; hard answer contradictions should not erase evidence credit; unfinished trajectories could earn answer/evidence credit with a cap; small trajectory/process rewards and SFT warm-start were also considered.

The review questions were whether the bottleneck was reward design, trajectory/process reward, question diversity, optimization, or sample size; whether the factorized reward was safe; whether to use SFT, trajectory reward, or both; whether 91 traces were enough; and what minimum ablation plan should precede another full run.

## Observed results

### Reward-v2 full run

| Metric | Result |
|---|---:|
| Training questions | 108 |
| Optimizer steps | 27 |
| Rollouts per question | 8 |
| Total trajectories | 864 |
| Valid finish trajectories | 220 / 864 (25.5%) |
| Positive-total-reward trajectories | 187 / 864 |
| Zero-total-reward trajectories | 616 / 864 |
| `-0.1` process-penalty trajectories | 61 / 864 |
| Mean authoritative total reward | 0.1475 |
| All-zero question groups | 38 / 108 |
| Mixed-reward question groups | 70 / 108 |
| DeepSeek cached records | 32 |
| DeepSeek errors | 0 |
| Trace events | 38,760 |

The prior grounded v11 run also had 864 trajectories and produced 213 valid finishes (24.7%), but had 319 positive trajectories and mean reward 0.2573. Therefore Reward-v2 improved finishing only marginally while reducing positive training signal substantially.

### Teacher-trace audit

The 91 saved teacher traces produced:

- Mean answer quality: `0.9893`.
- Mean evidence quality: `0.5545`.
- Mean grounded reward without the finish gate: `0.7699`.
- `91/91` positive under answer scoring.
- `83/83` Opus-4 fully-correct cases retained.
- Eight false positives against the strict Opus-4 label, indicating nonzero label noise.
- `0/91` terminal finish calls.
- `0/91` strict finish-gated rewards.
- Only `1/91` traces invoked DeepSeek; no judge errors occurred.

The audit shows that the answer/evidence scorer can recognize good content, but the teacher data does not teach the terminal action required by the GRPO environment.

## Diagnosis

### 1. Reward sparsity is the dominant immediate problem

Reward-v2 produced 616 zero-reward trajectories and 38 all-zero groups. An all-zero group contributes no useful GRPO advantage signal, and groups with one positive sample among seven zeros have high-variance, weak credit assignment. The nominal 864 rollouts therefore overstate the effective learning signal.

### 2. The finish objective is mismatched with the data

The environment requires a terminal `finish` call, but all 91 teacher traces lack one. The reward is therefore correctly strict according to its contract but operationally misaligned with the policy's demonstrations. SFT or another explicit format intervention is needed before expecting GRPO to discover the finish behavior.

### 3. Reward-v2 became harsher without becoming denser

The finish rate barely changed, while positive trajectories and mean reward fell compared with v11. This suggests the new gates removed reward mass faster than they created useful distinctions between partially successful trajectories.

### 4. Question diversity and optimization are secondary confounds

The 108-question training set is not large enough for a strong generalization claim, and 27 steps is a small optimization budget. However, these cannot be diagnosed cleanly until the reward produces a usable within-group signal. More rollouts alone will not repair a reward that produces mostly zeros.

### 5. Judge coverage is not the current bottleneck

DeepSeek had no observed provider errors, but it was used rarely because deterministic gates resolved most cases. The semantic judge is not responsible for the low reward mass; the main issue is the finish/data mismatch and sparse credit assignment.

## Suggested resolution

### A. Factorize the reward, but preserve safety boundaries

Keep separate components and log them independently:

- `F`: valid terminal finish, binary.
- `A`: answer correctness, including numeric, sign, unit, date, direction, and residual semantic checks.
- `E`: evidence grounding and citation quality.
- `T`: small trajectory/process signal.

The basic factorization is sound, but the exact combination needs testing. DeepSeek's safer recommendation is:

```text
R_answer = F * (0.65 * A + 0.25 * E) + 0.10 * F
R_total  = R_answer + T
```

with `T <= 0.05` and only for verifiable behaviors.

Important rules:

- A numeric `0` is a valid parsed answer, not a missing answer.
- Hard numeric, sign, unit, direction, and explicit contradiction checks may force `A=0`.
- A hard answer contradiction must not be treated as a correct answer merely because the evidence is relevant.
- Evidence should not be allowed to dominate answer correctness; consider capping `E` when `A=0`.
- Do not reward termination by itself; a short empty answer must not earn process credit.
- Do not let repeated retrieval, verbose citations, or tool-call volume generate reward.
- Use DeepSeek only for residual semantic answer/evidence cases and retain deterministic hard vetoes.

Our earlier proposal to award substantial answer/evidence credit to unfinished trajectories is useful as an ablation, but it is potentially hackable: the policy may farm evidence or partial answer credit while avoiding the terminal action. It should not become the default before comparison against a gated version.

### B. SFT the finish and answer format before GRPO

The 91 traces are enough for a narrow format intervention, not for teaching broad FinanceBench reasoning.

Recommended SFT target:

- Teach the valid terminal `finish` call.
- Teach the expected answer/evidence structure.
- Teach stopping after the answer is complete.
- Do not claim that 91 examples are sufficient to learn general financial reasoning.

Create finish-annotated examples by combining:

1. The 91 teacher traces with a verified canonical finish call appended at the correct turn.
2. A subset of the 220 successful GRPO trajectories that already contain valid finish calls.
3. A held-out format set not used for SFT selection.

A practical initial target is approximately 91–200 finish-annotated examples. Every synthetic or appended finish call must be parser-validated and checked against the final answer and evidence.

### C. Add trajectory reward cautiously

Trajectory reward is warranted, but it should be a small tie-breaker rather than a replacement for answer correctness:

- `T <= 0.05` total.
- Reward evidence that was actually retrieved and then cited.
- Reward evidence carried correctly across turns.
- Penalize repeated identical tool calls.
- Reward a valid terminal call only when the answer is present and grounded.
- Never reward tool-call count, retrieval volume, or termination alone.
- Track tool calls and evidence length to detect reward hacking.

### D. Run a decisive ablation ladder

Do not launch another full 27-step run until these smaller tests are complete:

1. **SFT-only format baseline:** finish-annotated SFT, no GRPO; measure finish rate and answer quality.
2. **Gated factorized reward:** `R = F*(0.65A+0.25E)+0.10F`, no trajectory reward.
3. **Gated reward plus trajectory shaping:** add `T <= 0.05` only for verified process behavior.
4. **Label-noise control:** remove or relabel the eight false positives against strict Opus labels and repeat the gated test.
5. **Unfinished-credit ablation:** test the earlier capped unfinished reward only as a controlled comparison, not as the assumed solution.

Use approximately 10–15 GRPO steps per short ablation with the same 108-question train split, batch 4, group 8, and max 8 turns so the comparisons remain controlled.

## Mandatory metrics and decision thresholds

Report these per run and per question group:

- Valid-finish rate overall and by question.
- Positive-trajectory fraction.
- All-zero-group count.
- Mean and median total reward.
- Answer quality and evidence quality separately.
- Finish reward, answer reward, evidence reward, and trajectory reward separately.
- Hard-veto counts by reason: direction, sign, number, unit, date, contradiction.
- DeepSeek usage, verdict, confidence, cache hit, and provider errors.
- Tool-call count, repeated-tool-call count, turns to finish, maximum-turn exits, parse errors, and token-limit exits.
- Per-question reward variance and number of nonzero samples in each group.
- Holdout answer accuracy and evidence quality against the untouched 42-question split.

Suggested gates for proceeding to another full run:

- Finish rate above approximately 60% by step 15 after SFT.
- At least 300 positive trajectories in an 864-trajectory comparison run, matching or exceeding v11.
- Fewer than 15 all-zero groups out of 108.
- No increase in hard numeric/directional false positives.
- No evidence of reward hacking through tool volume, copied citations, or premature finish calls.
- Improvement on the 42-question holdout against the untrained base-model baseline.

## Bottom line

The full Reward-v2 run should be classified as a useful diagnostic and a training regression, not as evidence that Nemotron cannot learn FinanceBench. The immediate fix is to align supervision with the environment's finish protocol, make reward components observable and less sparse, and test trajectory shaping only as a small controlled addition. SFT first, then short reward ablations, is the lowest-risk path; another full GRPO run should wait until the finish behavior and positive-group coverage improve.

## Source artifacts

- Full-run report: `results/tinker_runs/nemotron35_lightning_grpo_reward_v2_full_20260927/EXPERIMENT_REPORT.md`
- Full-run metrics: `results/tinker_runs/nemotron35_lightning_grpo_reward_v2_full_20260927/tinker_log/metrics.jsonl`
- Full-run rollout summaries: `results/tinker_runs/nemotron35_lightning_grpo_reward_v2_full_20260927/tinker_log/iteration_*/train_rollout_summaries.jsonl`
- Teacher audit summary: `results/reward_audits/teacher91_reward_v2_summary_final.json`
- DeepSeek detailed review: `results/reward_audits/deepseek_v41_reward_design_review_final_20260927.md`
