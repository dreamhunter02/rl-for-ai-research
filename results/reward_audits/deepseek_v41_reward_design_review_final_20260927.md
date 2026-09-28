# DeepSeek V4.1 Flash reward-design review (final)

## 1. Diagnosis and verdict

**Verdict: v2 is a regression on the objective that matters, and the run is not yet interpretable as a reward-learning result.** It is a finishing-behavior result with a broken reward signal.

Key facts:
- Finish rate barely moved (24.7% → 25.5%). That is noise-level, not a win.
- Positive trajectories collapsed 319 → 187 while mean reward fell 0.2573 → 0.1475. The reward got *sparser and harsher*, not better-shaped.
- 616/864 zero-reward and 38/108 all-zero groups means GRPO has almost no within-group advantage signal on ~35% of groups. With group 8 and batch 4, all-zero groups contribute zero gradient; mixed groups carry the run. Effective learning signal is roughly half the nominal batch.
- 61 trajectories at −0.1 (penalty) plus 220 finishes means ~583 trajectories are unfinished and mostly zero. The dominant learned behavior is "don't finish, don't get penalized much," which is a degenerate local optimum.
- The pretraining audit is the smoking gun: **0/91 teacher traces contain terminal finish calls**, and strict finish reward was 0/91. So the model was never shown the target behavior, yet finish was made a hard requirement. That mismatch alone explains the flat finish rate.
- 8 false positives against strict Opus labels means your correctness labeler is not clean; if A is derived from the same labeler, you are training on ~9% label noise at the top of the reward.

**Root cause:** reward-v2 changed the *shape* (factorization, caps, penalties) without fixing the *data* (no finish demonstrations) or the *credit assignment* (sparse terminal reward over 8 turns). You optimized a harder objective with the same weak signal.

## 2. Is factorized reward sound?

**Yes, factorization is the right direction, but the proposed weights and caps are hackable as written.**

- `R = 0.10F + 0.65A + 0.25E` is sound in principle: it separates "did it terminate," "is it right," "is it grounded." That is better than a monolithic reward.
- **Hack risk 1:** A unfinished trajectory capped at 0.25 can still earn 0.25A + 0.25E = up to 0.25 without finishing. If finishing is hard, the policy learns to farm partial credit forever. Cap must be *below* the expected value of a finished-but-wrong trajectory, or finish must gate A/E entirely.
- **Hack risk 2:** E is the easiest to game (cite retrieved text verbatim, retrieve a lot). 0.25E with "useful retrieval" shaping rewards tool calls, not grounding. Require E to be *conditional on A>0* or on a contradiction check.
- **Hack risk 3:** "numeric zero is valid" plus "hard contradictions set A=0 but keep E" means a contradictory answer can still collect 0.25E. That rewards fluent-but-wrong grounding.
- **Hack risk 4:** trajectory reward for "avoiding repeated tools" and "termination" is a pure process reward; it will be maximized by short, tool-light, early-terminating trajectories that never answer.

**Fix:** make F a *gate*, not an additive term. `R = F * (0.65A + 0.25E) + 0.10F`, with A=0 on contradiction and E capped at 0.25 only when A>0. Keep trajectory shaping ≤0.05 and only as a tie-breaker inside a group, never as a standalone reward.

## 3. SFT, trajectory reward, or both?

**Both, but SFT first and narrowly scoped.**

- **91 traces are enough for format/finish behavior, not broad reasoning.** Do not expect SFT on 91 traces to teach finance reasoning. It can teach: emit a terminal finish call, structure evidence, stop repeating tools. That is exactly the gap.
- **Create finish-annotated SFT data** by taking the 91 teacher traces and appending a canonical terminal finish call at the correct turn (the turn where the answer is complete). Have a strong model (Opus-4 or DeepSeek V4.1) generate the finish call conditioned on the trace, then verify it parses and references the final answer. Target ~91–200 examples, all with valid finish.
- **Trajectory reward:** keep it, but small (≤0.05) and only for: retrieval that was actually cited, evidence carried forward across turns, no repeated identical tool call, and a valid terminal call. Never reward termination alone.
- **Order:** SFT on finish-annotated traces → short GRPO with gated factorized reward → full run. Do not run full GRPO on a policy that has never emitted a finish call.

## 4. Minimum decisive ablation plan

Run these as small GRPO runs (same 108 questions, batch 4, group 8, max 8 turns), each ~10–15 steps, before any full run:

1. **SFT-only baseline:** finish-annotated SFT, no GRPO. Measure finish rate and A. This isolates whether SFT alone fixes finishing.
2. **Gated reward, no trajectory shaping:** `R = F*(0.65A+0.25E)+0.10F`. Tests whether factorization alone helps.
3. **Gated reward + trajectory shaping (≤0.05).** Tests whether shaping adds or hacks.
4. **Label-noise control:** rerun (2) with the 8 false positives removed from A. Tests sensitivity to labeler error.
5. **Finish-gate ablation:** (2) but with A/E allowed unfinished (your current cap). Confirms the hack hypothesis.

Decisive criteria: if (1) does not raise finish rate above ~60%, SFT data is insufficient. If (2) does not raise positive trajectories above v11's 319, the reward is still mis-shaped. If (3) beats (2) on A but not on finish, shaping is hacking.

## 5. Mandatory metrics and thresholds

Track per run, per group:
- **Finish rate:** target ≥60% by step 15; abort if <35% at step 10.
- **Positive-trajectory count:** target ≥300/864 (match v11); abort if <200.
- **All-zero groups:** target ≤15/108;
