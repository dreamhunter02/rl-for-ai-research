# Model Selection for FinanceBench On-Policy GRPO

Date: 2026-09-27

## Decision

Keep Qwen3.5-4B as the local trainable student, but do not launch another long GRPO run yet. First add deterministic process scaffolding and a small teacher-derived process signal, then run a short diagnostic; use Nemotron 3.5 Lightning as the strongest external engineering baseline/teacher candidate, not as the immediate Unsloth student.

The Qwen4B failure pattern is currently a control-and-scaffolding problem as much as a raw-capacity problem: it sometimes finds a correct answer, but often repeats searches, stops without `finish`, or reaches the turn limit without evidence. The aborted 17-rollout pilot is diagnostic rather than a clean capability benchmark because it was stopped before its optimizer step and used no process guidance.

## Evidence

### Qwen3.5-4B

- Two-seed 50-episode harness audit: answer reward `0.7040` and `0.7080`; grounded reward `0.4686` and `0.5163`.
- Clean one-question exact local-policy smoke: four fresh trajectories, rewards `0.9643`, `0.9063`, `0.0000`, and `0.5000`; three trajectories used tools; exact old/new ratio was `1.0`; gradient norm was `0.9528`.
- Strengths: locally loadable through Unsloth, exact same-policy token capture works, bf16 LoRA rank 32 fits comfortably on GB10, and the reward has useful variance.
- Weaknesses: unassisted local generation can repeat searches, omit `finish`, and spend many turns without evidence; the exact local smoke used only one question.

### Qwen3.5-9B

- Revised answer reward: approximately `0.2018` on 42 questions, with `8/42` strong answers.
- Local serving was approximately 12 tokens/second and earlier six-turn pilots frequently exhausted their turn budget without a correct final answer.
- Strengths: larger capacity in principle.
- Weaknesses: current measured retrieval-agent behavior is worse than Qwen4B's clean two-seed harness audit, and local rollout cost is substantially higher; increasing size alone is not justified by current evidence.

### Nemotron 3.5 Lightning

- Revised answer reward: approximately `0.4238` on 42 questions, with `18/42` strong answers.
- Ten-turn baseline was stronger than the shorter baseline under the revised scorer and was the best current external engineering candidate before local-policy bridge work.
- Strengths: better tool-use baseline and fast external inference.
- Weaknesses: it is not currently integrated into the Unsloth/TRL local training path, so it cannot immediately replace Qwen4B for exact on-policy LoRA GRPO.

### LFM2.5-8B-A1B DSpark

- Revised answer reward: approximately `0.1179` on 42 questions, with `5/42` strong answers.
- Frequent empty or incomplete responses make it unsuitable as the current student.

## Revised plan

1. Preserve the exact local Qwen4B bridge and the captured one-step smoke artifacts.
2. Add loop detection and a concise guardrail observation after repeated equivalent searches; log every intervention.
3. Add diagnostic process components for useful tool transitions, evidence acquisition, calculator use, premature answering, and repetition.
4. Use the 91 strict teacher traces to derive weak phase-level guidance (retrieval, evidence, calculation, conclusion), not exact tool-order imitation or raw hidden chain-of-thought distillation.
5. Run a short 5-question process-guided diagnostic with group size 2 before another 30-question run; require at least one useful tool transition and nonzero reward variance.
6. Compare four conditions on the same questions/seeds: base Qwen4B, guardrails only, process reward only, and guardrails plus bounded teacher process signal.
7. Only if tool-loop rate, finish rate, evidence reward, and grounded reward improve should the 30-question group-4 pilot be repeated.

## What not to do

- Do not conclude that Qwen4B lacks all necessary reasoning capacity from the aborted 17-rollout sample.
- Do not switch to Qwen9B merely because it is larger; its measured baseline and latency do not support that choice.
- Do not replace the local student with Nemotron without first establishing a trainable Unsloth/TRL path.
- Do not force raw `<thought>` or hidden chain-of-thought tokens into the student target. If planning is exposed, use a short structured `plan` field immediately before an action and score whether it predicts the next useful phase, not whether it resembles a teacher's wording.
- Do not let process reward override answer/evidence correctness; keep it auxiliary and report its components separately.
