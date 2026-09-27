# FinanceBench reward redesign

The source of truth is `harness/reward_calculation.py`. The redesigned Tinker reward is enabled with `JUDGE_BACKEND=deepinfra` and a runtime-only `DEEPINFRA_API_KEY`; no credential is stored by the harness.

The reward contract is: a valid `finish(answer, evidence_document, evidence_page)` submission is required; deterministic contradiction, decision, numeric, and unit gates run first; DeepSeek V4.1 Flash is consulted only for qualitative residuals; and evidence grounding remains deterministic. For answer quality `Q`, a high-confidence `entailed` judge result passes the semantic residual, while `contradicted`, `insufficient`, or `ambiguous` does not. The final grounded reward is `R = clip(Q * (0.5 + 0.5 * E), 0, 1)`.

The evaluator caches judgments by a SHA-256 hash of model, question, gold answer, candidate answer, and bounded evidence. Provider errors are recorded in metrics and fall back to the deterministic score rather than being silently converted to zero. The cache contains judgment data only, never API credentials.

Before a new long GRPO run, calibrate the DeepSeek judge on a labeled residual set containing paraphrases, wrong directions, wrong numbers, unsupported claims, and verbose keyword stuffing. Do not start that run until live DeepInfra authentication and the calibration report are available.
