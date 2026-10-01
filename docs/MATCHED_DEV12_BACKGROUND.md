# Matched calculator-free dev12 evaluation

User authorized GitHub push, non-destructive repository cleanup, and background
base plus existing SFT adapter evaluation. Each phase uses the same frozen dev12,
8 turns, 1024 completion tokens per call, temperature 0, seed 0 and six tools.
Serving stays on the same SGLang process; only base versus LoRA model selection
differs. Runs are sequential to avoid competing inference affecting latency.

Run `harness/run_matched_dev12.py` from an immutable pushed code snapshot with
explicit corpus root, fresh output directory, endpoint, model IDs, commit and
adapter hash. Credentials are read through the existing private environment,
never saved to Git. The supervisor creates protocol.json, per-phase logs/raw
traces, status.json, and a latest-only v6 additive report. It does not restart
serving, train, delete artifacts or evaluate eval42.

The generation runner still records v5 diagnostic scores. Only the post-hoc v6
scores are presented in the new report. Provider failures and gold conflicts
remain unresolved, not silently zero. Every phase must contain the exact 12
unique frozen IDs. Generation errors remain in the denominator and are reported.

Cleanup keeps virtual environments and bulky raw teacher artifacts local and
ignored. No checkpoint, source corpus, raw evidence or pre-existing user file is
deleted. Code, tests, documentation and compact saved-dev reports are versioned.
The remote original working checkout is not reset or overwritten; the run uses
a separate code snapshot and reads its corpus from the original checkout.
