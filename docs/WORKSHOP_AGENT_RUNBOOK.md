# Agent runbook: produce the ICAIF RL4LLM paper evidence

Read `docs/REVIEW_FIXES.md` and `NEMOTRON_FINANCEBENCH_GRPO_EXECUTION_BRIEF-3.md` first. Execute the following work on the machine with the filings and Tinker access. The code is an implementation, not evidence of a training gain. Never fill missing paper cells with expected results.

## 1. Freeze the implementation and verify the real SDK

Apply the patch against f29df1a in an isolated checkout. Keep upstream commit `361d3f2337ed55576c241866353e4a9b72a9e9b6` for B0 and historical scorer comparisons. Do not merge or submit the paper automatically.

```bash
python -m venv .venv-workshop
source .venv-workshop/bin/activate
python -m pip install -r requirements-workshop.txt
PYTHONPATH=harness python -m unittest discover -s tests -v
python -m compileall -q harness
mkdir -p results/paper_2026_rl4llm
python -m pip freeze > results/paper_2026_rl4llm/dependencies.lock.txt
```

All SDK contract tests must execute on this machine; skipped SDK tests are not a passed runtime gate. The development workspace lacked Tinker/cookbook and could only run offline tests. The cookbook Git commit is pinned, while the resolved package versions must be archived using the lock command above. Confirm Nemotron renderer round trips, generated-token masks, one recoverable content parse error, terminal finish and fresh sampling-client weights with a small local/provider smoke run before GRPO. No training has been launched by the code-writing task.

## 2. Prepare labels, splits and corpus

```bash
python harness/workshop_prepare.py prepare --split split.json --split-manifest results/paper_2026_rl4llm_agents/split_manifest.json --out artifacts/workshop --seed 0
```

This preserves the recorded 96/12/42 question IDs and reports document overlap. The existing train/dev split is not document-disjoint; disclose that limitation. Omitting --split-manifest creates a different document-grouped split and requires a separately frozen protocol. Read `artifacts/workshop/protocol.json`, document train/eval overlap, and freeze the IDs before comparing methods.

Create `artifacts/workshop/targets.reviewed.json` from the generated draft. Every entry remains `reviewed: false` until someone verifies it against source text. Record reviewer identity/method and date in `review_notes`; do not claim human review for automated checks. Numeric targets need value, unit, scale, precision and derived status. Boolean targets are yes/no-only; answers requiring further facts use text. Text targets have exact aliases and required facts. Initial workshop grading is binary for fully correct numeric/boolean/text answers; it does not award unverified partial credit.

For each support span, specify document_id, one-based page, exact quote and claim ID. Spans sharing a claim ID are alternatives; separate claim IDs must all be supported. PDF extraction whitespace may differ from the dataset's evidence string: verify and replace the quote using the actual indexed page, not approximate matching. Derived targets additionally need a reviewed `expression` and `operands`, each with value, unit, scale, metric, period and its own supporting spans. The model's variable names may differ, but its expression structure and operand semantics must match the reviewed derivation. Equivalent alternative algebra not encoded in the rubric receives no provenance credit; audit this limitation before freezing the evaluator.

```json
{
  "answer_type": "numeric", "value": "20", "unit": "percent", "scale": "ones",
  "precision": 1, "derived": true, "reviewed": false,
  "expression": "profit/revenue*100",
  "operands": {
    "profit": {"value": "20", "unit": "USD", "scale": "million", "metric": "Profit", "period": "2023", "support": [{"document_id": "EXAMPLE", "page": 2, "quote": "Profit in 2023 was USD 20 million."}]},
    "revenue": {"value": "100", "unit": "USD", "scale": "million", "metric": "Revenue", "period": "2023", "support": [{"document_id": "EXAMPLE", "page": 2, "quote": "Revenue in 2023 was USD 100 million."}]}
  },
  "support": []
}
```

That is a schema example, not a FinanceBench label. Keep all target values, evidence annotations and derived flags out of prompts.

```bash
export FINANCEBENCH_ROOT="$PWD"
export FINANCEBENCH_SPLIT="$PWD/artifacts/workshop/split.json"
export FINANCEBENCH_TARGETS="$PWD/artifacts/workshop/targets.reviewed.json"
export FINANCEBENCH_PDF_PAGES=1
python harness/workshop_prepare.py preflight --split "$FINANCEBENCH_SPLIT" --targets "$FINANCEBENCH_TARGETS" --corpus --out results/paper_2026_rl4llm/preflight.json
python harness/workshop_scorer_audit.py --out results/paper_2026_rl4llm
```

The root contains `filings/`, `text/` and the original split/data. Strict page loading rejects empty filings and whole-document text without form-feed page boundaries; old page caches are not reused. Archive corpus/cache fingerprints, code commit and judge prompt/cache hashes. Preflight validates every required document and literal target quote.

## 3. Measure the corrected base and run controlled training

Set `TINKER_API_KEY` securely in the environment or use the existing local key file. Set `JUDGE_BACKEND=deepinfra` and supply the judge credential through the existing environment/keyring mechanism if judging text paraphrases. Freeze the judge model/config/cache and audit approximately 50 training trajectories. With no judge, non-exact text answers remain unresolved. Training excludes the entire unresolved group with masked tokens and zeroed group totals; evaluation preserves its denominator and reports uncertainty bounds. Provider failures are unresolved, not negative labels.

```bash
python harness/workshop_eval.py --condition B1 --run-id B1-dev --split dev --out results/paper_2026_rl4llm/B1-dev
```

Inspect finish rate, failure reasons and visibility first. Use eight turns and 1,024 generated tokens initially. Change budgets only from development evidence; then use the frozen values for all corrected arms. `workshop_eval.py` uses the same renderer, tools, environment, targets and judge as training. `eval_agent.py` now delegates to this same evaluator; old --backend/--limit commands intentionally fail.

```bash
EPOCHS=3 STEPS=10 BATCH=8 GROUP=8 LR=1e-5 SEED=0 MAX_TURNS=8 MAX_TOKENS=1024 EVAL_EVERY=4 SAVE_EVERY=4 DEV_SPLIT=dev LOG_PATH=results/tinker_runs/workshop_pilot_lr1e5 python harness/train_financebench.py
EPOCHS=3 STEPS=10 BATCH=8 GROUP=8 LR=5e-5 SEED=0 MAX_TURNS=8 MAX_TOKENS=1024 EVAL_EVERY=4 SAVE_EVERY=4 DEV_SPLIT=dev LOG_PATH=results/tinker_runs/workshop_pilot_lr5e5 python harness/train_financebench.py
```

`STEPS` and cookbook evaluation cadence count nominal batches, not actual optimizer updates. `optimizer_audit.jsonl` records completed optimizer calls and skipped empty batches; use it to label curves. The pilot ceiling is ten nominal batches here (at most ten updates): if skips prevent ten updates, report the observed count rather than silently extending sampling. `EPOCHS=3` preserves remainder batches and caps exposure. Update protocol.md with this bounded scheduling rule before starting. `SEED` fixes dataset order; it is not a claim that the provider's training initialization or generation is bitwise seeded.

Select LR using dev grounded success, then correctness, then cost. Include base as a checkpoint candidate. Restart final runs from base; never continue from a pilot for one arm and restart another. Set STEPS to the intended nominal cap derived from actual split size, at most three epochs. Prefer a second seed to another reward variant. Optional retrieval shaping uses `RETRIEVAL_WEIGHT=0.1`; baseline is zero. Keep this reward weight out of the final correctness metric.

```bash
EPOCHS=3 STEPS=36 BATCH=8 GROUP=8 LR=5e-5 SEED=0 MAX_TURNS=8 MAX_TOKENS=1024 EVAL_EVERY=4 SAVE_EVERY=4 DEV_SPLIT=dev LOG_PATH=results/tinker_runs/workshop_final_seed0 python harness/train_financebench.py
```

The LR and 36-batch command are illustrative: replace LR with the dev-selected value and STEPS with the actual split-derived cap. Do not infer success from a completed run. `group_audit.jsonl` records sampled/retained/all-zero/all-equal groups and reward variance. Filter its split field to train when computing training coverage; dev groups are also logged. The synchronous training hook replaces cookbook's retain-one-constant fallback and skips the entire optimizer path for zero-signal batches. `workshop_rollouts/` saves full histories, visible observations, receipts and calculations. Preserve upstream metrics for loss/KL and sampled-token masks; do not call the old exported `final_reward` field the total reward.

## 4. Freeze checkpoints and generate matched paper results

Run final evaluation once after checkpoint selection. Optional sampling seeds require installed SDK support; the evaluator refuses unsupported `SamplingParams.seed`. Unseeded provider sampling is explicitly recorded as null. If using repeated evaluations, use the same declared seed set for all arms; never report best-of-k as single-rollout accuracy.

```bash
python harness/workshop_eval.py --condition B1 --run-id B1 --out results/paper_2026_rl4llm/B1
python harness/workshop_eval.py --condition R1 --run-id R1-seed0 --training-seed 0 --checkpoint YOUR_SELECTED_TINKER_CHECKPOINT --out results/paper_2026_rl4llm/R1-seed0
python harness/workshop_report.py --protocol artifacts/workshop/protocol.json --predictions results/paper_2026_rl4llm/B1/predictions.jsonl results/paper_2026_rl4llm/R1-seed0/predictions.jsonl --out results/paper_2026_rl4llm --figures
```

The reporter rejects missing or duplicate scheduled questions and incompatible evaluator fingerprints. It produces CSV/Markdown main/failure/paired tables, question-level paired bootstrap intervals (10,000 resamples, fixed seed), document-cluster sensitivity, failure figures and a preliminary paper_results.md. Unknown cost stays unmeasured. It does not silently omit failed or unresolved episodes. Distinguish visible citations from semantically supported citations; finish the support/repetition audit from saved traces, and report all failures.

For the learning figure, evaluate saved checkpoints on dev with the same evaluator, join them to optimizer_audit.jsonl, and write a measured CSV with `run_id,training_seed,actual_optimizer_updates,dev_correct_pct,dev_grounded_pct,checkpoint`. Include the base at update zero.

```bash
python harness/workshop_learning_curve.py --csv results/paper_2026_rl4llm/dev_curve.csv --out results/paper_2026_rl4llm
```

B0 remains an explicitly separate original-harness control: rerun the original commit or use settings-matched existing logs, then freeze an adapter for its free-form submissions into the same reviewed answer rubric. Do not invent receipts or terminal finishes. Where original trace visibility/termination cannot support a common metric, mark grounding unmeasurable and report B0 descriptively; exclude it from strict paired tables. The automated reporter intentionally enforces identical corrected-harness fingerprints for B1/R1, and will reject a mixed B0 export. The paper must distinguish this diagnostic B0 analysis from the controlled B1→R1 learning comparison.

## 5. Finish the exact handoff

Complete every artifact in the brief's reporting appendix. Specifically add citation/provenance adversarial cases to table3 using the regression suite, the trace audit with actual reviewer provenance, historical v11/v2 rescoring where possible, observed compute/token costs, explanations of source/operand coverage, case studies, full commands and the four-page outline. Update completion_report.md with complete/partial/blocked status. No claims of a positive GRPO effect until measured; keep negative runs. No synthetic-data expansion before the deadline unless the protocol is explicitly revised.

The scope deliberately uses strict reviewed derivations and conservative text grading. A changed rubric must be versioned and applied consistently to every arm. Acceptance gates are real SDK tests, corpus/label preflight, a successful matched base smoke run, an actual optimizer-update smoke run, and faithful evidence exports. Stop the paid run on a failed gate, preserve diagnostics, and report the concrete blocker.
