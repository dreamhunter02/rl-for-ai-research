# FinanceBench harness audit — 2026-09-30

## Scope and method

Reviewed all 24 saved dev trajectories: 12 historical Qwen3.5-4B baseline and 12 QLoRA SFT trajectories. Read their assistant calls, delivered tool observations, submissions and scores, plus the matching frozen targets. Inspected retrieval, observation budgeting, calculator, scorer, semantic judge, teacher/evaluation runner, SFT selection and assistant-loss masking code.

Sources: `results/paper_2026_rl4llm/qwen35_4b_current_harness_baseline_dev12_v2.jsonl`, `qwen35_4b_qlora_dev12_20260930.jsonl`, and `targets_frozen_rubric-v3.json`. Local HEAD: `d5ef4176a5c3dbe7acb9564f21060e605aa473b2`; checkout contains existing modified/untracked harness files, so HEAD alone does not identify the audited code. The historical baseline lacks a complete settings manifest.

Read-only offline checks reproduced unit mismatch, statement classification, company filtering and strict expression matching. Checked the actual SparkTwo training file `/home/dreamhunter/qwen4b_sft_train.jsonl`: 284 examples, 73 containing the calculator metric/period error. This was a programmatic training-file check, not a manual review of every teacher trajectory. No model/API evaluation, training, source edits or saved-score changes were performed. This report is the only new artifact.

## Main conclusion

The harness has consequential false-negative scoring, insufficiently checked positive scores, and an unnecessarily difficult tool contract. These coexist with genuine model errors. Current scores should not be presented as a clean measure of model ability, and making grading uniformly more permissive would not solve the problem.

SFT calculator calls: **16 attempts, 16 errors across five questions**. Thirteen errors were `Operand metric and period must identify text in its source quote`, two were quote-provenance failures, and one was malformed operand JSON/dictionary validation. Baseline: nine calculator calls, six errors and three constant-only results marked `provenance_valid=false`. Neither run produced a successful source-backed calculation.

## Per-question review

Rewards below are original recorded rewards, not corrected scores. IDs omit the common `financebench_id_` prefix.

| ID | Base → SFT reward | Evidence and diagnosis |
|---|---:|---|
| 03029 | 1 → 1 | Correct capital expenditure extraction in both. SFT finishes in three assistant turns versus seven. Useful positive control: simple search/read/finish works. |
| 00499 | 0.5 → 0.5 | Both say 3M is not capital-intensive and receive A=1, but supply unsupported total-assets figures. SFT cites a derivatives table and claims $42,666m assets; its delivered evidence does not establish that number. Also omits reference fixed-assets/assets and ROA metrics. Retrieval misroutes to notes; the judge's positive verdict needs review, not automatic promotion to full reward. |
| 00807 | 0 → 0 | Baseline confuses current and quick ratios. SFT computes reference 0.96 correctly but concludes “Yes” against the reference's “No.” Five rejected calculate calls consume turns. Both receive G=1 from page matching despite absent verified derivations. SFT merits a separate numeric-component diagnostic, not full-answer correctness. |
| 02987 | 0 → 0 | Baseline malformed finish and incorrect 20.15. SFT returns correct 24.26 with `unit=ratio`; target uses `number`, making A=0. Changing only the submitted unit in memory makes A=1. Its two calculator calls also fail; it passes a computed average as if it were a source operand. Correctness normalization does not repair derivation provenance. |
| 04735 | 0 → 0 | Both return 0.66. Baseline also uses erroneous `scale=thousand`; SFT correctly uses `ones` but `ratio` versus target `number` makes A=0. Unit-only diagnostic makes SFT A=1. Three SFT calculator failures. Gold operand quotes are 2,970 and 3,200 characters, exceeding the 2,400-character read cap; headers and rows are in separate delivered windows. Grounding is unreachable under this exact-quote contract. SFT invents `calc_1`. |
| 07507 | 0 → 0 | Baseline finds actual income statements but ends with invalid scale and inaccurate 65.0. SFT never reads the income statement; gives unsupported 10.4% and wrong answer type. Retrieval classification contributes, but the final answer is genuinely wrong. |
| 00438 | 1 → 1 | Correct operating-margin direction and figures in both. Baseline uses three constant-only calculations; SFT has four rejected source-backed calls. Both receive full reward because the target is text and derivation checking applies only to derived numeric targets. SFT also reverses the cost/revenue-growth explanation in its prose; full credit does not establish that every claim is correct. |
| 00540 | 0 → 0 | `company=AES Corporation` returns no hits in both because metadata contains AES. Retrieval then favors notes. Baseline eventually reads inventory but supplies wrong turnover; SFT claims inventory is not reported without reading the actual balance sheet. Company alias handling should improve; incorrect final assertions should remain failures. |
| 06655 | 0 → 0.5 | Baseline wrong 113.66. SFT gets reference 93.86 but has two calculator failures, no final citations, and invented `calc_1`. A=1/G=0 is defensible. Verbose copied quotes, computed intermediate operands and weak recovery feedback contribute; the saved trace does not prove token truncation because provider finish reasons are not logged. |
| 03882 | 0 → 1 | Both extract source 1,615.9. Baseline first includes prohibited numeric prose, then submits `unit="USD millions"` alongside `scale=million`; unit normalizer accepts that unknown string but equality rejects it. SFT uses USD/million and passes the target's whole-million precision. Both continue searching after already obtaining the answer. |
| 01936 | 0 unresolved → 0 unresolved | Both confuse fiscal Q2 with calendar Q2. SFT reads the correct Dec 31, 2022 note with $81m employee costs out of $93m total, then incorrectly claims the requested filing is unavailable. Metadata lacks explicit fiscal-period/calendar-date mapping. Model reasoning also fails. Keep unresolved distinct from confirmed wrong; don't silently award a pass. |
| 00684 | 1 → 0.5 | Both identify declining gross margin and read page 33's explicit 18.5%/19.4% table. Baseline also cites gold page 50, SFT cites page 33 only. Exact gold-page/quote matching causes SFT G=0 despite supporting evidence for its core answer. Its extra commentary needs separate checking; this finding does not validate every added claim. |

## Priority 1 — Correctness and grounding must measure the intended task

`harness/workshop_reward.py:62` requires exact normalized-unit equality. Targets 02987 and 04735 specify `number` for ratios, while the public tool explicitly offers `ratio`. Offline unit-only substitution changes both SFT A scores from 0 to 1. Preserve currency, percent, scale and sign distinctions; canonicalize dimensionless ratios using reviewed quantity types, not a universal alias that also erases units such as days.

`workshop_reward.py:255` requires the same gold page and exact quote in a single receipt. This rejects alternate legitimate evidence (00684), evidence assembled across read windows (04735), and harmless extraction whitespace variation. Replace monolithic gold spans with per-claim evidence requirements; support alternatives and verified row/header associations. Keep citation authenticity checks.

`workshop_reward.py:270` also matches hidden metric labels and period strings, and line 287 matches expression ASTs exactly. An offline synthetic check with identical operands and provenance scored canonical `ocf15/cl15` G=1 but mathematically equivalent `(ocf15/cl15)+0` G=0. The synthetic receipts deliberately bypassed the read cap to isolate expression matching; they are not real successful trajectories. Accept verified equivalent derivations through bounded canonicalization while preserving operand semantics, not merely coincident final numbers.

Preflight should attempt valid oracle tool trajectories through the actual public tool limits before training. A target whose required evidence cannot be delivered is a protocol failure, not a model failure.

## Priority 2 — Make source-backed calculation usable

`workshop_reward.py:181` requires each operand quote to be an exact substring of one prior receipt; line 183 requires metric and period within that same quote. Financial rows usually omit the date because it is a column header. The model repeatedly changes `FY2022` to `2022` or a date but keeps the same row quote, and receives the same generic error.

Use source-backed cell/row references with separately linked header evidence, retaining column order, period, unit and sign. Put intermediate arithmetic into the expression over raw operands, or explicitly support chained verified calculations. Do not pretend an average like 267.5 appears in a source row containing 253 and 282.

Return structured errors naming the failing operand and missing field, including a source-independent example of the required shape. Expose a real nested operand schema rather than only an unstructured dictionary description. Validate supplied receipt/calculation references before terminal acceptance without revealing target answers. Currently `finish` can accept nonexistent `calc_1` as schema-valid; grounding later fails, leaving no repair opportunity.

Simply increasing eight turns to twelve will not repair these issues; it may permit more identical retries. After corrections, select a common turn/token budget on train/dev and use it consistently across comparisons.

## Priority 3 — Fix retrieval and reading contracts before adding expensive search

`financebench_harness.py:177`: the statement classifier recognizes singular “STATEMENT OF INCOME” but not plural “STATEMENTS OF INCOME.” A direct check returns `other` for the common plural title. Conversely, derivatives-note text containing “Balance Sheet” is classified `balance` and can receive the +8 bonus. The traces show 00499 routed to derivatives page 92, and 07507 to notes/contents rather than the income statement.

`financebench_harness.py:235`: company filtering requires normalized exact equality, so AES Corporation does not match AES. Add deterministic company aliases and expose fiscal year, fiscal quarter, calendar period end, and PDF-page versus printed-page labels from filing metadata. These are corpus metadata, not benchmark answers.

`financebench_harness.py:282`: multiple queries are merged using maximum raw BM25 score, and prose/table results are merged by raw score across separate indexes. A broad company-name query can crowd out metric queries. Deduplicate page-level results and evaluate per-query coverage/rank fusion on train/dev; don't assume a semantic-search service is needed yet.

`financebench_harness.py:349`: `read_table` is a page-text read with metadata, not a structured table reader. `include_neighbors` returns page numbers, not the neighboring header/footnote text promised by the system description. Preserve table headers on continuation or describe the actual contract accurately.

`finance_env.py:139`: grep snippets are cut to 500 characters before observation truncation is measured. Thus a snippet may end mid-sentence while `observation.truncated=false`; this happens in 01936. Add per-hit truncation, character offsets and continuation, preserving the matched line. The backend selector is also misleading: text/pdfgrep/rga requests all use page_text.

## Priority 4 — Separate answer components without rewarding fabricated claims

00807 contains correct arithmetic but the wrong reference conclusion. Keep strict whole-answer accuracy for benchmark reporting; add separately logged numeric, conclusion and grounding components for learning diagnostics. Any partial-reward scheme needs explicit weights and adversarial tests, not a post-hoc decision to make this answer pass.

00438 shows the inverse loophole: text answers containing arithmetic bypass derived-numeric calculation validation and receive full reward despite failed calculate calls. Apply verification to claims regardless of answer serialization type.

00499 exposes judge over-credit: positive `numeric_ok` despite unsupported asset numbers and omitted reference metrics. `reward_calculation.py:265` asks for semantic equivalence but says evidence does not change factual correctness. Revise the evaluator to distinguish reference agreement, missing required facts, unsupported extra claims, and citation support. Store the judge's reason/reason_code, not only a verdict and confidence. Judge confidence is not independent validation.

01936's unresolved verdict must remain explicit. In GRPO, one unresolved member currently excludes the whole group (`finance_env.py`, `compute_group_rewards`); that is conservative but wastes an entire group. Improve deterministic checks and judge reliability before paid rollouts, and record exclusion causes.

## Priority 5 — Curate the behavior, not only the terminal score

`prepare_sft_dataset.py:11` selects F=A=G=1 terminal traces and excludes parse errors, but not intermediate tool-validation errors. `build_messages` copies those calls, and `train_qwen_sft.py:94` supervises all assistant tokens, including rejected actions. On SparkTwo, **73/284 examples (25.7%) contain the metric/period error**. This exposes the learner to bad calls; it does not by itself prove how much of the SFT behavior was caused by them.

Prefer clean, shortest valid demonstrations; repair and replay failed steps before inclusion. Preserve originals. If retaining recovery demonstrations deliberately, mark and mask rejected actions or maintain a separate recovery subset. Never append an invented successful finish or silently erase steps that later receipts depend on.

`curate_sft_traces.py:28` takes the first eligible trace per question/teacher, not the best one. Rank verified candidates by evidence completeness, absence of invalid calls, and unnecessary repetition. Terminal reward alone is not a sufficient training-quality gate.

## Additional code-only risks (not causes established by these 24 traces)

The Responses API branch of `generate_teacher_traces.py:212` reuses `output` for the model output list and a tool-output string, then iterates it as model objects. It needs a focused regression test before using that backend. Current dev runs use Chat Completions, so this is not their failure cause.

`eval_current_harness.py` resumes by question ID without validating a configuration fingerprint. Persist model/adapter hashes, actual system prompt, corpus/target/scorer hashes, tool schemas, provider completion reasons and token usage; reject incompatible resume settings. The evaluator uses the teacher runner, including its teacher-specific prompt suffix and forced final-turn finish. Match these settings deliberately to the intended RL/deployment environment.

## Recommended next gate

Create offline regression cases from these failures before changing reward weights or running more training: dimensionless units; valid alternate-page support; multi-window header/row evidence; source-backed calculations and equivalent formulas; unsupported numbers; fiscal periods; and clean SFT eligibility. Retain wrong-year, wrong-sign, wrong-column and fabricated-source rejection tests. Version the corrected rubric, rescore both saved dev runs under it, and report old and new scores separately. Do not treat a scoring correction as a model improvement or tune against eval42.
