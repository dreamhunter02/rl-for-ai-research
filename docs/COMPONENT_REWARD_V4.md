# Calculator-free harness and component rewards

## Tool review

| Tool | Job | Necessity and repair |
|---|---|---|
| bm25_search | Find filings and relevant prose/table candidates across the corpus | Primary discovery; invalid scope now gives an actionable error. Company aliases no longer require AES versus AES Corporation exact spelling. |
| grep_document | Locate terms/regex matches inside a known filing | Useful complement to ranked search. Returns indexed text, not three distinct backends. Clipped snippets include offsets and continuation instructions. |
| search_tables | Rank table-like pages with statement hints | Specialized shortcut overlapping table-scope search. Retained pending an actual removal ablation. Fixed plural income-statement titles and misleading notes-page classification. |
| read | Retrieve source text by page/passage and character window | Primary evidence access; receipts contain delivered text only. |
| read_table | Resolve a table ID to its page window, title and units | Convenience wrapper, NOT a structured table parser. Neighbor pages are pointers, not delivered evidence; prompt now states this explicitly. |
| finish | Submit one structured final answer and citations | Required terminal action. Optional derivation replaces calculator record references. |

`calculate` is absent from active registrations and current SFT schemas. Legacy
calculation records remain readable in historical scoring code and saved traces.
No historical dev tool call was erased or retroactively made successful.

The audit does not establish that every convenience tool improves accuracy.
It also does not establish how Qwen behaves under the repaired interface: that
requires a new, separately authorized model evaluation. Current work regrades
existing trajectories only.

## Reward contract

R = F × A × (0.5 + 0.5G). For mixed questions A = 0.6N + 0.4S;
numeric-only A=N; semantic-only A=S.

F is accepted finish; N is requested final-number correctness; S is semantic
correctness; G is the fraction of assessed final claims supported by authentic
final citations. As of v5, applicability and requested claim IDs are determined
from the question alone, before reference or candidate exposure. A second pass
uses the reference to answer only those requirements; mode and claim IDs cannot
change. Optional gold metrics are not requirements. Semantic equivalence permits
different valid supporting explanations, but not materially contradictory ones.
Volunteered numeric claims are still checked for cited support and arithmetic
under grounding even when N is not applicable. Grounding uses atomic, nonredundant
candidate claims rather than a checklist of reference facts.
The judge interprets equivalent units and recomputes arithmetic; receipt matching
is deterministic, but exact gold-page/quote/expression matches are not required.

Provider errors and malformed or ambiguous judgments remain unresolved with null
reward. The RL SDK gets a temporary float zero solely for transport; the existing
group-exclusion mechanism excludes the entire unresolved group. Missing judge
configuration fails before paid rollout generation. Legacy scorer mode exists
for historical diagnostics, not as the corrected calculator-free reward.

Tool counts, error observations, duplicate identical calls, and delivered character
counts are diagnostics. There is no diversity bonus, no unvalidated efficiency
penalty, and no automatic attribution of infrastructure failures to the model.

## Configuration and use

Copy the variable names from `configs/component_reward.env.example` into the
deployment's private environment. Set endpoint to a chat-completions URL, model
to an available model ID, and either API_KEY or KEY_FILE. Never commit credentials.
An exported API_KEY overrides KEY_FILE. Judge caches contain prompts' hashes and
judgments, not authentication headers.

Offline regrading:

```sh
PYTHONPATH=harness python harness/rescore_traces.py \
  --before results/paper_2026_rl4llm/qwen35_4b_current_harness_baseline_dev12_v2.jsonl \
  --after results/paper_2026_rl4llm/qwen35_4b_qlora_dev12_20260930.jsonl \
  --out results/trace_comparisons/new_reward_review
```

Use `--rubric-from PREVIOUS_OUTPUT` with another configured model for an independent
judgment using exactly the same rubrics. Use `--merge-secondary SECONDARY_OUTPUT`
with the primary output directory to build an offline comparison without replacing
primary scores. Every result is linked to the complete original record hash.

Migration writes new outputs and refuses an existing output directory:

```sh
PYTHONPATH=harness python harness/migrate_teacher_traces.py \
  --root results/teacher_traces --split artifacts/workshop/split.json \
  --targets results/paper_2026_rl4llm/targets_frozen_rubric-v3.json \
  --out results/teacher_traces/new_migration --judge --concurrency 4
```

All rows are reconciled into candidates, quarantine, duplicate lineage, or excluded
non-training records. Candidate migration removes failed calculator exchanges,
not successful calculator-dependent reasoning. Finish arguments must agree with
the tool's accepted echo. Candidate does NOT mean training-ready. Only source-replayed,
fully judged `clean.jsonl` records qualify. A balanced shortlist is judged first;
unselected alternatives remain in `candidates.jsonl`, without invented rewards.
`--fallback-model` explicitly permits a logged alternate judge on provider failures,
not on wrong answers or semantic ambiguity. Frozen unresolved targets are quarantined.

## Validation boundaries

The dev report is post-hoc development-set reward repair, not benchmark improvement.
It preserves original scores and all trace events, plus independent judge judgments.
Agreement on correctness does not prove correctness; fractional grounding differs
between judges and should be calibrated before GRPO. No eval42 data was used to
select these repairs. No training, serving restart, or adapter deletion was performed.

The full test suite runs with `PYTHONPATH=harness python -m pytest tests -q` in the
configured cookbook environment. Live judge controls require the explicit
`RUN_LIVE_JUDGE_TESTS=1` opt-in and configured endpoint; default tests do not call it.
