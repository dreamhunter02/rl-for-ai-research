# FinanceBench search agent

Train and evaluate a financial-filings search agent with Qwen3.5-4B SFT and
GRPO experiments. The active harness exposes search, bounded evidence reads,
and structured `finish`; no calculator tool.

## Repository map

| Path | Purpose |
|---|---|
| `harness/` | Retrieval environment, judge, teacher preparation, training and evaluation entry points |
| `data/teacher_traces/` | One selected calculator-free SFT dataset, provenance and limitations |
| `data/targets.json`, `split.json` | Frozen targets and **96 train / 12 dev / 42 eval** questions |
| `configs/`, `requirements-workshop.txt` | Runtime configuration and pinned Tinker cookbook dependency |
| `tests/`, `docs/` | Regression tests, recipes and reward contracts |

## Start here

1. Follow [the recipes](docs/RECIPES.md) for corpus setup, SFT and matched dev12 evaluation.
2. Read [teacher data notes](data/teacher_traces/README.md) before selecting training data.
3. Read [the reward contract](docs/COMPONENT_REWARD_V4.md): live generation uses
   v5 diagnostic scoring; the latest offline report uses v6 additive scoring.
   These are **not yet one unified GRPO reward path**.

The selected teacher file contains **247 traces across 73 training questions**.
It was selected under v4, not revalidated under v6. It is not a new native-rollout
dataset, and it is not evidence that the existing SFT adapter was trained on
this exact file. No benchmark improvement is claimed here.

The frozen eval42 is held out from prompt, reward and checkpoint selection.
Targets are preserved inputs, not a claim that every annotation is correct.

## Outputs stay out of Git

`results/`, `artifacts/`, corpus files, caches and adapters are local/ignored.
Historical reports, run logs and plans were moved to a verified external archive;
see [archive and recovery](docs/ARCHIVE.md). Git history is unchanged.
The old `make_split.py` was archived to prevent accidentally regenerating the split.

Some older harness entry points remain for compatibility and regression coverage.
Use the entry points documented in the recipes; historical commands may require
inputs restored from the archive. This cleanup does not refactor reward logic or
remove legacy code that current tests/imports still exercise.
