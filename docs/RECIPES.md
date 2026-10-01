# Supported execution recipes

Run commands from the repository root. These are entry points, not permission
to start paid training, judge calls or evaluation. Keep outputs outside Git.

## Environment and corpus

The pinned Tinker runtime uses `requirements-workshop.txt`. QLoRA additionally
needs a compatible Unsloth/Transformers/TRL CUDA environment; that environment
is separate from the lightweight harness dependencies. Use the existing Spark
training environment rather than installing CUDA packages into the Tinker venv.

Set `FINANCEBENCH_FILINGS`, `FINANCEBENCH_TEXT` and `FINANCEBENCH_CACHE` to the
existing PDF, extracted-text and page-cache directories when the corpus is
outside this checkout. Keep `FINANCEBENCH_ROOT` pointed at this checkout so its
versioned split and targets are used. Set `FINANCEBENCH_SPLIT` to the absolute
path of `split.json` and `FINANCEBENCH_TARGETS` to `data/targets.json` for Tinker.
Do not regenerate the split or feed evaluator targets to the model.

Configure judge variables from `configs/component_reward.env.example` through
a protected environment/key file. Never commit credentials. No paid call is
needed to run the offline suite:

```sh
PYTHONPATH=harness python -m pytest tests --import-mode=importlib -q
```

## SFT: one explicit, new output directory

The selected dataset has not been revalidated under the latest reward; inspect
its provenance before authorizing another run. The command below is a starting
recipe, not a claim it reproduces the existing adapter:

```sh
python harness/train_qwen_sft.py \
  --input data/teacher_traces/sft.jsonl \
  --model /hf/models--Qwen--Qwen3.5-4B \
  --output /path/to/new-sft-run \
  --qlora-4bit --max-length 24576 --epochs 2 \
  --learning-rate 1e-4 --batch-size 1 --gradient-accumulation 4 \
  --lora-rank 16 --lora-alpha 32 --seed 42
```

Keep `train_metrics.json`, checkpoints and `final_adapter` with the run. Capture
the code commit, full package freeze, data hash and launch command separately.
Do not silently truncate or drop oversize trajectories.

## Serving: existing Spark deployment

The ongoing matched run uses SGLang 0.5.20 in SparkTwo's existing `lfm-sglang`
container at `http://10.0.0.136:18361/v1`. Its model IDs are `Qwen3.5-4B` and
`Qwen3.5-4B:qwen-sft`. Adapter source:
`/home/dreamhunter/qwen4b_financebench_qlora_24k_20260930/final_adapter`.

Verify both IDs with `/v1/models` before evaluation. Do not restart this serving
process while evaluation is active. This records the existing deployment, not
a portable container-build recipe; retain its image digest and launch settings
in each evaluation protocol rather than guessing a new configuration.

## Matched base versus SFT dev12

Use an existing server, an unused output directory, the actual code commit and
the SHA256 of the served adapter weights:

```sh
python harness/run_matched_dev12.py \
  --root "$PWD" --run-dir /path/to/new-dev12-run \
  --base-url http://10.0.0.136:18361/v1 \
  --base-model Qwen3.5-4B --sft-model Qwen3.5-4B:qwen-sft \
  --code-commit COMMIT_SHA --adapter-sha256 ADAPTER_SHA256
```

The supervisor runs base, then SFT, then v6 judging. Both phases use dev12,
8 turns, 1024 tokens, temperature 0 and seed 0. It produces raw JSONL, logs,
`protocol.json`, `status.json` and `report/comparison.html`. It does not train
or evaluate eval42. Generation records v5 diagnostics; the HTML presents v6.

For already saved matched traces, supply both inputs explicitly to
`harness/additive_rescore.py` with `--root`, `--before`, `--after` and `--out`.
Old implicit historical input paths are not populated in a clean checkout.

## Teacher generation and GRPO

`harness/generate_teacher_traces.py` generates current-harness demonstrations;
use `--help` for endpoint/model selection and explicit output paths. Restrict
generation to the frozen training split. Preserve raw traces and migration
lineage separately from selected SFT messages.

`harness/train_financebench.py` is the Tinker entry point. Existing Nemotron
configs preserve older protocols; they are not a ready-to-launch v6 GRPO recipe.
Unify and validate the chosen reward path and learning-signal gate before
authorizing another paid run. This cleanup launches neither training nor eval.

## Qwen 3.5 4B Unsloth GRPO smoke

`harness/unsloth_financebench_grpo_v6.py` uses Unsloth's 4-bit Qwen loader and
the saved SFT LoRA as its starting policy. Its optimizer is a custom one-step
on-policy GRPO loop because each FinanceBench attempt contains several tool
calls; it is not TRL's single-completion `GRPOTrainer` recipe. The live judge
uses the same v6 formula as `additive_rescore.py`:
`R = .15E + F(.20G + .45A + .15B + .05)`. Unresolved judge results are excluded,
and an all-equal group makes no optimizer update.

The September 30 smoke on SparkTwo used train IDs `04672` and `01865`, two
generations each, eight turns, 512 new tokens per turn, a 16,384-token context,
and learning rate `1e-6`. For fast diagnosis it indexed only those questions'
two source filings. This restriction makes its retrieval scores incomparable
with full-corpus evaluation. The four rewards were 0.05, 0.40, 0.90 and 0.05;
two groups were retained and one optimizer step completed. This verifies a
nonflat learning signal and runnable update, not that the new adapter improves
held-out accuracy. The new adapter is separate from the served SFT adapter.

To reproduce in the existing SparkTwo Unsloth environment, set
`FINANCEBENCH_ROOT`, `FINANCEBENCH_FILINGS`, `FINANCEBENCH_TEXT`,
`FINANCEBENCH_CACHE`, and the protected `COMPONENT_JUDGE_*` variables, then run
the script with `--model` pointing to the SFT `final_adapter`, `--split` to the
frozen `split.json`, `--teacher-dataset` to the selected SFT JSONL (tool schema
only), and a fresh `--output-dir`. The exact arguments, split hash, four
rollouts, summary, and adapter are under
`results/grpo_runs/unsloth_qwen35_4b_v6_smoke_20260930/` locally (ignored by
Git). Use `--rollouts-from` with an identical protocol only to resume a failed
optimizer step without regenerating or rejudging those trajectories.
