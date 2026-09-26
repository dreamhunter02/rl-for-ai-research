# SFT Plan: Qwen3.5-4B Student for FinanceBench Agentic Retrieval

## Goal
Train Qwen3.5-4B as a resource-constrained student model using SFT on Opus-judged teacher traces, so it learns the FinanceBench retrieval/search procedure and the explicit `finish` submission convention.

## Data
- **Primary SFT pool**: `results/teacher_traces/sft_judged_strict.jsonl` — 91 traces, 91 unique questions, mean deterministic reward 0.38, 27 strong (>=0.9), avg 5.5 tool calls.
- **Post-processing**: For each trace without a `finish` tool call, deterministically append a synthetic `finish(answer=..., evidence_document=..., evidence_page=...)` turn extracted from the final answer and evidence tool calls. Flag as `synthetic_finish=True` in metadata.
- **Result**: All 91 traces end with a `finish` call, teaching the student both the search procedure and the termination convention.

## Model
- **Student**: Qwen3.5-4B (bf16, ~8.9 GB weights, cached on Sparky Two at `/hf/models--Qwen--Qwen3.5-4B`).
- **Recipe**: Unsloth bf16 LoRA (not 4-bit QLoRA; Qwen3.5 delta-net/Mamba-hybrid layers quantize poorly).
  - `r=16`, `lora_alpha=16`, `lora_dropout=0`, `bias="none"`
  - `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
  - `use_gradient_checkpointing="unsloth"`
  - `max_seq_length=4096` (start; scale up after verifying loss curve)
  - Expected VRAM: ~10 GB (fits comfortably in DGX Spark 121 GB unified memory)

## Training
- **Hardware**: DGX Spark (Sparky One or Sparky Two; 121 GB unified memory, GB10 GPU).
- **Container**: `qwen-train` (Transformers 5.15.1, PyTorch 2.13.0+cu130, Unsloth available).
- **Loss**: Standard SFT next-token loss on the full trajectory (system + question + tool calls + tool outputs + assistant responses + finish).
- **Epochs**: Start with 1 epoch; monitor validation loss. If underfitting, scale to 2-3 epochs.
- **Batch size**: 1 (sequence length 4096); gradient accumulation 4-8 steps.
- **Learning rate**: 2e-4 (Unsloth default for bf16 LoRA on small models).

## Evaluation
- **Baseline**: Pre-SFT Qwen3.5-4B on the 30-question ablation set (mean 0.159, 28/29 finish, 0 empty, avg 10.8 tool calls).
- **Post-SFT**: Run the same 30 questions with the SFT'd 4B model, same harness settings (12 turns, 4096 max tokens, finish tool enabled).
- **Success criteria**:
  - Mean reward >= 0.25 (vs 0.159 baseline)
  - Finish termination rate >= 90%
  - Empty answers = 0
  - Avg tool calls <= 10 (efficiency)
- **Long-context eval**: Run the 2 questions that overflowed 16K (00685, 03838) with a 32K context to see if SFT improves their completion.

## Risks & Mitigations
1. **16K context ceiling**: The 4B model's default context is 16K; longest questions overflow. Mitigation: at eval time, either (a) increase the SGLang server `--max-total-tokens` to 32K, or (b) cap harness observation length (`OBS_CAP`) to fit within 16K.
2. **Weaker than 9B**: The 4B is a smaller model; frame as "resource-constrained student" in the paper. If SFT improves it to >= 0.25 mean, it's a strong result for a 4B model.
3. **Synthetic finish traces**: The appended `finish` calls are deterministic, not model-generated. Risk: the model learns to always call `finish` without proper evidence. Mitigation: the 91 traces are Opus-judged for evidence consistency, so the underlying reasoning is sound; the `finish` is just a formatting wrapper.
4. **Overfitting to 91 traces**: 91 unique questions is a small dataset. Mitigation: monitor validation loss on a held-out subset (5 questions); if overfitting, reduce epochs or increase dropout.

## Next Steps
1. Download Qwen3.5-4B weights to Sparky One (8.9 GB; ~2 min at current network speeds).
2. Apply synthetic `finish` post-processing to `sft_judged_strict.jsonl` → create `sft_judged_strict_finish.jsonl`.
3. Write the Unsloth SFT script (`harness/sft_qwen35_4b.py`) following the recipe above.
4. Run 1-epoch SFT on the 91 traces; monitor loss.
5. Evaluate the SFT'd 4B on the 30-question ablation set; compare to baseline.
6. If successful, scale to 2-3 epochs and re-evaluate.
7. Report results in `journal.md` and the paper.

## File Locations
- SFT traces (raw): `results/teacher_traces/sft_judged_strict.jsonl`
- SFT traces (with synthetic finish): `results/teacher_traces/sft_judged_strict_finish.jsonl` (to be created)
- Unsloth SFT script: `harness/sft_qwen35_4b.py` (to be created)
- 30-question ablation IDs: `results/local_eval/gpt56_finish_ablation_ids.txt`
- 4B baseline results: `results/teacher_traces/qwen35_4b_finish_ablation30.jsonl`
- Server config: SGLang, port 18360, `--max-total-tokens 16384` (upgrade to 32768 for long-context eval)


## Reward gate before SFT
The old deterministic grader is retained only as a frozen baseline. The active harness scorer is now `financebench_harness.score_answer`, which parses the submitted answer, handles typed conclusions and numeric units/rounding, checks directional contradictions, and reports quality separately from finish/format mechanics. On the 91 strict traces, the new mean is `0.8777`; all 46 traces previously scored zero but judged correctness=4 and grounding=2 now score at least `0.95`. The audit and regression tests must remain green before creating the final SFT JSONL.
