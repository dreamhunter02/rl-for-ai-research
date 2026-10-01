"""LoRA SFT for Qwen 3.5 4B on verified current-harness trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def qlora_model_load_kwargs(max_length: int) -> dict[str, Any]:
    return {
        "max_seq_length": max_length,
        "load_in_4bit": True,
        "load_in_16bit": False,
        "full_finetuning": False,
        "fast_inference": False,
    }


def qlora_adapter_kwargs(rank: int, alpha: int, seed: int, max_length: int) -> dict[str, Any]:
    return {
        "r": rank,
        "target_modules": LORA_TARGET_MODULES,
        "lora_alpha": alpha,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": seed,
        "max_seq_length": max_length,
    }


def sft_optimizer(qlora_4bit: bool) -> str:
    return "adamw_8bit" if qlora_4bit else "adamw_torch_fused"


def inner_tokenizer(processor: Any) -> Any:
    """Return the text tokenizer when Unsloth loads a multimodal processor."""
    return getattr(processor, "tokenizer", processor)


def validate_sft_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("SFT dataset is empty")
    trajectory_keys = [
        (str(row.get("financebench_id", "")), str(row.get("teacher_model", "")))
        for row in rows
    ]
    if len(trajectory_keys) != len(set(trajectory_keys)):
        raise ValueError("SFT dataset contains a duplicate question/teacher trajectory")
    for row in rows:
        finish_calls = [
            call.get("function", {})
            for message in row.get("messages", [])
            if message.get("role") == "assistant"
            for call in message.get("tool_calls", [])
            if call.get("function", {}).get("name") == "finish"
        ]
        if len(finish_calls) != 1 or not finish_calls[0].get("arguments", {}).get("answer_type"):
            raise ValueError(f"{row.get('financebench_id')}: exactly one typed finish is required")
        if not row.get("tools"):
            raise ValueError(f"{row.get('financebench_id')}: tool schemas are required")


def _token_fields(encoded: Any) -> tuple[list[int], list[int]]:
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
        return input_ids, attention_mask
    input_ids = list(encoded)
    return input_ids, [1] * len(input_ids)


def chat_template_with_generation_tags(template: str) -> str:
    if "{% generation %}" in template:
        return template
    assistant_start = '{%- elif message.role == "assistant" %}'
    assistant_end = "{{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"
    if assistant_start not in template or assistant_end not in template:
        raise ValueError("Unsupported chat template: assistant branch was not found")
    template = template.replace(assistant_start, assistant_start + "\n        {% generation %}", 1)
    return template.replace(
        assistant_end,
        "{{- '<|im_end|>\\n' }}\n        {% endgeneration %}\n    {%- elif message.role == \"tool\" %}",
        1,
    )


def tokenize_with_assistant_labels(row: dict[str, Any], tokenizer: Any) -> dict[str, list[int]]:
    """Tokenize a conversation and mask every non-assistant message from loss."""
    messages = row["messages"]
    template_args = {
        "tools": row["tools"],
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
        "return_assistant_tokens_mask": True,
    }
    encoded = tokenizer.apply_chat_template(messages, **template_args)
    input_ids, attention_mask = _token_fields(encoded)
    assistant_mask = list(encoded.get("assistant_masks", []))
    if len(assistant_mask) != len(input_ids):
        raise ValueError("Chat template did not return a valid assistant mask")
    labels = [token if supervised else -100 for token, supervised in zip(input_ids, assistant_mask)]
    if not any(label != -100 for label in labels):
        raise ValueError("Trajectory has no stable assistant tokens")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def select_length_safe_rows(
    rows: list[dict[str, Any]],
    tokenized_rows: list[dict[str, list[int]]],
    max_length: int,
    drop_oversize: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, list[int]]], list[dict[str, Any]]]:
    dropped = [
        {"financebench_id": row.get("financebench_id", ""), "tokens": len(tokenized["input_ids"])}
        for row, tokenized in zip(rows, tokenized_rows)
        if len(tokenized["input_ids"]) > max_length
    ]
    if dropped and not drop_oversize:
        raise ValueError(
            f"longest trajectory has {max(item['tokens'] for item in dropped)} tokens, "
            f"above --max-length={max_length}; increase the limit or use "
            "--drop-oversize to exclude whole trajectories without truncation"
        )
    kept = [
        (row, tokenized)
        for row, tokenized in zip(rows, tokenized_rows)
        if len(tokenized["input_ids"]) <= max_length
    ]
    return [item[0] for item in kept], [item[1] for item in kept], dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="/hf/models--Qwen--Qwen3.5-4B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--drop-oversize", action="store_true")
    parser.add_argument("--qlora-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    FastLanguageModel = None
    if args.qlora_4bit:
        import unsloth  # noqa: F401 - must patch libraries before transformers is imported
        from unsloth import FastLanguageModel

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    validate_sft_rows(rows)
    if args.qlora_4bit:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            **qlora_model_load_kwargs(args.max_length),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = inner_tokenizer(tokenizer)
    tokenizer.chat_template = chat_template_with_generation_tags(tokenizer.chat_template)
    tokenized_rows = [tokenize_with_assistant_labels(row, tokenizer) for row in rows]
    rows, tokenized_rows, dropped_oversize = select_length_safe_rows(
        rows, tokenized_rows, args.max_length, args.drop_oversize,
    )
    if not rows:
        raise ValueError("Every SFT trajectory exceeded --max-length")
    lengths = [len(row["input_ids"]) for row in tokenized_rows]

    if args.qlora_4bit:
        model = FastLanguageModel.get_peft_model(
            model,
            **qlora_adapter_kwargs(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                seed=args.seed,
                max_length=args.max_length,
            ),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=LORA_TARGET_MODULES,
        ))
    model.print_trainable_parameters()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dropped_oversize.json").write_text(json.dumps(dropped_oversize, indent=2) + "\n")
    config = TrainingArguments(
        output_dir=str(output),
        do_train=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=sft_optimizer(args.qlora_4bit),
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(tokenized_rows),
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )
    result = trainer.train()
    trainer.save_model(str(output / "final_adapter"))
    tokenizer.save_pretrained(str(output / "final_adapter"))
    metrics = {
        **result.metrics,
        "examples": len(rows),
        "unique_questions": len({row["financebench_id"] for row in rows}),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "model": args.model,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "qlora_4bit": args.qlora_4bit,
        "optimizer": sft_optimizer(args.qlora_4bit),
        "dropped_oversize": len(dropped_oversize),
    }
    (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    print(json.dumps(metrics, default=str), flush=True)


if __name__ == "__main__":
    main()
