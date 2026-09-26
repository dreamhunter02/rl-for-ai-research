#!/usr/bin/env python3
"""Tiny Unsloth Qwen3.5-4B GRPO smoke test.

This deliberately uses a toy exact-answer verifier. It validates the
Qwen3.5 bf16-LoRA + Unsloth/TRL GRPO stack before connecting FinanceBench.
"""
import argparse
import json
import re
from pathlib import Path

# Unsloth must be imported before transformers/trl.
import unsloth  # noqa: F401
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


EXAMPLES = [
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 17 + 25?", "answer": "42"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 9 * 6?", "answer": "54"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 144 / 12?", "answer": "12"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 81 - 37?", "answer": "44"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 13 * 7?", "answer": "91"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 256 / 16?", "answer": "16"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 68 + 19?", "answer": "87"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 100 - 63?", "answer": "37"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 15 * 8?", "answer": "120"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 225 / 15?", "answer": "15"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 39 + 46?", "answer": "85"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 72 - 28?", "answer": "44"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 11 * 11?", "answer": "121"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 196 / 14?", "answer": "14"},
    {"prompt": "Solve this arithmetic problem. Give the final integer after Answer:. What is 27 + 58?", "answer": "85"},
]


def _completion_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        parts = []
        for message in item:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return " ".join(parts)
    if isinstance(item, dict):
        return str(item.get("content", item.get("text", "")))
    return str(item)


def exact_answer_reward(completions, answer, **kwargs):
    """Mixed verifier: exact > formatted-but-wrong > numeric-only > empty."""
    rewards = []
    for completion, gold in zip(completions, answer):
        text = _completion_text(completion)
        matches = re.findall(r"(?i)\banswer\s*:\s*(-?\d+)\b", text)
        predicted = matches[-1] if matches else ""
        has_number = bool(re.search(r"-?\d+", text))
        has_answer_marker = bool(re.search(r"(?i)\banswer\s*:", text))
        if predicted == str(gold):
            reward = 1.0
        elif has_answer_marker and has_number:
            reward = 0.25
        elif has_number:
            reward = 0.10
        else:
            reward = 0.0
        # A tiny bounded tie-breaker prevents identical coarse scores from
        # becoming completely flat when sampled completions differ in length.
        reward = min(1.0, reward + 0.01 * min(len(text), 100) / 100)
        rewards.append(reward)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/hf/models--Qwen--Qwen3.5-4B")
    parser.add_argument("--output-dir", default="/workspace/results/unsloth_grpo_smoke/qwen35_4b_r32_16k_mixed30")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=32)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(EXAMPLES)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        fast_inference=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
    )

    config = GRPOConfig(
        output_dir=str(out),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        max_prompt_length=256,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.max_steps,
        save_only_model=True,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        reward_funcs=[exact_answer_reward],
        train_dataset=dataset,
    )
    result = trainer.train()
    metrics = dict(result.metrics)
    metrics.update({
        "model": args.model,
        "output_dir": str(out),
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "max_seq_length": args.max_seq_length,
        "max_completion_length": args.max_completion_length,
        "lora_rank": args.lora_rank,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    })
    (out / "smoke_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
