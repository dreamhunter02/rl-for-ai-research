import unittest

from train_qwen_sft import (
    inner_tokenizer,
    chat_template_with_generation_tags,
    qlora_adapter_kwargs,
    qlora_model_load_kwargs,
    sft_optimizer,
    select_length_safe_rows,
    tokenize_with_assistant_labels,
    validate_sft_rows,
)


class TrainQwenSftTests(unittest.TestCase):
    def test_inner_tokenizer_unwraps_unsloth_processor(self):
        text_tokenizer = object()
        processor = type("Processor", (), {"tokenizer": text_tokenizer})()
        self.assertIs(inner_tokenizer(processor), text_tokenizer)
        self.assertIs(inner_tokenizer(text_tokenizer), text_tokenizer)

    def test_qlora_uses_4bit_unsloth_and_memory_efficient_optimizer(self):
        self.assertEqual(qlora_model_load_kwargs(24576), {
            "max_seq_length": 24576,
            "load_in_4bit": True,
            "load_in_16bit": False,
            "full_finetuning": False,
            "fast_inference": False,
        })
        adapter = qlora_adapter_kwargs(rank=16, alpha=32, seed=42, max_length=24576)
        self.assertEqual(adapter["r"], 16)
        self.assertEqual(adapter["lora_alpha"], 32)
        self.assertEqual(adapter["use_gradient_checkpointing"], "unsloth")
        self.assertEqual(adapter["max_seq_length"], 24576)
        self.assertEqual(sft_optimizer(qlora_4bit=True), "adamw_8bit")
        self.assertEqual(sft_optimizer(qlora_4bit=False), "adamw_torch_fused")

    def test_explicit_labels_train_only_assistant_messages(self):
        class Tokenizer:
            roles = {"system": 1, "user": 2, "assistant": 3, "tool": 4}

            def apply_chat_template(self, messages, **kwargs):
                ids = [token for message in messages for token in (
                    self.roles[message["role"]], 10 + self.roles[message["role"]]
                )]
                mask = [value for message in messages for value in (
                    0, int(message["role"] == "assistant")
                )]
                return {
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "assistant_masks": mask,
                }

        row = {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "call"},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "finish"},
            ],
            "tools": [{"type": "function", "function": {"name": "finish"}}],
        }

        tokenized = tokenize_with_assistant_labels(row, Tokenizer())

        self.assertEqual(tokenized["input_ids"], [1, 11, 2, 12, 3, 13, 4, 14, 3, 13])
        self.assertEqual(tokenized["labels"], [-100, -100, -100, -100, -100, 13, -100, -100, -100, 13])

    def test_qwen_template_assistant_branch_gets_generation_markers(self):
        template = (
            '{%- elif message.role == "assistant" %}\nassistant body\n'
            "{{- '<|im_end|>\\n' }}\n"
            '    {%- elif message.role == "tool" %}'
        )

        patched = chat_template_with_generation_tags(template)

        self.assertIn("{% generation %}", patched)
        self.assertIn("{% endgeneration %}", patched)

    def test_length_filter_drops_whole_oversize_trajectories_without_truncation(self):
        rows = [{"financebench_id": "short"}, {"financebench_id": "long"}]
        tokenized = [{"input_ids": [1, 2]}, {"input_ids": [1, 2, 3, 4]}]

        kept_rows, kept_tokenized, dropped = select_length_safe_rows(
            rows, tokenized, max_length=3, drop_oversize=True,
        )

        self.assertEqual([row["financebench_id"] for row in kept_rows], ["short"])
        self.assertEqual(kept_tokenized, [tokenized[0]])
        self.assertEqual(dropped, [{"financebench_id": "long", "tokens": 4}])

    def test_dataset_requires_unique_current_finish_trajectories(self):
        valid = [{
            "financebench_id": "q1",
            "teacher_model": "teacher-a",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "tool_calls": [{
                    "function": {
                        "name": "finish",
                        "arguments": {"answer_type": "text", "answer_text": "answer"},
                    },
                }]},
                {"role": "tool", "tool_call_id": "c1", "content": "accepted"},
            ],
            "tools": [{"type": "function", "function": {"name": "finish"}}],
        }]
        validate_sft_rows(valid)

        validate_sft_rows(valid + [{**valid[0], "teacher_model": "teacher-b"}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_sft_rows(valid + valid)
        with self.assertRaisesRegex(ValueError, "typed finish"):
            validate_sft_rows([{**valid[0], "financebench_id": "q2", "messages": valid[0]["messages"][:1]}])


if __name__ == "__main__":
    unittest.main()
