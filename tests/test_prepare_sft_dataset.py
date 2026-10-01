import unittest

from prepare_sft_dataset import build_messages, eligible_record, identified_teacher


class PrepareSftDatasetTests(unittest.TestCase):
    def test_export_excludes_unknown_and_synthetic_teacher_identity(self):
        for model in ('', 'unknown', 'gold_recovery_v1'):
            self.assertFalse(identified_teacher({'teacher_model': model}))
        self.assertFalse(identified_teacher({}))
        self.assertTrue(identified_teacher({'teacher_model': 'openai/openai/gpt-5.6-terra'}))

    def test_calculator_and_failed_tool_actions_are_not_eligible(self):
        good = {'termination_reason':'finish','score':{'F':1,'A':1,'G':1},'tool_calls':[{'name':'finish'}]}
        self.assertFalse(eligible_record({**good,'tool_calls':[{'name':'calculate'},{'name':'finish'}]}))
        self.assertFalse(eligible_record({**good,'trace':[{'role':'tool','content':'{"error":"invalid page"}'}]}))

    def test_only_fully_verified_finished_trace_is_eligible(self):
        good = {
            "termination_reason": "finish",
            "score": {"F": 1, "A": 1, "G": 1, "unresolved": False},
            "tool_calls": [{"name": "finish", "arguments": {"answer_type": "text"}}],
        }
        self.assertTrue(eligible_record(good))
        for mutation in (
            {"termination_reason": "max_turns"},
            {"score": {"F": 1, "A": 0, "G": 1}},
            {"score": {"F": 1, "A": 1, "G": 0}},
            {"score": {"F": 1, "A": 1, "G": 1, "unresolved": True}},
            {"tool_calls": [{"name": "finish", "arguments": {}, "parse_error": "bad"}]},
        ):
            candidate = {**good, **mutation}
            self.assertFalse(eligible_record(candidate), mutation)

    def test_trace_converts_to_openai_tool_messages_without_plain_answer(self):
        record = {
            "question": "What was revenue?",
            "trace": [
                {"role": "assistant", "tool_calls": [{
                    "name": "read",
                    "arguments": {"document_id": "D", "page": 1},
                    "call_id": "c1",
                }]},
                {"role": "tool", "call_id": "c1", "content": "evidence"},
                {"role": "assistant", "tool_calls": [{
                    "name": "finish",
                    "arguments": {
                        "answer_type": "numeric",
                        "value": "100",
                        "unit": "USD",
                        "scale": "million",
                    },
                    "call_id": "c2",
                }]},
                {"role": "tool", "call_id": "c2", "content": "accepted"},
            ],
        }

        messages = build_messages(record, "SYSTEM")

        self.assertEqual(messages[:2], [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "What was revenue?"},
        ])
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["arguments"], {
            "document_id": "D", "page": 1,
        })
        self.assertEqual(messages[3], {
            "role": "tool", "tool_call_id": "c1", "content": "evidence",
        })
        self.assertEqual(messages[-2]["tool_calls"][0]["function"]["name"], "finish")
        self.assertNotEqual(messages[-1]["role"], "assistant")

    def test_trace_with_rejected_then_accepted_finish_is_not_eligible(self):
        record = {
            "termination_reason": "finish",
            "score": {"F": 1, "A": 1, "G": 1, "unresolved": False},
            "tool_calls": [
                {"name": "finish", "arguments": {"answer_type": "numeric", "answer_text": "invalid"}},
                {"name": "finish", "arguments": {"answer_type": "numeric", "value": "100"}},
            ],
        }

        self.assertFalse(eligible_record(record))


if __name__ == "__main__":
    unittest.main()
