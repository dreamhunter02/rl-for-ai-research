import asyncio
import json
import unittest

from finance_env import FinanceAnswerReward


class TinkerRewardTests(unittest.TestCase):
    def test_grounded_finish_reward_uses_read_evidence(self):
        history = [
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "read-1",
                "function": {"name": "read", "arguments": json.dumps({"document_id": "amcor_8k", "start": 1, "end": 2})},
            }]},
            {"role": "tool", "tool_call_id": "read-1", "content": "Amcor entered into supplemental indentures."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "finish-1",
                "function": {"name": "finish", "arguments": json.dumps({"answer": "Supplemental indentures."})},
            }]},
        ]
        reward, parts = asyncio.run(FinanceAnswerReward(["Amcor entered into supplemental indentures."])(history))
        self.assertEqual(reward, 1.0)
        self.assertEqual(parts["answer_quality"], 1.0)
        self.assertEqual(parts["evidence_quality"], 1.0)
        self.assertEqual(parts["strong_source"], 1.0)

    def test_correct_ungrounded_answer_is_capped_below_full_reward(self):
        history = [
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "Answer: supplemental indentures.", "tool_calls": []},
        ]
        reward, parts = asyncio.run(FinanceAnswerReward(["Amcor entered into supplemental indentures."])(history))
        self.assertEqual(parts["answer_quality"], 1.0)
        self.assertEqual(parts["evidence_quality"], 0.0)
        self.assertEqual(reward, 0.5)


if __name__ == "__main__":
    unittest.main()
