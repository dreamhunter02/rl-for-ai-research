import json
import unittest
from pathlib import Path

import financebench_harness as hb


class RewardDesignTests(unittest.TestCase):
    def test_known_fully_correct_zero_reward_traces_are_recovered(self):
        path = Path(__file__).resolve().parents[1] / "results" / "teacher_traces" / "sft_judged_strict.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        target = [row for row in rows if row.get("deterministic_reward") == 0 and row.get("judge", {}).get("correctness") == 4 and row.get("judge", {}).get("grounding") == 2]
        self.assertEqual(len(target), 46)
        scores = [hb.reward(row["gold"], row.get("answer_text", "")) for row in target]
        self.assertGreaterEqual(min(scores), 0.9)

    def test_finish_and_verbose_answer_extraction(self):
        self.assertEqual(hb.extract_answer_text("**Answer:** No, the ratio was 0.54"), "No, the ratio was 0.54")
        self.assertEqual(hb.reward("$11588.00", "Answer: $11,588.00"), 1.0)

    def test_narrative_direction_and_missing_finish_are_scored(self):
        answer = (
            "The table shows an increase/(decrease) due to factors. "
            "Operating margin decreased by 1.7% because of litigation and PFAS exit costs."
        )
        self.assertGreaterEqual(hb.reward("Operating margin decreased by 1.7% due to litigation and PFAS exit costs.", answer), 0.9)

    def test_missing_required_numeric_claim_fails_grounding(self):
        trace = [
            {"role": "assistant", "tool_calls": [{"name": "bm25_search", "call_id": "c1"}]},
            {"role": "tool", "call_id": "c1", "content": "customer diversification and many customers; no concentration percentage reported"},
        ]
        evidence, parts = hb.score_evidence(
            "Yes, one customer accounted for 16% of consolidated net revenue",
            "Answer: Yes, AMD has a diversified customer base",
            trace,
        )
        self.assertEqual(evidence, 0.0)
        self.assertEqual(parts["gold_claim_support"], 0.0)

    def test_adversarial_wrong_answers_do_not_pass(self):
        self.assertLessEqual(hb.reward("No. quick ratio was 0.54 for Verizon.", "Answer: No, quick ratio was 0.64"), 0.5)
        self.assertEqual(hb.reward("No, company had negative working capital of -1561M.", "Answer: No, company had positive working capital of 1561M"), 0.0)
        self.assertEqual(hb.reward("Yes, sales increased 3.0% vs international decline of 0.6%.", "Answer: sales increased 3.0% vs international increased 0.6%"), 0.0)


if __name__ == "__main__":
    unittest.main()
