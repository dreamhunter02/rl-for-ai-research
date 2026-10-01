import unittest
import json
import tempfile
from pathlib import Path

from build_gold_recovery_traces import _covered_ids, choose_claim_supports, choose_operand_support


class GoldRecoveryTraceTests(unittest.TestCase):
    def setUp(self):
        class Index:
            pages_by_doc = {
                "DOC": [{"page": 2, "text": "prefix Revenue 100 in 2024. Profit 20 in 2024. suffix"}],
            }

        self.index = Index()

    def test_claim_support_chooses_shortest_exact_readable_quote(self):
        target = {"support": [
            {"claim": "answer", "document_id": "DOC", "page": 2, "quote": "Revenue 100 in 2024. Profit 20 in 2024."},
            {"claim": "answer", "document_id": "DOC", "page": 2, "quote": "Revenue 100 in 2024."},
        ]}

        selected = choose_claim_supports(self.index, target, max_read=30)

        self.assertEqual(selected[0]["quote"], "Revenue 100 in 2024.")
        self.assertEqual(selected[0]["start"], 7)

    def test_operand_support_requires_metric_period_and_value(self):
        operand = {
            "value": "100",
            "metric": "Revenue",
            "period": "2024",
            "support": [
                {"document_id": "DOC", "page": 2, "quote": "Profit 20 in 2024."},
                {"document_id": "DOC", "page": 2, "quote": "Revenue 100 in 2024."},
            ],
        }

        selected = choose_operand_support(self.index, operand, max_read=30)

        self.assertEqual(selected["quote"], "Revenue 100 in 2024.")

    def test_covered_ids_excludes_full_score_with_invalid_trajectory_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({
                "financebench_id": "q1",
                "termination_reason": "finish",
                "score": {"F": 1, "A": 1, "G": 1},
                "tool_calls": [
                    {"name": "finish", "arguments": {"answer_type": "text"}},
                    {"name": "finish", "arguments": {"answer_type": "text"}},
                ],
            }) + "\n")

            self.assertEqual(_covered_ids([str(path)]), set())


if __name__ == "__main__":
    unittest.main()
