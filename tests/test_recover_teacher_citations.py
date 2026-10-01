import unittest

from recover_teacher_citations import (
    canonical_support_reads,
    execute_support_reads,
    insert_tool_turn_before_finish,
    repair_citations,
)


class CitationRecoveryTests(unittest.TestCase):
    def test_canonical_support_reads_target_exact_quote_window(self):
        class Index:
            pages_by_doc = {
                "DOC": [{"page": 4, "text": "prefix Revenue was 100. suffix"}],
            }

        requests = canonical_support_reads(Index(), {
            "support": [{"claim": "revenue", "document_id": "DOC", "page": 4, "quote": "Revenue was 100."}],
        })

        self.assertEqual(requests, [{
            "document_id": "DOC",
            "page": 4,
            "start": 7,
            "end": 23,
        }])

    def test_insert_tool_turn_places_real_receipt_before_finish(self):
        trace = [{"role": "assistant", "tool_calls": [{"name": "finish", "call_id": "f1", "arguments": {}}]}]

        repaired = insert_tool_turn_before_finish(
            trace,
            [{"name": "read", "call_id": "repair-read-1", "arguments": {"document_id": "DOC", "page": 4}}],
            ["receipt-json"],
        )

        self.assertEqual(repaired[0]["tool_calls"][0]["name"], "read")
        self.assertEqual(repaired[1], {"role": "tool", "call_id": "repair-read-1", "content": "receipt-json"})
        self.assertEqual(repaired[2]["tool_calls"][0]["name"], "finish")

    def test_execute_support_reads_advances_to_a_later_finish_turn(self):
        class Result:
            content = "receipt-json"

        class Session:
            def __init__(self):
                self.turn = 0

            def start_turn(self):
                self.turn += 1

            def execute(self, name, arguments, call_id):
                self.delivered_turn = self.turn
                return Result()

        session = Session()
        calls, observations = execute_support_reads(
            session,
            [{"document_id": "DOC", "page": 4}],
            "abcdef1234567890",
        )

        self.assertEqual(session.delivered_turn, 1)
        self.assertEqual(session.turn, 2)
        self.assertEqual(calls[0]["name"], "read")
        self.assertEqual(observations, ["receipt-json"])

    def test_repair_citations_cover_every_claim_with_strong_exact_receipts(self):
        target = {
            "support": [
                {"claim": "revenue", "document_id": "DOC", "page": 4, "quote": "Revenue was 100."},
                {"claim": "profit", "document_id": "DOC", "page": 5, "quote": "Profit was 20."},
            ],
        }
        receipts = {
            "r1": {"receipt_id": "r1", "tool": "read", "document_id": "DOC", "page": 4, "text": "Revenue was 100."},
            "r2": {"receipt_id": "r2", "tool": "grep_document", "document_id": "DOC", "page": 5, "text": "Profit was 20."},
            "r3": {"receipt_id": "r3", "tool": "bm25_search", "document_id": "DOC", "page": 5, "text": "Profit was 20."},
        }

        citations = repair_citations(target, receipts, {}, {"answer_type": "text"})

        self.assertEqual(citations, [
            {"receipt_id": "r1", "document_id": "DOC", "page": 4},
            {"receipt_id": "r2", "document_id": "DOC", "page": 5},
        ])

    def test_repair_citations_includes_every_derived_operand_receipt(self):
        target = {
            "derived": True,
            "support": [{"claim": "answer", "document_id": "DOC", "page": 4, "quote": "Revenue was 100."}],
        }
        receipts = {
            "r1": {"receipt_id": "r1", "tool": "read", "document_id": "DOC", "page": 4, "text": "Revenue was 100."},
            "r2": {"receipt_id": "r2", "tool": "read_table", "document_id": "DOC", "page": 6, "text": "Assets were 50."},
        }
        calculations = {"c1": {"operands": {"revenue": {"receipt_id": "r1"}, "assets": {"receipt_id": "r2"}}}}

        citations = repair_citations(
            target,
            receipts,
            calculations,
            {"answer_type": "numeric", "calc_id": "c1"},
        )

        self.assertEqual(citations, [
            {"receipt_id": "r1", "document_id": "DOC", "page": 4},
            {"receipt_id": "r2", "document_id": "DOC", "page": 6},
        ])


if __name__ == "__main__":
    unittest.main()
