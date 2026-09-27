import unittest

from process_guidance import guardrail_for, process_components, teacher_phase_signal


class ProcessGuidanceTests(unittest.TestCase):
    def test_repeated_call_guardrail(self):
        calls = [
            {"name": "bm25_search", "arguments": {"query_list": ["revenue"]}},
            {"name": "bm25_search", "arguments": {"query_list": ["revenue"]}},
        ]
        self.assertIsNotNone(guardrail_for(calls))

    def test_search_pivot_is_positive(self):
        trace = [
            {"role": "assistant", "tool_calls": [{"name": "bm25_search", "arguments": {}}]},
            {"role": "tool", "content": "search"},
            {"role": "assistant", "tool_calls": [{"name": "read", "arguments": {}}]},
        ]
        parts = process_components(trace, question="What was revenue?")
        self.assertEqual(parts["pivoted_to_evidence"], 1.0)
        self.assertGreater(parts["process_quality"], 0.0)

    def test_arithmetic_without_calculator_is_penalized(self):
        trace = [{"role": "assistant", "tool_calls": [{"name": "finish", "arguments": {"answer": "10%"}}]}]
        parts = process_components(trace, question="What was the percentage increase?", answer="10%")
        self.assertEqual(parts["arithmetic_compliance"], 0.0)
        self.assertGreater(parts["premature_answer"], 0.0)

    def test_teacher_signal_uses_phase_sets_not_order(self):
        teacher = [
            {"role": "assistant", "tool_calls": [{"name": "bm25_search", "arguments": {}}]},
            {"role": "assistant", "tool_calls": [{"name": "read_table", "arguments": {}}]},
            {"role": "assistant", "tool_calls": [{"name": "finish", "arguments": {}}]},
        ]
        student = [
            {"role": "assistant", "tool_calls": [{"name": "search_tables", "arguments": {}}]},
            {"role": "assistant", "tool_calls": [{"name": "finish", "arguments": {}}]},
        ]
        self.assertAlmostEqual(teacher_phase_signal(student, teacher), 2 / 3)


if __name__ == "__main__":
    unittest.main()
