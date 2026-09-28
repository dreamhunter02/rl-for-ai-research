import asyncio
import json
import unittest

from finance_env import CoercingTool, FinanceAnswerReward, FinanceRLDataset, _bound_read_payload, _compact_hit, load_financebench
from tinker_cookbook.tool_use import ToolInput, simple_tool_result


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
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "finish-1",
                "function": {"name": "finish", "arguments": json.dumps({"answer": "Supplemental indentures."})},
            }]},
        ]
        reward, parts = asyncio.run(FinanceAnswerReward(["Amcor entered into supplemental indentures."])(history))
        self.assertEqual(parts["answer_quality"], 1.0)
        self.assertEqual(parts["evidence_quality"], 0.0)
        self.assertEqual(reward, 0.5)


    def test_unsupported_finish_citation_removes_grounding_credit(self):
        history = [
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "read-1", "function": {"name": "read", "arguments": json.dumps({"document_id": "amcor_8k", "page": 3})},
            }]},
            {"role": "tool", "tool_call_id": "read-1", "content": json.dumps({"document_id": "amcor_8k", "page_start": 3, "page_end": 3, "text": "Amcor entered into supplemental indentures.", "provenance": "amcor_8k:pages=3-3"})},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "finish-1", "function": {"name": "finish", "arguments": json.dumps({"answer": "Supplemental indentures.", "evidence_document": "wrong_doc", "evidence_page": 99})},
            }]},
        ]
        reward, parts = asyncio.run(FinanceAnswerReward(["Amcor entered into supplemental indentures."])(history))
        self.assertEqual(parts["answer_quality"], 1.0)
        self.assertEqual(parts["citation_valid"], 0.0)
        self.assertEqual(parts["evidence_quality"], 0.0)
        self.assertEqual(reward, 0.0)


    def test_supported_finish_citation_retains_grounding_credit(self):
        history = [
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "read-1", "function": {"name": "read", "arguments": json.dumps({"document_id": "amcor_8k", "page": 3})}}]},
            {"role": "tool", "tool_call_id": "read-1", "content": json.dumps({"document_id": "amcor_8k", "page_start": 3, "page_end": 3, "text": "Amcor entered into supplemental indentures.", "provenance": "amcor_8k:pages=3-3"})},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "finish-1", "function": {"name": "finish", "arguments": json.dumps({"answer": "Supplemental indentures.", "evidence_document": "amcor_8k", "evidence_page": 3})}}]},
        ]
        reward, parts = asyncio.run(FinanceAnswerReward(["Amcor entered into supplemental indentures."])(history))
        self.assertEqual(parts["citation_valid"], 1.0)
        self.assertEqual(reward, 1.0)

    def test_tinker_metrics_are_numeric(self):
        history = [
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "finish-1",
                "function": {"name": "finish", "arguments": json.dumps({"answer": "Supplemental indentures."})},
            }]},
        ]
        _, parts = asyncio.run(FinanceAnswerReward(["Amcor entered into supplemental indentures."])(history))
        self.assertTrue(all(isinstance(value, (int, float, bool)) for value in parts.values()), parts)

    def test_missing_finish_is_gated(self):
        history = [
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "Answer: supplemental indentures.", "tool_calls": []},
        ]
        reward, parts = asyncio.run(FinanceAnswerReward(["Amcor entered into supplemental indentures."])(history))
        self.assertEqual(reward, 0.0)
        self.assertEqual(parts["finish_missing"], 1.0)
        self.assertEqual(parts["finish_gate"], 0.0)

    def test_coercing_tool_forwards_schema_arguments_without_mutating_call(self):
        seen = {}

        class FakeTool:
            name = "bm25_search"

            def to_spec(self):
                return {"name": self.name, "parameters": {"properties": {"query_list": {"type": "array"}}}}

            async def run(self, call):
                seen["call"] = call
                return simple_tool_result("ok")

        call = ToolInput(arguments={"query_list": '["revenue", "cash"]'}, call_id="c1")
        result = asyncio.run(CoercingTool(FakeTool()).run(call))
        self.assertEqual(seen["call"].arguments["query_list"], ["revenue", "cash"])
        self.assertEqual(seen["call"].call_id, "c1")
        self.assertEqual(call.arguments["query_list"], '["revenue", "cash"]')
        self.assertFalse(result.should_stop)

    def test_finish_tool_result_stops_episode(self):
        class FakeTool:
            name = "finish"

            def to_spec(self):
                return {"name": self.name, "parameters": {"properties": {"answer": {"type": "string"}}}}

            async def run(self, call):
                return simple_tool_result("accepted")

        result = asyncio.run(CoercingTool(FakeTool()).run(ToolInput({"answer": "0"}, "finish-1")))
        self.assertTrue(result.should_stop)


    def test_retrieval_hit_is_compact_and_provenance_preserving(self):
        hit = _compact_hit({
            "document_id": "doc", "page": 4, "passage_id": "p:doc:page:4:chunk:0",
            "score": 1.25, "section": "Balance sheet", "text": "x" * 1000,
            "internal": "must not leak",
        }, kind="prose")
        self.assertEqual(hit["document_id"], "doc")
        self.assertEqual(hit["page"], 4)
        self.assertEqual(hit["passage_id"], "p:doc:page:4:chunk:0")
        self.assertLessEqual(len(hit["snippet"]), 300)
        self.assertNotIn("internal", hit)


    def test_read_payload_is_bounded_without_losing_pagination(self):
        payload = _bound_read_payload({"start": 0, "end": 8000, "total_chars": 9000, "has_more": True, "next_start": 8000, "text": "x" * 8000, "provenance": "doc:page=1"})
        self.assertLessEqual(len(payload["text"]), 2800)
        self.assertTrue(payload["has_more"])
        self.assertEqual(payload["next_start"], payload["end"])
        self.assertEqual(payload["provenance"], "doc:page=1")


    def test_repaired_train_dev_split_has_disjoint_counts(self):
        train = load_financebench("train96")
        dev = load_financebench("dev")
        self.assertEqual(len(train), 96)
        self.assertEqual(len(dev), 12)
        self.assertTrue({x["financebench_id"] for x in train}.isdisjoint({x["financebench_id"] for x in dev}))

    def test_dataset_keeps_remainder_batch(self):
        self.assertEqual(len(FinanceRLDataset(list(range(10)), 4)), 3)
        self.assertEqual(FinanceRLDataset(list(range(10)), 4).get_batch(2), [8, 9])


if __name__ == "__main__":
    unittest.main()
