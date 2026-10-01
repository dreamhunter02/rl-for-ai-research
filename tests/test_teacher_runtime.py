import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import financebench_harness as hb
from teacher_runtime import TeacherHarnessSession


class TeacherEndpointClientTests(unittest.TestCase):
    def test_resume_ids_include_real_trajectories_but_not_empty_rate_limits(self):
        import generate_teacher_traces as generator

        with tempfile.TemporaryDirectory() as directory:
            success = Path(directory) / "success.jsonl"
            failure = Path(directory) / "failure.jsonl"
            success.write_text(json.dumps({
                "financebench_id": "verified-id",
                "trace": [{"role": "assistant", "tool_calls": []}],
            }) + "\n")
            failure.write_text("\n".join([
                json.dumps({
                    "financebench_id": "rejected-with-trace",
                    "trace": [{"role": "assistant", "tool_calls": []}],
                    "error": "typed gate failed",
                }),
                json.dumps({
                    "financebench_id": "rate-limited-id",
                    "trace": [],
                    "error": "RateLimitError('Error code: 429')",
                }),
            ]) + "\n")

            completed = generator.completed_trace_ids([success, failure])

        self.assertEqual(completed, {"verified-id", "rejected-with-trace"})

    def test_rate_limit_errors_are_identified_before_advancing_dataset(self):
        import generate_teacher_traces as generator

        class RateLimitError(Exception):
            pass

        self.assertTrue(generator.is_rate_limit_error(RateLimitError("Error code: 429")))
        self.assertTrue(generator.is_rate_limit_error(RuntimeError("HTTP 429")))
        self.assertFalse(generator.is_rate_limit_error(RuntimeError("HTTP 500")))

    def test_gpt_56_terra_request_omits_unsupported_temperature(self):
        import generate_teacher_traces as generator

        terra = generator.chat_completion_options(
            "openai/openai/gpt-5.6-terra",
            temperature=0.7,
            seed=42,
        )
        deepseek = generator.chat_completion_options(
            "nvidia/deepseek-ai/deepseek-v4-flash",
            temperature=0.7,
            seed=42,
        )

        self.assertEqual(terra, {"seed": 42})
        self.assertEqual(deepseek, {"temperature": 0.7, "seed": 42})

    def test_teacher_client_bounds_stalled_endpoint_requests(self):
        import generate_teacher_traces as generator

        client = generator.teacher_client(
            base_url="http://127.0.0.1:9/v1",
            api_key="test-only",
            request_timeout_s=120.0,
            max_retries=1,
        )

        self.assertEqual(client.timeout.connect, 5.0)
        self.assertEqual(client.timeout.read, 120.0)
        self.assertEqual(client.timeout.write, 120.0)
        self.assertEqual(client.max_retries, 1)


class TeacherRuntimeTests(unittest.TestCase):
    def test_teacher_generator_imports_without_legacy_eval_agent_api(self):
        import generate_teacher_traces as generator

        self.assertIn("typed finish", generator.TEACHER_SYSTEM.lower())

    def test_endpoint_key_can_be_read_from_environment_without_cli_secret(self):
        import generate_teacher_traces as generator

        with patch.dict(os.environ, {"TEACHER_ENDPOINT_KEY": "secret"}, clear=False):
            self.assertEqual(generator.resolve_api_key("", "TEACHER_ENDPOINT_KEY"), "secret")
        self.assertEqual(generator.resolve_api_key("explicit", "TEACHER_ENDPOINT_KEY"), "explicit")

    def test_rows_and_corpus_documents_come_from_explicit_frozen_split(self):
        import generate_teacher_traces as generator

        split = {
            "train": [{
                "financebench_id": "train-1",
                "question": "What was revenue?",
                "answer": "100",
                "company": "Example",
                "doc_name": "TRAIN_DOC",
            }],
            "dev": [{
                "financebench_id": "dev-1",
                "question": "Was revenue disclosed?",
                "answer": "Yes",
                "company": "Example",
                "doc_name": "DEV_DOC",
            }],
            "eval": [{"doc_name": "EVAL_DOC"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(split))

            rows = generator._rows("train", path)
            documents = generator.split_documents(path)

        self.assertEqual([row["financebench_id"] for row in rows], ["train-1"])
        self.assertEqual(documents, ["DEV_DOC", "EVAL_DOC", "TRAIN_DOC"])

    def test_unresolved_target_is_not_eligible_for_teacher_generation(self):
        import generate_teacher_traces as generator

        self.assertFalse(generator.teacher_target_is_resolved({
            "adjudication_status": "unresolved",
            "adjudication_reason": "source contradicts frozen answer",
        }))
        self.assertTrue(generator.teacher_target_is_resolved({
            "adjudication_status": "resolved",
        }))

    def test_malformed_tool_arguments_become_recoverable_observation(self):
        import generate_teacher_traces as generator

        arguments, error = generator.parse_tool_arguments('{"answer_text":"unterminated')

        self.assertEqual(arguments, {})
        self.assertEqual(error, "Tool arguments were not valid JSON; correct them and retry.")

    def setUp(self):
        index = hb.StructuredIndex(
            [{
                "document_id": "D",
                "page": 1,
                "text": "Revenue in 2023 was USD 100 million.",
            }],
            [],
            [],
        )
        self.session = TeacherHarnessSession(index, max_turns=4)

    def test_finish_schema_is_current_typed_contract(self):
        finish = next(spec for spec in self.session.tool_specs() if spec["name"] == "finish")
        properties = finish["parameters"]["properties"]
        self.assertIn("answer_type", properties)
        self.assertIn("citations", properties)
        self.assertIn("derivation", properties)
        self.assertNotIn("calc_id", properties)
        self.assertNotIn("answer", properties)

    @patch.dict(os.environ, {'FINANCEBENCH_SCORER':'legacy'})
    def test_real_read_receipt_and_typed_finish_score_full_reward(self):
        self.session.start_turn()
        read = self.session.execute(
            "read",
            {"document_id": "D", "page": 1},
            "read-1",
        )
        receipt = json.loads(read.content)["receipt_id"]

        self.session.start_turn()
        finish = self.session.execute(
            "finish",
            {
                "answer_type": "numeric",
                "value": "100",
                "unit": "USD",
                "scale": "million",
                "citations": [{
                    "receipt_id": receipt,
                    "document_id": "D",
                    "page": 1,
                }],
            },
            "finish-1",
        )
        target = {
            "reviewed": True,
            "adjudication_status": "resolved",
            "answer_type": "numeric",
            "value": "100",
            "unit": "USD",
            "scale": "million",
            "precision": 0,
            "derived": False,
            "support": [{
                "document_id": "D",
                "page": 1,
                "quote": "Revenue in 2023 was USD 100 million.",
            }],
        }

        self.assertTrue(finish.should_stop)
        self.assertEqual(self.session.accepted["value"], "100")
        score = self.session.score(target)
        self.assertEqual((score["F"], score["A"], score["G"]), (1, 1, 1))

    def test_legacy_finish_answer_is_rejected_without_stopping(self):
        self.session.start_turn()
        result = self.session.execute("finish", {"answer": "100"}, "legacy-1")
        self.assertFalse(result.should_stop)
        self.assertIsNone(self.session.accepted)

    def test_removed_calculator_is_rejected_without_mutating_state(self):
        quote = "Revenue in 2023 was USD 100 million. " + ("context " * 200)
        index = hb.StructuredIndex([{"document_id": "L", "page": 1, "text": quote}], [], [])
        session = TeacherHarnessSession(index, max_turns=4)
        session.start_turn()
        receipt = json.loads(session.execute(
            "read", {"document_id": "L", "page": 1}, "long-read",
        ).content)["receipt_id"]
        operand = {
            "value": "100", "unit": "USD", "scale": "million",
            "metric": "Revenue", "period": "2023", "receipt_id": receipt, "quote": quote,
        }
        session.start_turn()

        result = json.loads(session.execute(
            "calculate",
            {"expression": "a+b+c", "operands": {"a": operand, "b": operand, "c": operand}},
            "long-calculate",
        ).content)

        self.assertIn('Unknown tool', result['error'])
        self.assertEqual(session.state.calculations,{})

    @patch.dict(os.environ, {'FINANCEBENCH_SCORER':'legacy'})
    def test_semantic_judge_resolves_grounded_text_paraphrase(self):
        class Judge:
            async def judge(self, **kwargs):
                self.kwargs = kwargs
                return {
                    "verdict": "entailed",
                    "confidence": 0.95,
                    "numeric_ok": True,
                    "cache_hit": False,
                }

        self.session.start_turn()
        read = self.session.execute(
            "read", {"document_id": "D", "page": 1}, "read-text",
        )
        receipt = json.loads(read.content)["receipt_id"]
        self.session.start_turn()
        self.session.execute(
            "finish",
            {
                "answer_type": "text",
                "answer_text": "Sales reached one hundred million dollars.",
                "citations": [{
                    "receipt_id": receipt,
                    "document_id": "D",
                    "page": 1,
                }],
            },
            "finish-text",
        )
        target = {
            "reviewed": True,
            "adjudication_status": "resolved",
            "answer_type": "text",
            "aliases": ["Revenue was USD 100 million."],
            "required_facts": ["Revenue was USD 100 million."],
            "support": [{
                "document_id": "D",
                "page": 1,
                "quote": "Revenue in 2023 was USD 100 million.",
            }],
        }
        judge = Judge()

        score = self.session.score(target, question="What was revenue?", judge=judge)

        self.assertEqual((score["F"], score["A"], score["G"]), (1, 1, 1))
        self.assertEqual(score["judge_verdict"], "entailed")
        self.assertIn("Revenue was USD 100 million", judge.kwargs["gold"])


if __name__ == "__main__":
    unittest.main()
