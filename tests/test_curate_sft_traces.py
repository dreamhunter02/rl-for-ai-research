import unittest

from curate_sft_traces import normalized_model, select_balanced


class CurateSftTracesTests(unittest.TestCase):
    def test_prefers_fewer_calls_among_equally_verified_traces(self):
        good={'financebench_id':'q','teacher_model':'m','termination_reason':'finish','score':{'F':1,'A':1,'G':1}}
        slow={**good,'tag':'slow','tool_calls':[{'name':'read'},{'name':'read'},{'name':'finish'}]}
        fast={**good,'tag':'fast','tool_calls':[{'name':'read'},{'name':'finish'}]}
        self.assertEqual(select_balanced([slow,fast])[0]['tag'],'fast')
    def test_model_aliases_collapse_to_one_teacher(self):
        self.assertEqual(
            normalized_model("deepseek_v4_flash"),
            "nvidia/deepseek-ai/deepseek-v4-flash",
        )

    def test_selects_one_full_reward_trace_per_question_and_teacher(self):
        good = {
            "score": {"F": 1, "A": 1, "G": 1, "unresolved": False},
            "termination_reason": "finish",
            "tool_calls": [{"name": "finish", "arguments": {"answer_type": "text"}}],
        }
        rows = [
            {**good, "financebench_id": "q1", "teacher_model": "deepseek_v4_flash", "trace": [1]},
            {**good, "financebench_id": "q1", "teacher_model": "nvidia/deepseek-ai/deepseek-v4-flash", "trace": [2]},
            {**good, "financebench_id": "q1", "teacher_model": "gpt56_terra", "trace": [3]},
            {**good, "financebench_id": "q2", "teacher_model": "gpt56_terra", "trace": [4]},
            {"score": {"F": 1, "A": 1, "G": 0}, "financebench_id": "q3", "teacher_model": "gpt56_terra"},
        ]

        selected = select_balanced(rows)

        self.assertEqual([(r["financebench_id"], r["teacher_model"]) for r in selected], [
            ("q1", "nvidia/deepseek-ai/deepseek-v4-flash"),
            ("q1", "openai/openai/gpt-5.6-terra"),
            ("q2", "openai/openai/gpt-5.6-terra"),
        ])


if __name__ == "__main__":
    unittest.main()
