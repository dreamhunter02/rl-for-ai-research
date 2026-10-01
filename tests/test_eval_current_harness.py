import unittest

from eval_current_harness import summarize


def test_resume_rejects_configuration_changes(tmp_path):
    import eval_current_harness as ev
    assert hasattr(ev,'verify_run_config'), 'resume configuration gate missing'
    out=tmp_path/'run.jsonl'
    ev.verify_run_config(out,{'model':'one'})
    out.write_text('{}\n')
    import pytest
    with pytest.raises(ValueError): ev.verify_run_config(out,{'model':'two'})


class CurrentHarnessEvalTests(unittest.TestCase):
    def test_summary_counts_finish_correct_and_grounded_independently(self):
        records = [
            {"termination_reason": "finish", "score": {"F": 1, "A": 1, "G": 1}},
            {"termination_reason": "finish", "score": {"F": 1, "A": 1, "G": 0}},
            {"termination_reason": "max_turns", "score": {"F": 0, "A": 0, "G": 0}},
        ]

        self.assertEqual(
            summarize(records),
            {
                "n": 3,
                "finished": 2,
                "correct": 2,
                "grounded": 1,
                "finish_rate": 2 / 3,
                "correct_rate": 2 / 3,
                "grounded_rate": 1 / 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
