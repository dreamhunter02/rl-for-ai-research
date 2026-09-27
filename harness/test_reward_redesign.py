import asyncio
import unittest

from reward_calculation import RewardConfig, _numeric_or_decision_hard_veto, answer_quality_with_judge, reward_formula


class FakeJudge:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def judge(self, **kwargs):
        self.calls += 1
        return dict(self.result)


class RewardRedesignTests(unittest.TestCase):
    def test_judge_recovers_qualitative_paraphrase(self):
        judge = FakeJudge({"verdict": "entailed", "confidence": 0.96, "cache_hit": False, "reason_code": "paraphrase"})
        q, meta = asyncio.run(answer_quality_with_judge(
            question="What industry does AMCOR primarily operate in?",
            gold="Amcor is a global leader in packaging production for various use cases.",
            candidate="Amcor primarily operates in the packaging industry, focusing on flexible and rigid packaging.",
            evidence="Amcor is a global leader in developing and producing responsible packaging.",
            judge=judge,
            config=RewardConfig(),
        ))
        self.assertEqual(q, 1.0)
        self.assertEqual(meta["judge_used"], 1.0)
        self.assertEqual(judge.calls, 1)

    def test_hard_decision_veto_beats_judge(self):
        judge = FakeJudge({"verdict": "entailed", "confidence": 0.99})
        q, meta = asyncio.run(answer_quality_with_judge(
            question="Has Verizon increased its debt?",
            gold="No. Verizon's debt decreased by $229 million.",
            candidate="Yes. Verizon's debt increased by $407 million.",
            evidence="Debt increased from $1,325 million to $1,732 million.",
            judge=judge,
            config=RewardConfig(),
        ))
        self.assertEqual(q, 0.0)
        self.assertEqual(meta["hard_gate"], "explicit_contradiction")
        self.assertEqual(judge.calls, 0)

    def test_ambiguous_judge_does_not_reward(self):
        judge = FakeJudge({"verdict": "ambiguous", "confidence": 0.99})
        q, meta = asyncio.run(answer_quality_with_judge(
            question="What industry does AMCOR primarily operate in?",
            gold="Amcor is a global leader in packaging production.",
            candidate="The company has diverse operations worldwide.",
            evidence="Amcor has operations in several regions.",
            judge=judge,
            config=RewardConfig(),
        ))
        self.assertEqual(q, 0.0)
        self.assertEqual(meta["judge_verdict"], "ambiguous")

    def test_numeric_scale_suffixes_do_not_trigger_veto(self):
        self.assertEqual(_numeric_or_decision_hard_veto("Revenue was $4.2 Billion.", "Revenue was $4200M."), (False, ""))
        self.assertEqual(_numeric_or_decision_hard_veto("Revenue was $4.2B.", "Revenue was $4.2 million."), (True, "numeric_mismatch"))

    def test_grounded_formula(self):
        self.assertEqual(reward_formula(1.0, 1.0), 1.0)
        self.assertEqual(reward_formula(1.0, 0.0), 0.5)
        self.assertEqual(reward_formula(0.0, 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
