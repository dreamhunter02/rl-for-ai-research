"""The live GRPO reward must use the same additive components as offline v6."""

import asyncio

import pytest

from live_additive_reward import LiveAdditiveJudge


class JudgeFixture:
    async def make_rubric(self, question, reference):
        return {"mode": "numeric", "numeric": [{"id": "n1", "expected": "9.5 times"}],
                "semantic": [], "reference": reference}

    async def _retry(self, prompt, payload):
        if "reference" in payload:
            return {"claims": [{"id": "b1", "expected": "9.5 times"}]}
        return {"numeric": [{"id": "n1", "correct": True, "reason": "9.5 times"}],
                "semantic": [], "retrieval": {"coverage": "full", "receipt_ids": ["r1"], "reason": "source"},
                "grounding": [{"claim": "9.5 times", "supported": True,
                               "receipt_ids": ["r1"], "reason": "source and arithmetic"}],
                "completeness": [{"id": "b1", "correct": True, "reason": "complete"}],
                "material_contradiction": False, "unresolved": False}


def test_live_reward_matches_v6_for_accepted_grounded_finish():
    scorer = LiveAdditiveJudge(JudgeFixture())
    receipts = {"r1": {"receipt_id": "r1", "document_id": "D", "page": 2,
                       "text": "Cost: 95. Inventory: 10."}}
    submission = {"answer_type": "numeric", "value": "9.5", "unit": "ratio", "scale": "ones",
                  "citations": [{"receipt_id": "r1", "document_id": "D", "page": 2}]}
    result = asyncio.run(scorer.score("q1", "What is the ratio?", "9.5 times", submission, receipts))
    assert result["reward"] == pytest.approx(1.0)
    assert (result["E"], result["G"], result["A"], result["B"], result["F"]) == (1, 1, 1, 1, 1)
    assert not result["unresolved"]


def test_live_reward_keeps_retrieval_credit_without_finish():
    scorer = LiveAdditiveJudge(JudgeFixture())
    receipts = {"r1": {"receipt_id": "r1", "document_id": "D", "page": 2, "text": "Cost: 95."}}
    result = asyncio.run(scorer.score("q1", "What is the ratio?", "9.5 times", None, receipts))
    assert result["F"] == 0
    assert result["reward"] == pytest.approx(0.15)


def test_empty_gold_claims_use_reference_checked_core_rubric():
    class EmptyGoldJudge(JudgeFixture):
        async def make_rubric(self, question, reference):
            return {"mode": "semantic", "numeric": [], "semantic": [
                {"id": "s1", "expected": "American Express had no debt securities registered to trade on a national securities exchange as of 2022."}
            ], "reference": reference}

        async def _retry(self, prompt, payload):
            return {"claims": []}

    scorer = LiveAdditiveJudge(EmptyGoldJudge())
    rubric = asyncio.run(scorer.rubric("financebench_id_00476", "Which debt securities are registered?", "There are none"))
    assert rubric["gold_claims"] == [{"id": "b1", "expected": "American Express had no debt securities registered to trade on a national securities exchange as of 2022."}]
