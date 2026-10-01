"""Live v6 FinanceBench scoring for on-policy rollouts."""
from __future__ import annotations

import asyncio

from additive_rescore import GOLD_PROMPT, PROMPT, VERSION, assemble
from component_reward import authentic_citations


class LiveAdditiveJudge:
    def __init__(self, judge):
        self.judge = judge
        self._rubrics = {}

    async def rubric(self, question_id, question, reference):
        if question_id not in self._rubrics:
            rubric = await self.judge.make_rubric(question, reference)
            gold = await self.judge._retry(GOLD_PROMPT, {"protocol": VERSION, "reference": reference})
            claims = gold.get("claims")
            if claims == []:
                core = rubric["numeric"] + rubric["semantic"]
                claims = [{"id": f"b{n}", "expected": item["expected"]}
                          for n, item in enumerate(core, 1) if item.get("expected")]
                rubric["gold_claims_source"] = "reference_checked_core_rubric"
            if not isinstance(claims, list) or not claims or any(
                not isinstance(c, dict) or not c.get("id") or not c.get("expected") for c in claims
            ) or len({c["id"] for c in claims}) != len(claims):
                raise ValueError("invalid gold claims")
            rubric["gold_claims"] = claims
            self._rubrics[question_id] = rubric
        return self._rubrics[question_id]

    async def score(self, question_id, question, reference, submission, receipts):
        rubric = await self.rubric(question_id, question, reference)
        cited, invalid = authentic_citations(submission or {}, receipts)
        F = int(submission is not None)
        payload = dict(protocol=VERSION, question=question, rubric=rubric,
                       candidate=submission, retrieved_evidence=receipts,
                       valid_cited_receipt_ids=list(cited), invalid_citations=invalid)
        try:
            raw = await self.judge._retry(PROMPT, payload)
            result = assemble(raw, rubric, receipts, cited, F)
            result["invalid_citations"] = invalid
            return result
        except Exception as exc:
            return dict(E=None, G=None, A=None, B=None, F=F, reward=None, unresolved=True,
                        reason=f"{type(exc).__name__}: {str(exc)[:180]}", scorer_version=VERSION)

    def score_sync(self, question_id, question, reference, submission, receipts):
        return asyncio.run(self.score(question_id, question, reference, submission, receipts))
