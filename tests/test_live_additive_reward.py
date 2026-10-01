"""Explicit opt-in controls for the offline additive protocol."""
import asyncio
import os
import pytest
from additive_rescore import PROMPT, VERSION, assemble
from component_reward import ComponentJudge

pytestmark=pytest.mark.skipif(os.environ.get('RUN_LIVE_JUDGE_TESTS')!='1',reason='explicit live-judge opt-in required')


@pytest.mark.parametrize('answer,cited,F,A,B,G,want', [
    ('No. The quick ratio is below 1.',True,1,1,0,1,.85),
    ('No; quick ratio is 0.5, below the threshold of 1.',True,1,1,1,1,1.),
    ('Yes, healthy. The quick ratio is 0.5.',True,1,0,0,1,.4),
    ('No; quick ratio is 0.5, below the threshold of 1.',False,1,1,1,0,.8),
])
def test_live_additive_separation(answer,cited,F,A,B,G,want):
    rubric={'mode':'semantic','numeric':[],'semantic':[{'id':'s1','expected':'No, not healthy under the stated threshold.'}],
            'gold_claims':[{'id':'b1','expected':'No, not healthy.'},{'id':'b2','expected':'Quick ratio is 0.5.'}]}
    receipts={'r1':{'document_id':'D','page':1,'text':'Quick assets: 50. Current liabilities: 100. Both USD million.'}}
    payload=dict(protocol=VERSION,question='Is liquidity healthy using a quick-ratio threshold of 1?',rubric=rubric,
                 candidate={'answer_text':answer},retrieved_evidence=receipts,
                 valid_cited_receipt_ids=['r1'] if cited else [],invalid_citations=[])
    raw=asyncio.run(ComponentJudge()._retry(PROMPT,payload))
    result=assemble(raw,rubric,receipts,receipts if cited else {},F)
    assert result['E']==1 and result['A']==A and result['B']==B and result['G']==G
    assert result['reward']==pytest.approx(want)
