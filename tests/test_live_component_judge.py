"""Opt-in paid/remote judge controls; ordinary test runs never call an endpoint."""
import asyncio
import os
import pytest
from component_reward import ComponentJudge, score_episode

pytestmark=pytest.mark.skipif(os.environ.get('RUN_LIVE_JUDGE_TESTS')!='1',reason='explicit live-judge opt-in required')


@pytest.mark.parametrize('question,reference,mode', [
    ('Is 3M a capital-intensive business based on FY2022 data?',
     'No. CAPEX/Revenue 5.1%; fixed assets/total assets 20%; return on assets 12.4%.', 'semantic'),
    ('How much did revenue grow in 2022?', 'Revenue grew 10%, from 100 to 110.', 'numeric'),
    ('Calculate the quick ratio and assess whether liquidity is healthy using a threshold of 1.',
     '0.96; no, it is below 1.', 'mixed'),
    ('What is the inventory turnover ratio? If this metric is not meaningful, explain why.',
     '9.5 times.', 'numeric'),
])
def test_live_question_controls_applicability(question,reference,mode):
    rubric=asyncio.run(ComponentJudge().make_rubric(question,reference))
    assert rubric['mode']==mode
    if mode=='semantic': assert rubric['numeric']==[]


def test_live_semantic_answer_still_checks_volunteered_arithmetic():
    result=asyncio.run(score_episode(question='Is liquidity healthy using a threshold of 1?',
        rubric={'mode':'semantic','numeric':[],'semantic':[{'id':'s1','expected':'No; below 1'}]},
        submission={'answer_type':'text','answer_text':'No. The ratio is 0.6, calculated as 50 divided by 100.',
                    'citations':[{'receipt_id':'r1','document_id':'D','page':1}]},
        receipts={'r1':{'document_id':'D','page':1,'text':'Liquid assets 50. Current liabilities 100.'}},
        F=1,judge=ComponentJudge()))
    assert not result['unresolved'] and result['N'] is None and result['S']==1 and result['G']<1


@pytest.mark.parametrize('value,unit,scale,N', [('0.5','ratio','ones',1),('0.5','ratio','thousand',0),('2','ratio','ones',0)])
def test_live_final_result_and_scale(value,unit,scale,N):
    result=asyncio.run(score_episode(question='What is the current ratio?',rubric={'mode':'numeric','numeric':[{'id':'n1','expected':'0.5 times'}],'semantic':[]},
        submission={'answer_type':'numeric','value':value,'unit':unit,'scale':scale,'citations':[{'receipt_id':'r1','document_id':'D','page':1}]},
        receipts={'r1':{'document_id':'D','page':1,'text':'Current assets 50. Current liabilities 100. Both in USD millions.'}},F=1,judge=ComponentJudge()))
    assert not result['unresolved'] and result['N']==N


def test_live_right_number_wrong_conclusion():
    result=asyncio.run(score_episode(question='Is liquidity healthy? Use the quick ratio and a threshold of 1.',rubric={'mode':'mixed','numeric':[{'id':'n1','expected':'0.5'}],'semantic':[{'id':'s1','expected':'No; below 1'}]},
        submission={'answer_type':'text','answer_text':'Yes, healthy; the quick ratio is 0.5.','citations':[{'receipt_id':'r1','document_id':'D','page':1}]},
        receipts={'r1':{'document_id':'D','page':1,'text':'Cash plus receivables 50. Current liabilities 100.'}},F=1,judge=ComponentJudge()))
    assert not result['unresolved'] and result['N']==1 and result['S']==0


def test_live_correct_answer_without_citation_not_grounded():
    result=asyncio.run(score_episode(question='What is days payable outstanding?',rubric={'mode':'numeric','numeric':[{'id':'n1','expected':'40 days'}],'semantic':[]},
        submission={'answer_type':'numeric','value':'40','unit':'number','scale':'ones','citations':[]},receipts={},F=1,judge=ComponentJudge()))
    assert not result['unresolved'] and result['N']==1 and result['G']==0 and result['reward']==.5
