import asyncio
import importlib
import math
import pytest


def module():
    assert importlib.util.find_spec('component_reward'), 'component reward is not implemented'
    return importlib.import_module('component_reward')


@pytest.mark.parametrize('mode,N,S,G,F,want', [
    ('mixed', 1, 0, 1, 1, .6), ('numeric', 1, None, 0, 1, .5),
    ('semantic', None, 1, 1, 1, 1), ('mixed', 1, 1, 1, 0, 0)])
def test_formula(mode, N, S, G, F, want):
    assert module().aggregate_reward(F=F,N=N,S=S,G=G,mode=mode)['reward'] == want


@pytest.mark.parametrize('value', [None, float('nan'), 2, -1])
def test_invalid_component_is_unresolved(value):
    r = module().aggregate_reward(F=1,N=value,S=None,G=1,mode='numeric')
    assert r['unresolved'] and r['reward'] is None


class Judge:
    async def judge_components(self, payload):
        return {'numeric': [{'id':'n1','correct':True,'reason':'0.96 matches'}],
                'semantic': [{'id':'s1','correct':False,'reason':'Yes contradicts No'}],
                'grounding': [{'claim':'ratio','supported':True,'receipt_ids':['r1'],'reason':'source operands'}],
                'unresolved':False}


def test_mixed_keeps_numeric_credit_but_not_wrong_conclusion():
    r=asyncio.run(module().score_episode(question='healthy?',rubric={'mode':'mixed','numeric':[{'id':'n1','expected':'0.96'}],'semantic':[{'id':'s1','expected':'No'}]},submission={'answer_text':'Yes, 0.96','citations':[{'receipt_id':'r1','document_id':'D','page':1}]},receipts={'r1':{'document_id':'D','page':1,'text':'current assets 15,754; inventories 5,280; liabilities 10,936'}},F=1,judge=Judge()))
    assert (r['N'],r['S'],r['A'],r['reward']) == (1,0,.6,.6)


def test_fabricated_citation_cannot_ground():
    r=asyncio.run(module().score_episode(question='healthy?',rubric={'mode':'mixed','numeric':[{'id':'n1','expected':'0.96'}],'semantic':[{'id':'s1','expected':'No'}]},submission={'answer_text':'Yes, 0.96','citations':[{'receipt_id':'r1','document_id':'FAKE','page':1}]},receipts={'r1':{'document_id':'D','page':1,'text':'source'}},F=1,judge=Judge()))
    assert r['G']==0 and r['reward']==.3


def test_cache_identity_changes_with_model_and_evidence():
    m=module()
    assert m.cache_key({'a':1},'model1') != m.cache_key({'a':1},'model2')
    assert m.cache_key({'a':1},'model1') != m.cache_key({'a':2},'model1')


def test_live_missing_judge_is_explicitly_unresolved():
    m=module()
    assert hasattr(m,'score_live'), 'live scoring adapter missing'
    r=asyncio.run(m.score_live(question='Revenue?',target={'value':'100','unit':'USD','scale':'million'},submission={'value':'100'},receipts={},judge=None))
    assert r['unresolved'] and r['reward'] is None


def test_grounding_requires_nonempty_claim():
    class Bad(Judge):
        async def judge_components(self,p):
            raw=await super().judge_components(p);raw['grounding'][0].pop('claim');return raw
    r=asyncio.run(module().score_episode(question='healthy?',rubric={'mode':'mixed','numeric':[{'id':'n1','expected':'0.96'}],'semantic':[{'id':'s1','expected':'No'}]},submission={'answer_text':'Yes','citations':[]},receipts={},F=1,judge=Bad()))
    assert r['unresolved'] and r['reward'] is None


def test_rate_limit_failure_keeps_http_status():
    from urllib.error import HTTPError
    class Limited(Judge):
        async def judge_components(self,p): raise HTTPError('https://test',429,'limited',{},None)
    r=asyncio.run(module().score_episode(question='Q',rubric={'mode':'mixed','numeric':[{'id':'n1','expected':'1'}],'semantic':[{'id':'s1','expected':'No'}]},submission={'answer_text':'x'},receipts={},F=1,judge=Limited()))
    assert r['unresolved'] and r['reason']=='HTTPError:429'


def test_reference_cannot_add_numeric_requirements(monkeypatch):
    m=module()
    judge=m.ComponentJudge(endpoint='https://test',model='test',key='test')
    async def request(prompt,payload):
        if 'reference' not in payload:
            return {'mode':'semantic','numeric':[], 'semantic':[{'id':'s1','expected':'Whether capital intensive'}]}
        return {'mode':'mixed','numeric':[{'id':'n1','expected':'5.1%'}], 'semantic':[{'id':'s1','expected':'No'}]}
    monkeypatch.setattr(judge,'_retry',request)
    with pytest.raises(ValueError,match='question requirements'):
        asyncio.run(judge.make_rubric('Is 3M capital intensive?', 'No; CAPEX/revenue 5.1%.'))


def test_question_requirements_are_preserved_with_reference(monkeypatch):
    m=module()
    judge=m.ComponentJudge(endpoint='https://test',model='test',key='test')
    async def request(prompt,payload):
        if 'reference' not in payload:
            return {'mode':'semantic','numeric':[], 'semantic':[{'id':'s1','expected':'Whether capital intensive'}]}
        return {'mode':'semantic','numeric':[], 'semantic':[{'id':'s1','expected':'No'}]}
    monkeypatch.setattr(judge,'_retry',request)
    rubric=asyncio.run(judge.make_rubric('Is 3M capital intensive?', 'No; CAPEX/revenue 5.1%.'))
    assert rubric['requirements']['numeric']==[]
    assert rubric['requirements']['semantic'][0]['expected']=='Whether capital intensive'
    assert rubric['reference']=='No; CAPEX/revenue 5.1%.'
