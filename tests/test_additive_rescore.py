import importlib
import pytest


def module():
    assert importlib.util.find_spec('additive_rescore'), 'additive scorer missing'
    return importlib.import_module('additive_rescore')


@pytest.mark.parametrize('E,G,A,B,F,want', [
    (1,1,1,0,1,.85),(1,1,1,1,1,1),(1,1,0,0,1,.4),
    (1,1,1,1,0,.15),(0,0,1,0,1,.5),
])
def test_reward_headroom(E,G,A,B,F,want):
    assert module().reward(E=E,G=G,A=A,B=B,F=F)==pytest.approx(want)


@pytest.mark.parametrize('changes',[{'B':1,'A':0},{'E':float('nan')},{'G':2},{'B':.5}])
def test_invalid_components_rejected(changes):
    with pytest.raises(ValueError):
        module().reward(**dict(dict(E=1,G=1,A=1,B=0,F=1),**changes))


def test_report_has_only_latest_scores_and_escapes_text():
    row={'financebench_id':'q','question':'<script>x</script>','gold':'No','trace':[],
         'answer_text':'No','score':{'reward':.123456}}
    s={'E':1,'G':1,'A':1,'B':0,'F':1,'reward':.85,'unresolved':False}
    page=module().render([row],[row],[s],[s])
    assert '<th>Original</th>' not in page and '.123456' not in page
    assert '<script>x</script>' not in page and '&lt;script&gt;' in page
    assert '0.850' in page and '<th>Latest score</th>' in page


def test_unknown_receipts_cannot_earn_evidence_or_grounding():
    raw={'numeric':[],'semantic':[{'id':'s1','correct':True,'reason':'No'}],
         'retrieval':{'coverage':'full','receipt_ids':['fake'],'reason':'all found'},
         'grounding':[{'claim':'fact','supported':True,'receipt_ids':['fake'],'reason':'supported'}],
         'completeness':[{'id':'b1','correct':True,'reason':'covered'}],
         'material_contradiction':False,'unresolved':False}
    rubric={'mode':'semantic','numeric':[],'semantic':[{'id':'s1','expected':'No'}],
            'gold_claims':[{'id':'b1','expected':'No'}]}
    s=module().assemble(raw,rubric,{}, {},1)
    assert s['E']==0 and s['G']==0 and s['B']==1


def test_missing_gold_fact_loses_only_bonus():
    raw={'numeric':[],'semantic':[{'id':'s1','correct':True,'reason':'No'}],
         'retrieval':{'coverage':'full','receipt_ids':['r1'],'reason':'all found'},
         'grounding':[{'claim':'fact','supported':True,'receipt_ids':['r1'],'reason':'supported'}],
         'completeness':[{'id':'b1','correct':False,'reason':'omitted'}],
         'material_contradiction':False,'unresolved':False}
    rubric={'mode':'semantic','numeric':[],'semantic':[{'id':'s1','expected':'No'}],
            'gold_claims':[{'id':'b1','expected':'5.1%'}]}
    s=module().assemble(raw,rubric,{'r1':{'text':'source'}},{'r1':{'text':'source'}},1)
    assert s['A']==1 and s['B']==0 and s['reward']==pytest.approx(.85)


def test_unresolved_judgment_preserves_audit_payload():
    import asyncio
    class Judge:
        async def _retry(self,prompt,payload):
            return {'unresolved':True,'reason':'reference ambiguity'}
    row={'financebench_id':'q','question':'Q','submission':{'answer_text':'a'},'score':{'F':1},'trace':[]}
    result=asyncio.run(module().score(row,{'mode':'semantic'},Judge()))
    assert result['unresolved'] and result['judgment']['reason']=='reference ambiguity'


def test_fresh_input_paths_are_explicit_and_paired(tmp_path):
    assert hasattr(module(),'input_paths'), 'fresh evaluation inputs unsupported'
    assert module().input_paths(tmp_path,'base.jsonl','sft.jsonl')==[__import__('pathlib').Path('base.jsonl'),__import__('pathlib').Path('sft.jsonl')]
    with pytest.raises(ValueError): module().input_paths(tmp_path,'base.jsonl',None)
