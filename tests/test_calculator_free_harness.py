import asyncio
import json
import pytest


def test_current_tool_names_exclude_calculator():
    from teacher_runtime import TOOL_NAMES
    assert set(TOOL_NAMES)=={'bm25_search','grep_document','search_tables','read','read_table','finish'}


def test_finish_accepts_derivation_without_calculator():
    pytest.importorskip('chz')
    from finance_env import Bm25Tool
    from tinker_cookbook.tool_use.types import ToolInput
    tools=Bm25Tool(None)
    result=asyncio.run(tools.finish.run(ToolInput(arguments={'answer_type':'numeric','value':'0.96','unit':'ratio','scale':'ones','derivation':'(15754-5280)/10936'})))
    assert result.should_stop and tools.state.accepted['derivation']=='(15754-5280)/10936'
    assert 'calc_id' not in tools.finish.to_spec()['parameters']['properties']


def test_grep_marks_clipped_text_and_offsets():
    pytest.importorskip('chz')
    from finance_env import Bm25Tool
    from financebench_harness import StructuredIndex
    from tinker_cookbook.tool_use.types import ToolInput
    tools=Bm25Tool(StructuredIndex([{'document_id':'D','page':1,'text':'revenue '+('1 '*600)}],[],[]))
    r=asyncio.run(tools.grep_document.run(ToolInput(arguments={'document_id':'D','patterns':['revenue']})))
    payload=json.loads(r.messages[0]['content']); hit=payload['matches'][0]
    assert hit['truncated'] and hit['end']==hit['start']+len(hit['text'])
    assert hit['next_start']==hit['end']


def test_invalid_search_scope_returns_actionable_error():
    pytest.importorskip('chz')
    from finance_env import Bm25Tool
    from tinker_cookbook.tool_use.types import ToolInput
    r=asyncio.run(Bm25Tool(None).bm25_search.run(ToolInput(arguments={'query_list':['sales'],'scope':'typo'})))
    assert 'scope' in json.loads(r.messages[0]['content'])['error']


def test_live_teacher_and_rl_component_scores_match():
    pytest.importorskip('chz')
    from finance_env import FinanceAnswerReward
    from teacher_runtime import TeacherHarnessSession
    from financebench_harness import StructuredIndex
    class Judge:
        async def make_rubric(self,*args): return {'mode':'numeric','numeric':[{'id':'n1','expected':'100'}],'semantic':[]}
        async def judge_components(self,p):
            return {'numeric':[{'id':'n1','correct':True,'reason':'100 matches'}],'semantic':[],
                    'grounding':[{'claim':'revenue','supported':True,'receipt_ids':['r1'],'reason':'source'}],'unresolved':False}
    session=TeacherHarnessSession(StructuredIndex([{'document_id':'D','page':1,'text':'Revenue 100'}],[],[]))
    session.execute('read',{'document_id':'D','page':1},'r')
    session.execute('finish',{'answer_type':'numeric','value':'100','unit':'USD','scale':'ones','citations':[{'receipt_id':'r1','document_id':'D','page':1}]},'f')
    target={'answer_type':'numeric','value':'100','unit':'USD','scale':'ones'}
    teacher=session.score(target,question='Revenue?',judge=Judge())
    reward,metrics=asyncio.run(FinanceAnswerReward(['100'],question='Revenue?',episode_state=session.state,target=target,judge=Judge())([]))
    assert reward==teacher['reward']==1 and metrics['N']==1


def test_teacher_preserves_content_and_generation_metadata(monkeypatch):
    pytest.importorskip('chz')
    import generate_teacher_traces as gen
    from financebench_harness import StructuredIndex
    from types import SimpleNamespace as Obj
    call=Obj(id='f',function=Obj(name='finish',arguments='{"answer_type":"text","answer_text":"No","citations":[]}'))
    response=Obj(choices=[Obj(message=Obj(content='Evidence is sufficient.',tool_calls=[call]),finish_reason='tool_calls')],usage=Obj(prompt_tokens=100,completion_tokens=20))
    client=Obj(chat=Obj(completions=Obj(create=lambda **kwargs:response)))
    monkeypatch.setattr(gen,'teacher_client',lambda *args:client)
    result=gen.run_teacher(StructuredIndex([],[],[]),{'question':'Q'},'m',2,base_url='http://test',api_key='none')
    assert result['trace'][0]['content']=='Evidence is sufficient.'
    assert result['generation_metadata'][0]['finish_reason']=='tool_calls'
    assert result['generation_metadata'][0]['usage']['completion_tokens']==20
