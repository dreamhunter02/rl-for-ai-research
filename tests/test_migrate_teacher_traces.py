import importlib
import json
from copy import deepcopy


def module():
    assert importlib.util.find_spec('migrate_teacher_traces'), 'migration missing'
    return importlib.import_module('migrate_teacher_traces')


def sample():
    finish={'name':'finish','call_id':'f','arguments':{'answer_type':'text','answer_text':'100','citations':[]}}
    return {'financebench_id':'train1','question':'Revenue?','trace':[
        {'role':'assistant','tool_calls':[{'name':'calculate','call_id':'c','arguments':{}},{'name':'read','call_id':'r','arguments':{'document_id':'D','page':1}}]},
        {'role':'tool','call_id':'c','content':'{"error":"bad operands"}'},
        {'role':'tool','call_id':'r','content':'{"receipt_id":"r1","document_id":"D","page":1,"text":"100"}'},
        {'role':'assistant','tool_calls':[finish]},
        {'role':'tool','call_id':'f','content':json.dumps({'finish':True,'accepted':finish['arguments']})}]}


def test_removes_failed_calculator_preserves_other_call_and_original():
    row=sample(); old=deepcopy(row); result=module().migrate_record(row)
    assert row==old and result['status']=='candidate'
    assert [c['name'] for c in result['record']['tool_calls']]==['read','finish']
    assert len(result['record']['trace'])==4 and result['edits']


def test_successful_calculator_is_quarantined_not_relabelled_reasoning():
    row=sample();row['trace'][1]['content']='{"value":"100","calc_id":"c1"}'
    assert module().migrate_record(row)['status']=='quarantined'


def test_missing_finish_cannot_be_invented():
    row=sample();row['trace']=row['trace'][:3]
    assert 'missing_accepted_finish' in module().migrate_record(row)['reasons']


def test_rejects_orphaned_tool_observation():
    row=sample();row['trace'][2]['call_id']='unknown'
    assert module().migrate_record(row)['status']=='quarantined'


def test_judge_shortlist_normalizes_teacher_aliases():
    m=module(); assert hasattr(m,'shortlist'), 'shortlist missing'
    a={'financebench_id':'q','teacher_model':'gpt56_terra','tool_calls':[{},{}]}
    b={'financebench_id':'q','teacher_model':'openai/openai/gpt-5.6-terra','tool_calls':[{}]}
    selected=m.shortlist([a,b])
    assert len(selected)==1 and len(selected[0]['tool_calls'])==1


def test_finish_arguments_must_match_observed_accepted_answer():
    row=sample(); row['trace'][-2]['tool_calls'][0]['arguments']['answer_text']='WRONG'
    result=module().migrate_record(row)
    assert result['status']=='quarantined' and 'finish_echo_mismatch' in result['reasons']


def test_fallback_only_on_provider_failure():
    m=module(); assert hasattr(m,'provider_failure'), 'provider failure classifier missing'
    assert m.provider_failure({'reason':'HTTPError:429','unresolved':True})
    assert not m.provider_failure({'reason':'judge_unresolved','unresolved':True})
    assert not m.provider_failure({'reward':0,'unresolved':False})
