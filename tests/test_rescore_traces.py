import asyncio
import importlib
import json


def module():
    assert importlib.util.find_spec('rescore_traces'), 'rescoring not implemented'
    return importlib.import_module('rescore_traces')


def test_receipts_only_from_delivered_source_observations():
    trace=[{'role':'assistant','tool_calls':[{'name':'read','call_id':'a'},{'name':'calculate','call_id':'b'}]},
           {'role':'tool','call_id':'a','content':json.dumps({'receipt_id':'r1','document_id':'D','page':1,'text':'100'})},
           {'role':'tool','call_id':'b','content':json.dumps({'receipt_id':'fake','text':'999'})}]
    receipts,metrics=module().reconstruct(trace)
    assert list(receipts)==['r1'] and metrics['total_calls']==2


def test_duplicate_calls_are_diagnostic_not_reward():
    trace=[{'role':'assistant','tool_calls':[{'name':'read','call_id':str(i),'arguments':{'page':1}}]} for i in range(2)]
    assert module().reconstruct(trace)[1]['identical_repeat_calls']==1


def test_html_escapes_candidate_content():
    record={'financebench_id':'id','question':'<script>bad</script>','score':{'reward':0},'trace':[]}
    scored={'new_score':{'reward':.5,'N':1,'G':0},'tool_metrics':{}}
    page=module().render_html([record],[record],[scored],[scored])
    assert '<script>bad</script>' not in page and '&lt;script&gt;bad' in page
    assert 'post-hoc' in page.lower() and '0.5' in page


def test_html_distinguishes_nonapplicable_numeric_from_zero():
    record={'financebench_id':'id','question':'Qualitative?', 'score':{},'trace':[]}
    scored={'new_score':{'N':None,'S':1,'reward':.6,'unresolved':False},'tool_metrics':{}}
    assert '<td>N</td><td>—</td><td>N/A</td>' in module().render_html([record],[record],[scored],[scored])


def test_frozen_rubric_rejects_changed_source():
    m=module(); assert hasattr(m,'frozen_rubric'), 'frozen rubric validation missing'
    import pytest
    with pytest.raises(ValueError):m.frozen_rubric({'source_hash':'bad','new_score':{'rubric':{}}},{'question':'changed'})


def test_frozen_rubric_rejects_old_protocol_even_for_same_source():
    import hashlib
    import pytest
    record={'question':'Is it capital intensive?'}
    sidecar={'source_hash':hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),
             'new_score':{'scorer_version':'workshop-rubric-v4-components','rubric':{}}}
    with pytest.raises(ValueError,match='protocol'):
        module().frozen_rubric(sidecar,record)


def test_malformed_source_observation_is_not_silently_dropped():
    import pytest
    with pytest.raises(ValueError,match='source observation'):
        module().reconstruct([{'role':'assistant','tool_calls':[{'name':'read','call_id':'x'}]},
                              {'role':'tool','call_id':'x','content':'{"text":"truncated'}])


def test_second_judge_keeps_primary_score_and_marks_grounding_disagreement():
    m=module(); assert hasattr(m,'merge_review'), 'judge comparison missing'
    primary={'financebench_id':'q','source_hash':'hash','new_score':{'N':1,'S':None,'A':1,'G':.5,'reward':.75,'unresolved':False}}
    secondary={'financebench_id':'q','source_hash':'hash','new_score':{'N':1,'S':None,'A':1,'G':1,'reward':1,'unresolved':False}}
    merged=m.merge_review(primary,secondary)
    assert merged['new_score']['reward']==.75 and merged['judge_comparison']['answer_agreement']
    assert not merged['judge_comparison']['grounding_agreement']
