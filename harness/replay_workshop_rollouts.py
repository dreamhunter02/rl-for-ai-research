from __future__ import annotations
import argparse, collections, json
from pathlib import Path
from workshop_reward import EpisodeState, score_submission

def jdump(x): return json.dumps(x, ensure_ascii=False, sort_keys=True)
def tool_errors(ep):
    out=[]; calc=[]; judge=[]
    for m in ep.get('history',[]):
        if m.get('role')=='tool':
            text=m.get('content','')
            try: obj=json.loads(text)
            except Exception: obj={}
            err=obj.get('error') if isinstance(obj,dict) else None
            if err:
                out.append(str(err))
                if any(x in str(err).lower() for x in ('calculation','operand','expression','calculate')): calc.append(str(err))
                if any(x in str(err).lower() for x in ('judge','deepinfra','credential')): judge.append(str(err))
    return out,calc,judge
def finish_calls(ep):
    n=0
    for m in ep.get('history',[]):
        for call in m.get('tool_calls',[]) if m.get('role')=='assistant' else []:
            if call.get('function',{}).get('name')=='finish': n+=1
    return n
def finish_type(ep,state):
    if state.accepted: return 'accepted_finish'
    if finish_calls(ep): return 'finish_rejected'
    return {'max_turns':'turn_limit','max_tokens':'length','max_sampled_tokens':'length','parse_error':'parse','context_overflow':'context'}.get(ep.get('stop_reason'),'other')
def category(ep,state,result,calc,errors):
    if state.accepted and result.get('grounded_success'): return 'correct_grounded_finish'
    if calc: return 'calculation_input_error'
    if state.validation_errors or errors: return 'tool_validation_error'
    if not state.accepted: return 'missing_terminal_finish'
    if result.get('unresolved'): return 'judge_or_semantic_unresolved'
    if result.get('fabricated_citation') or result.get('G')==0: return 'unsupported_or_ungrounded_evidence'
    if result.get('A')==0: return 'incorrect_answer'
    return 'other'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rollouts',default='results/paper_2026_rl4llm/provider_smoke_lr1e5_v3/workshop_rollouts');ap.add_argument('--targets',default='artifacts/workshop/targets.reviewed.json');ap.add_argument('--out',default='results/paper_2026_rl4llm/replay_v2');x=ap.parse_args()
    targets=json.loads(Path(x.targets).read_text()); root=Path(x.rollouts); out=Path(x.out);out.mkdir(parents=True,exist_ok=True)
    by=collections.defaultdict(list); all_rows=[]
    for fp in sorted(root.glob('*.json')):
        d=json.loads(fp.read_text()); q=d['question_id']; target=targets[q]
        for idx,ep in enumerate(d.get('episodes',[])):
            state=EpisodeState(**ep['state'])
            result=score_submission(target,state.accepted,state)
            errors,calc,judge=tool_errors(ep)
            # Last-score fields are preserved diagnostically; no credential/value is copied.
            ls=state.last_score or {}
            row={'question_id':q,'trajectory_file':str(fp),'episode_index':idx,'sampled_groups':1,'sampled_trajectories':len(d.get('episodes',[])),'retained_groups':0,'finish_type':finish_type(ep,state),'stop_reason':ep.get('stop_reason'),'reward_revised':result['reward'],'correct_revised':result['correct'],'grounded_success_revised':result['grounded_success'],'A':result['A'],'G':result['G'],'Ret':result['Ret'],'unresolved':result['unresolved'],'fabricated_citation':result['fabricated_citation'],'validation_errors':state.validation_errors,'tool_validation_errors':len(errors),'tool_error_messages':errors[:8],'calculation_input_errors':len(calc),'calculation_error_messages':calc[:8],'judge_calls':ls.get('judge_calls',0),'judge_errors':ls.get('judge_errors',0),'judge_coverage':ls.get('judge_coverage',0),'failure_category':category(ep,state,result,calc,errors)}
            by[q].append(row);all_rows.append(row)
    # The provider trainer's audit gate is group-level: unresolved groups are
    # withheld; otherwise a nonconstant reward vector is retained.
    summaries=[]
    for q,rows in sorted(by.items()):
        rewards=[r['reward_revised'] for r in rows]; unresolved=any(r['unresolved'] for r in rows); retained=int(not unresolved and len(set(rewards))>1)
        for r in rows:r['retained_groups']=retained
        finish=collections.Counter(r['finish_type'] for r in rows); cats=collections.Counter(r['failure_category'] for r in rows); errs=sum(r['tool_validation_errors'] for r in rows); calcs=sum(r['calculation_input_errors'] for r in rows)
        summaries.append({'question_id':q,'sampled_groups':1,'sampled_trajectories':len(rows),'retained_groups':retained,'reward_values_revised':rewards,'reward_min':min(rewards) if rewards else 0,'reward_max':max(rewards) if rewards else 0,'finish_types':dict(finish),'tool_validation_errors':errs,'calculation_input_errors':calcs,'judge_calls':sum(r['judge_calls'] for r in rows),'judge_errors':sum(r['judge_errors'] for r in rows),'judge_coverage':sum(r['judge_coverage'] for r in rows),'accepted_finishes':sum(r['finish_type']=='accepted_finish' for r in rows),'correct_grounded_finishes':sum(r['grounded_success_revised'] for r in rows),'failure_categories':dict(cats)})
    with (out/'trajectory_replay.jsonl').open('w') as f:
        for r in all_rows:f.write(jdump(r)+'\n')
    with (out/'per_question_failure_table.jsonl').open('w') as f:
        for r in summaries:f.write(jdump(r)+'\n')
    totals={'questions':len(summaries),'saved_trajectories':len(all_rows),'sampled_groups':sum(r['sampled_groups'] for r in summaries),'retained_groups':sum(r['retained_groups'] for r in summaries),'accepted_finishes':sum(r['accepted_finishes'] for r in summaries),'correct_grounded_finishes':sum(r['correct_grounded_finishes'] for r in summaries),'tool_validation_errors':sum(r['tool_validation_errors'] for r in summaries),'calculation_input_errors':sum(r['calculation_input_errors'] for r in summaries),'judge_calls':sum(r['judge_calls'] for r in summaries),'judge_errors':sum(r['judge_errors'] for r in summaries),'rubric_version':targets[next(iter(targets))].get('rubric_version')}
    (out/'summary.json').write_text(json.dumps(totals,indent=2)+'\n')
    with (out/'per_question_failure_table.md').open('w') as f:
        f.write('| question | sampled groups | trajectories | retained groups | finish types | tool errors | calc errors | failure categories |\n|---|---:|---:|---:|---|---:|---:|---|\n')
        for r in summaries:f.write(f"| {r['question_id']} | {r['sampled_groups']} | {r['sampled_trajectories']} | {r['retained_groups']} | {jdump(r['finish_types'])} | {r['tool_validation_errors']} | {r['calculation_input_errors']} | {jdump(r['failure_categories'])} |\n")
    print(json.dumps(totals,indent=2))
if __name__=='__main__':main()
