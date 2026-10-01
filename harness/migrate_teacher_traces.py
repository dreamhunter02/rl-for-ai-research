"""Conservative, non-destructive calculator-free migration with source lineage."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from component_reward import ComponentJudge, score_episode
from rescore_traces import reconstruct
from workshop_reward import validate_submission


def provider_failure(score):
    return bool(score.get('unresolved') and str(score.get('reason','')).split(':')[0] in ('HTTPError','TimeoutError','URLError'))


def shortlist(rows):
    from curate_sft_traces import normalized_model
    selected={}
    for row in sorted(rows,key=lambda r:len(r.get('tool_calls',[]))):
        model=normalized_model(row.get('teacher_model',row.get('model','unknown')))
        if model=='unknown':
            stem=Path(row.get('migration',{}).get('source',{}).get('source','')).name.split('.')[0]
            from curate_sft_traces import MODEL_ALIASES
            if stem in MODEL_ALIASES: model=MODEL_ALIASES[stem]
        key=(row['financebench_id'],model)
        if key not in selected:
            row=copy.deepcopy(row); row['teacher_model']=model
            selected[key]=row
    return list(selected.values())


def migrate_record(record):
    row=copy.deepcopy(record); edits=[]; reasons=[]
    trace=row.get('trace')
    if trace is None and row.get('messages'):
        trace=[]
        for m in row['messages']:
            if m.get('role') not in ('assistant','tool'): continue
            event=copy.deepcopy(m)
            if m['role']=='assistant':
                event['tool_calls']=[dict(name=c['function']['name'],arguments=c['function']['arguments'],call_id=c['id']) for c in m.get('tool_calls',[])]
            else: event['call_id']=event.pop('tool_call_id','')
            trace.append(event)
        row['question']=next((m['content'] for m in row['messages'] if m.get('role')=='user'),'')
        edits.append('converted_chat_messages_to_trace')
    trace=trace or []; calls={}; removed=set(); observations={}; cleaned=[]
    for event in trace:
        for c in event.get('tool_calls',[]):
            cid=c.get('call_id'); args=c.get('arguments',{})
            if isinstance(args,str):
                try: c['arguments']=json.loads(args)
                except ValueError: reasons.append('malformed_arguments')
            if not cid or cid in calls: reasons.append('missing_or_duplicate_call_id')
            calls[cid]=c
            if c.get('name')=='calculate': removed.add(cid)
        if event.get('role')=='tool':
            cid=event.get('call_id')
            if cid not in calls or cid in observations: reasons.append('orphan_or_duplicate_observation')
            observations[cid]=event
    if set(calls)!=set(observations): reasons.append('unpaired_calls')
    for cid in removed:
        try: content=json.loads(observations[cid]['content'])
        except (KeyError,ValueError,TypeError): content={}
        if not isinstance(content,dict) or not content.get('error'):
            reasons.append('successful_or_unknown_calculator_dependency')
    accepted=None; finish_count=0
    for event in trace:
        if event.get('role')=='assistant':
            original=event.get('tool_calls',[])
            kept=[c for c in original if c.get('call_id') not in removed]
            if len(kept)!=len(original): edits.append('removed_calculator_calls')
            event['tool_calls']=kept
            for c in kept:
                if c.get('parse_error'): reasons.append('parse_error')
                if c.get('name')=='finish':
                    finish_count+=1
                    if isinstance(c.get('arguments'),dict) and 'calc_id' in c['arguments']:
                        c['arguments'].pop('calc_id'); edits.append('removed_obsolete_calc_id')
            if not kept and not event.get('content'): continue
            if removed and event.get('content') and 'calc_' in event['content']: reasons.append('text_references_removed_calculation')
        elif event.get('role')=='tool':
            cid=event.get('call_id')
            if cid in removed: edits.append('removed_calculator_observation'); continue
            try: payload=json.loads(event.get('content',''))
            except (ValueError,TypeError): reasons.append('invalid_tool_json'); cleaned.append(event); continue
            if not isinstance(payload,dict): reasons.append('invalid_tool_payload'); continue
            if payload.get('error'): reasons.append('residual_tool_error')
            if payload.get('finish') is True and calls.get(cid,{}).get('name')=='finish':
                accepted=copy.deepcopy(payload.get('accepted'))
                if isinstance(accepted,dict):
                    accepted.pop('calc_id',None); payload['accepted']=accepted
            for k in ('turns_remaining','next_action'):
                if k in payload: payload.pop(k); edits.append('removed_stale_turn_metadata')
            event['content']=json.dumps(payload,ensure_ascii=False)
        cleaned.append(event)
    flattened=[c for e in cleaned for c in e.get('tool_calls',[])]
    if finish_count!=1 or not accepted: reasons.append('missing_accepted_finish')
    elif not flattened or flattened[-1].get('name')!='finish': reasons.append('actions_after_finish')
    else:
        try:
            accepted=validate_submission(accepted)
            submitted=validate_submission(flattened[-1]['arguments'])
            fields=('answer_type','value','unit','scale','decision','answer_text','citations','derivation')
            canonical=lambda s:{k:s.get(k,[] if k=='citations' else '') for k in fields}
            if canonical(submitted)!=canonical(accepted): reasons.append('finish_echo_mismatch')
        except ValueError: reasons.append('invalid_finish')
    if accepted and 'calc_' in json.dumps(accepted): reasons.append('answer_references_removed_calculation')
    row.update(trace=cleaned,tool_calls=flattened,submission=accepted,termination_reason='finish' if accepted else 'unresolved',
               score=None,migration={'version':'calculator-free-v1','edited':bool(edits),'native_rollout':False,'edits':sorted(set(edits))})
    row.pop('messages',None);row.pop('tools',None)
    return dict(status='quarantined' if reasons else 'candidate',record=row,edits=sorted(set(edits)),reasons=sorted(set(reasons)))


def verify_sources(receipts):
    import financebench_harness as hb
    import re
    pages={}; norm=lambda s:re.sub(r'\s+',' ',s).strip()
    for rid,r in receipts.items():
        doc=r['document_id']
        if doc not in pages: pages[doc]={p['page']:p['text'] for p in hb.load_pages(doc)}
        text=pages[doc].get(r['page'],'')
        if not text or norm(r['text']) not in norm(text): return False
    return bool(receipts)


async def run(args):
    root=Path(args.root); out=Path(args.out)
    if out.exists(): raise ValueError('Output directory exists; choose a new migration version')
    out.mkdir(parents=True)
    split=json.loads(Path(args.split).read_text()); train={r['financebench_id']:r for r in split['train']}
    targets=json.loads(Path(args.targets).read_text()) if args.targets else {}
    files=sorted(root.rglob('*.jsonl'))+[Path(p) for p in args.extra]
    seen={}; counts=Counter(); inventory=[]; candidates=[]
    with (out/'lineage.jsonl').open('w') as lineage,(out/'quarantine.jsonl').open('w') as rejected,(out/'candidates.jsonl').open('w') as candidate_file:
        for path in files:
            data=path.read_bytes(); inventory.append(dict(source=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
            for n,line in enumerate(data.decode().splitlines(),1):
                if not line.strip():continue
                counts['input_rows']+=1
                try: row=json.loads(line)
                except ValueError:
                    counts['invalid_json']+=1;lineage.write(json.dumps(dict(source=str(path),line=n,status='invalid_json'))+'\n');continue
                fid=row.get('financebench_id'); identity=hashlib.sha256(json.dumps([fid,row.get('teacher_model',row.get('model')),row.get('trace',row.get('messages'))],sort_keys=True).encode()).hexdigest()
                entry=dict(source=str(path),line=n,financebench_id=fid,identity=identity)
                if fid not in train:
                    counts['outside_train']+=1;entry['status']='outside_train'
                elif identity in seen:
                    counts['duplicates']+=1;entry.update(status='duplicate',duplicate_of=seen[identity])
                else:
                    seen[identity]=dict(source=str(path),line=n)
                    result=migrate_record(row); result['record']['gold']=train[fid]['answer']
                    if targets.get(fid,{}).get('adjudication_status')=='unresolved':
                        result['status']='quarantined'; result['reasons'].append('target_unresolved')
                    result['record']['question']=train[fid]['question']
                    result['record']['migration']['source_identity']=identity
                    result['record']['migration']['source']=dict(source=str(path),line=n)
                    entry.update(status=result['status'],reasons=result['reasons']); counts[result['status']]+=1
                    if result['status']=='candidate':
                        candidates.append(result['record']);candidate_file.write(json.dumps(result['record'],ensure_ascii=False)+'\n')
                    else: rejected.write(json.dumps(result,ensure_ascii=False)+'\n')
                lineage.write(json.dumps(entry)+'\n')
    manifest=dict(counts=dict(counts),inventory=inventory,candidate_questions=len({r['financebench_id'] for r in candidates}),judged=0,clean=0)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='inventory'}),flush=True)
    if not args.judge: return
    all_candidate_count=len(candidates)
    candidates=shortlist(candidates)
    manifest['shortlisted_for_judging']=len(candidates)
    manifest['unselected_candidates_retained']=all_candidate_count-len(candidates)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    judge=ComponentJudge(); semaphore=asyncio.Semaphore(args.concurrency); rubrics={}; locks={fid:asyncio.Lock() for fid in train}
    fallback=ComponentJudge(model=args.fallback_model) if args.fallback_model else None
    async def grade(row):
        async with semaphore:
            fid=row['financebench_id']
            try:
                receipts,metrics=reconstruct(row['trace'])
                if not verify_sources(receipts): raise ValueError('source_replay_failed')
                async with locks[fid]:
                    if fid not in rubrics: rubrics[fid]=await judge.make_rubric(row['question'],row['gold'])
                score=await score_episode(question=row['question'],rubric=rubrics[fid],submission=row['submission'],receipts=receipts,F=1,judge=judge)
            except Exception as exc:
                from component_reward import unresolved
                score=unresolved(str(exc)[:160] if isinstance(exc,ValueError) else type(exc).__name__,1)
            if fallback and provider_failure(score):
                try:
                    async with locks[fid]:
                        if fid not in rubrics: rubrics[fid]=await fallback.make_rubric(row['question'],row['gold'])
                    score=await score_episode(question=row['question'],rubric=rubrics[fid],submission=row['submission'],receipts=receipts,F=1,judge=fallback)
                    score['fallback_from']=judge.model
                except Exception as exc:
                    from component_reward import unresolved
                    score=unresolved(type(exc).__name__,1)
            row['score']=score
            with (out/'judged.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            manifest['judged']+=1
            if score.get('grounded_success')==1:
                with (out/'clean.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                manifest['clean']+=1
            if manifest['judged']%20==0: print(json.dumps(dict(judged=manifest['judged'],total=len(candidates),clean=manifest['clean'])),flush=True)
    await asyncio.gather(*(grade(row) for row in candidates))
    manifest['judge_model']=judge.model
    manifest['fallback_model']=args.fallback_model or None
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(dict(judged=manifest['judged'],clean=manifest['clean'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--extra',action='append',default=[])
    p.add_argument('--out',required=True);p.add_argument('--split',required=True);p.add_argument('--judge',action='store_true');p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--targets',default='',help='Frozen target adjudications; unresolved targets are quarantined')
    p.add_argument('--fallback-model',default='',help='Explicit alternate judge on provider errors only; recorded per judgment')
    asyncio.run(run(p.parse_args()))
