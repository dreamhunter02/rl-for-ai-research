"""Rescore immutable saved dev trajectories; never re-execute model/tool calls."""
from __future__ import annotations
import argparse
import copy
import asyncio
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
from component_reward import ComponentJudge, SCORER_VERSION, score_episode, cache_key


def reconstruct(trace):
    names={}; receipts={}; seen=set(); repeats=0; errors=0; chars=0; counts=Counter()
    for event in trace:
        if event.get('role')=='assistant':
            for call in event.get('tool_calls',[]):
                fn=call.get('function',call); name=fn.get('name',''); counts[name]+=1
                names[call.get('call_id',call.get('id',''))]=name
                identity=json.dumps([name,fn.get('arguments',{})],sort_keys=True)
                repeats+=identity in seen; seen.add(identity)
        elif event.get('role')=='tool':
            content=event.get('content',''); chars+=len(str(content))
            name=names.get(event.get('call_id',event.get('tool_call_id','')),'')
            try: payload=json.loads(content) if isinstance(content,str) else content
            except (ValueError,TypeError):
                if name in ('read','read_table','grep_document'): raise ValueError('Malformed source observation')
                errors+=1;continue
            if not isinstance(payload,dict):
                if name in ('read','read_table','grep_document'): raise ValueError('Malformed source observation')
                continue
            errors+=bool(payload.get('error'))
            name=names.get(event.get('call_id',event.get('tool_call_id','')),'')
            if name in ('read','read_table'): candidates=[payload]
            elif name=='grep_document': candidates=payload.get('matches',[])
            else: candidates=[]
            for r in candidates:
                if r.get('receipt_id') and r.get('text'):
                    rid=r['receipt_id']
                    if rid in receipts and receipts[rid]!=r: raise ValueError('conflicting receipt ID')
                    receipts[rid]=r
    return receipts,dict(total_calls=sum(counts.values()),by_tool=dict(counts),identical_repeat_calls=repeats,
                         tool_error_observations=errors,delivered_chars=chars,
                         note='Diagnostics only: retries and errors are not automatically model faults; no diversity bonus.')


async def rescore_record(record, *, rubric, judge):
    try: receipts,metrics=reconstruct(record.get('trace',[]))
    except ValueError as exc:
        from component_reward import unresolved
        return dict(financebench_id=record['financebench_id'],model=record.get('model'),
                    source_hash=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),
                    original_score=record.get('score',{}),new_score=unresolved(str(exc),int(record.get('score',{}).get('F',0)==1)),
                    tool_metrics={'input_error':str(exc)})
    submission=record.get('submission')
    if not submission:
        for event in reversed(record.get('trace',[])):
            for call in event.get('tool_calls',[]):
                if call.get('name')=='finish' and isinstance(call.get('arguments'),dict):
                    submission=call['arguments']; break
            if submission: break
    score=await score_episode(question=record['question'],rubric=rubric,submission=submission,
                             receipts=receipts,F=int(record.get('score',{}).get('F',0)==1),judge=judge)
    return dict(financebench_id=record['financebench_id'],model=record.get('model'),
                source_hash=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),
                original_score=record.get('score',{}),new_score=score,tool_metrics=metrics)


def frozen_rubric(sidecar, record):
    identity=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    if sidecar.get('source_hash')!=identity: raise ValueError('Frozen rubric source mismatch')
    if sidecar.get('new_score',{}).get('scorer_version')!=SCORER_VERSION:
        raise ValueError('Frozen rubric protocol mismatch; regenerate question-led requirements')
    return sidecar['new_score']['rubric']


def merge_review(primary, secondary):
    if any(primary.get(k)!=secondary.get(k) for k in ('financebench_id','source_hash')):
        raise ValueError('Independent review source mismatch')
    p=primary['new_score'];s=secondary['new_score'];out=copy.deepcopy(primary)
    out['second_opinion']=s
    out['judge_comparison']=dict(answer_agreement=not p.get('unresolved') and not s.get('unresolved') and all(p.get(k)==s.get(k) for k in ('N','S','A')),
                                 grounding_agreement=p.get('G')==s.get('G'),note='Independent judge check, not a replacement score or proof of correctness.')
    return out


def render_html(before,after,before_scores,after_scores):
    esc=lambda v:html.escape(str(v))
    pretty=lambda v:esc(json.dumps(v,indent=2,ensure_ascii=False))
    out=['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>FinanceBench component reward audit</title><style>body{font:15px system-ui;background:#f4f6f9;color:#192534;margin:24px}article{background:white;padding:20px;margin:20px 0;border-radius:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f6;padding:12px;max-height:600px;overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}summary{cursor:pointer;font-weight:bold;padding:10px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style></head><body><h1>Base vs SFT: component reward audit</h1><p>Post-hoc rescoring of saved trajectories, not a new evaluation or model improvement. Original messages and scores are preserved. Historical baseline lacks a complete settings manifest. No tool-diversity or efficiency bonus.</p>']
    for label,rows in [('Base',before_scores),('SFT',after_scores)]:
        scores=[r['new_score'] for r in rows]
        out.append(f'<p><b>{label}</b>: correct {sum(s.get("correct",0) for s in scores)}/{len(rows)}; grounded {sum(s.get("grounded_success",0) for s in scores)}/{len(rows)}; unresolved {sum(bool(s.get("unresolved")) for s in scores)}.</p>')
    for b,a,bs,ass in zip(before,after,before_scores,after_scores):
        out.append(f'<article><h2>{esc(b["financebench_id"])}</h2><p>{esc(b["question"])}</p><p>Reference: {esc(b.get("gold",""))}</p><div class="pair">')
        for label,r,s in [('Base',b,bs),('SFT',a,ass)]:
            score=s['new_score']
            display=lambda key: 'N/A' if key in ('N','S') and score.get(key) is None and not score.get('unresolved',False) else score.get(key,'—')
            cells=''.join(f'<tr><td>{key}</td><td>{esc(r.get("score",{}).get(key,"—"))}</td><td>{esc(display(key))}</td></tr>' for key in ('F','N','S','A','G','reward','correct','grounded_success','unresolved'))
            out.append(f'<section><h3>{label}</h3><p>{esc(r.get("answer_text",r.get("submission","")))}</p><table><tr><th>Component</th><th>Original</th><th>New</th></tr>{cells}</table><details open><summary>Judge reasons and rubric</summary><pre>{pretty(score)}</pre></details><details><summary>Tool-use diagnostics</summary><pre>{pretty(s["tool_metrics"])}</pre></details>')
            if s.get('second_opinion'):
                out.append(f'<details><summary>Independent judge check · answer agreement: {esc(s["judge_comparison"]["answer_agreement"])} · grounding agreement: {esc(s["judge_comparison"]["grounding_agreement"])}</summary><pre>{pretty(s["second_opinion"])}</pre></details>')
            for i,event in enumerate(r.get('trace',[]),1):
                out.append(f'<details><summary>{i}. {esc(event.get("role",""))}</summary><pre>{pretty(event)}</pre></details>')
            out.append(f'<details><summary>Complete original record</summary><pre>{pretty(r)}</pre></details></section>')
        out.append('</div></article>')
    return ''.join(out)+'</body></html>'


async def run(args):
    groups=[[json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()] for p in (args.before,args.after)]
    expected={r['financebench_id'] for r in json.loads(Path(args.split).read_text())['dev']}
    for rows in groups:
        if len(rows)!=12 or len({r['financebench_id'] for r in rows})!=12 or {r['financebench_id'] for r in rows}!=expected:
            raise ValueError('require exactly 12 unique frozen dev IDs per input')
    groups[1]=[{r['financebench_id']:r for r in groups[1]}[b['financebench_id']] for b in groups[0]]
    if any(b['question']!=a['question'] or b.get('gold')!=a.get('gold') for b,a in zip(*groups)): raise ValueError('reference mismatch')
    if args.merge_secondary:
        out=Path(args.out); scores=[]; summary={}
        for label,records in zip(('base','sft'),groups):
            primary={r['financebench_id']:r for r in map(json.loads,(out/(label+'.jsonl')).read_text().splitlines())}
            secondary={r['financebench_id']:r for r in map(json.loads,(Path(args.merge_secondary)/(label+'.jsonl')).read_text().splitlines())}
            merged=[merge_review(primary[r['financebench_id']],secondary[r['financebench_id']]) for r in records]
            for source,score in zip(records,merged): frozen_rubric(score,source)
            scores.append(merged)
            (out/(label+'.reviewed.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
            summary[label]=dict(n=len(merged),old_mean_reward=sum(r['score'].get('reward',0) or 0 for r in records)/len(records),
                new_mean_reward=sum(r['new_score'].get('reward',0) or 0 for r in merged)/len(merged),
                old_correct=sum(r['score'].get('correct',0) for r in records),new_correct=sum(r['new_score']['correct'] for r in merged),
                old_grounded=sum(r['score'].get('grounded_success',0) for r in records),new_grounded=sum(r['new_score']['grounded_success'] for r in merged),
                answer_agreements=sum(r['judge_comparison']['answer_agreement'] for r in merged),
                grounding_agreements=sum(r['judge_comparison']['grounding_agreement'] for r in merged),
                unresolved=sum(r['new_score']['unresolved'] for r in merged))
        (out/'comparison.html').write_text(render_html(*groups,*scores))
        (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary));return
    judge=ComponentJudge(); limit=asyncio.Semaphore(args.concurrency)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    async def pair(b,a):
        async with limit:
            rubric=(frozen_rubric(json.loads((Path(args.rubric_from)/(b['financebench_id']+'.base.json')).read_text()),b)
                    if args.rubric_from else await judge.make_rubric(b['question'],b.get('gold','')))
            scores=await asyncio.gather(*(rescore_record(r,rubric=rubric,judge=judge) for r in (b,a)))
            for label,score in zip(('base','sft'),scores):
                path=out/(b['financebench_id']+'.'+label+'.json')
                path.write_text(json.dumps(score,indent=2,ensure_ascii=False))
            print(json.dumps({'id':b['financebench_id'],'scores':[s['new_score'].get('reward') for s in scores],'unresolved':[s['new_score']['unresolved'] for s in scores]}),flush=True)
            return scores
    pairs=await asyncio.gather(*(pair(b,a) for b,a in zip(*groups)))
    scores=[[p[i] for p in pairs] for i in (0,1)]
    for label,rows in zip(('base','sft'),scores):
        (out/(label+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (out/'comparison.html').write_text(render_html(*groups,*scores))
    manifest=dict(scorer_version=SCORER_VERSION,judge_model=judge.model,inputs={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.before,args.after)},n=24,unresolved=sum(s['new_score']['unresolved'] for rows in scores for s in rows))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True)
    p.add_argument('--split',default='split.json');p.add_argument('--out',required=True);p.add_argument('--concurrency',type=int,default=3)
    p.add_argument('--rubric-from',default='',help='Previous sidecar directory: reuse the exact candidate-blind rubrics for an independent judge')
    p.add_argument('--merge-secondary',default='',help='Offline: attach independent sidecars to existing --out without replacing primary scores')
    asyncio.run(run(p.parse_args()))
