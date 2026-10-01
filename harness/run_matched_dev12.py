"""Durable supervisor: matched base/adapter dev12 inference, then additive report.

Run from a frozen code snapshot, using an existing corpus and external server.
Never starts training, restarts serving, deletes data or accesses eval42.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


def eval_command(args,phase):
    root=Path(args.root);run=Path(args.run_dir)
    return [sys.executable,'-u',str(Path(__file__).with_name('eval_current_harness.py')),
            '--split','dev','--split-file',str(root/'split.json'),
            '--targets',str(root/'data/targets.json'),
            '--base-url',args.base_url,'--model',args.base_model if phase=='base' else args.sft_model,
            '--phase','baseline' if phase=='base' else 'post_sft',
            '--max-turns',str(args.max_turns),'--max-tokens',str(args.max_tokens),
            '--temperature','0','--seed','0','--out',str(run/(phase+'.jsonl'))]


def main():
    p=argparse.ArgumentParser()
    for name in ('root','run-dir','base-url','base-model','sft-model','code-commit','adapter-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--max-turns',type=int,default=8);p.add_argument('--max-tokens',type=int,default=1024)
    args=p.parse_args();run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=False)
    args.run_dir=str(run);args.root=str(Path(args.root).resolve())
    def status(state,**extra):
        data=dict(state=state,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**extra)
        temp=run/'status.tmp';temp.write_text(json.dumps(data,indent=2));temp.replace(run/'status.json')
        print(json.dumps(data),flush=True)
    status('preflight')
    try:
        root=Path(args.root);split=root/'split.json'
        manifest=dict(vars(args),temperature=0,seed=0,calculator=False,
            split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
            judge_model=os.environ.get('COMPONENT_JUDGE_MODEL'),
            judge_endpoint=os.environ.get('COMPONENT_JUDGE_ENDPOINT'),
            final_reward='0.15E + F*(0.20G + 0.45A + 0.15B + 0.05)',
            note='Fresh matched dev12; generation collects v5 diagnostics, final report uses v6 additive reward. No training.')
        host=args.base_url.removesuffix('/v1').rstrip('/')
        for name,url in [('models',args.base_url.rstrip('/')+'/models'),('serving',host+'/get_model_info')]:
            with urllib.request.urlopen(url,timeout=20) as response: manifest[name]=json.load(response)
        (run/'protocol.json').write_text(json.dumps(manifest,indent=2))
        env=dict(os.environ,FINANCEBENCH_ROOT=args.root)
        expected={r['financebench_id'] for r in json.loads(split.read_text())['dev']}
        errors={}
        for phase in ('base','sft'):
            status('generating',phase=phase)
            with (run/(phase+'.log')).open('w') as log:
                subprocess.run(eval_command(args,phase),cwd=args.root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            rows=[json.loads(x) for x in (run/(phase+'.jsonl')).read_text().splitlines() if x.strip()]
            if len(rows)!=12 or {r['financebench_id'] for r in rows}!=expected: raise ValueError(phase+' frozen dev12 mismatch')
            errors[phase]=[r['financebench_id'] for r in rows if r.get('termination_reason')=='error']
        status('scoring',generation_errors=errors)
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('additive_rescore.py')),
             '--root',args.root,'--before',str(run/'base.jsonl'),'--after',str(run/'sft.jsonl'),
             '--out',str(run/'report'),'--concurrency','2']
        with (run/'scoring.log').open('w') as log:
            subprocess.run(cmd,cwd=args.root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((run/'report/manifest.json').read_text())
        if not (run/'report/comparison.html').is_file(): raise ValueError('missing report')
        status('complete',report=str(run/'report/comparison.html'),unresolved=result['unresolved'],generation_errors=errors)
    except Exception as exc:
        status('failed',error=type(exc).__name__+': '+str(exc)[:500]);raise


if __name__=='__main__': main()
