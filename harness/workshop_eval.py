"""Matched B1/R1 evaluation through exactly the training environment and renderer."""
import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def json_default(value):
    if hasattr(value, 'model_dump'): return value.model_dump()
    if hasattr(value, '__dict__'): return vars(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def terminal_reason(stop, state):
    if stop == 'tool_stopped' and state.accepted: return 'accepted_finish'
    mapping = {'completed': 'plain_text', 'parse_error': 'parse', 'max_tokens': 'length',
               'max_sampled_tokens': 'length', 'context_overflow': 'context', 'max_turns': 'turn_limit'}
    if state.validation_errors and stop in ('completed', 'max_turns', 'tool_stopped'): return 'validation'
    return mapping.get(str(stop), 'other')


async def evaluate(args):
    import tinker
    import finance_env as fe
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.rollouts import do_single_rollout
    from train_financebench import load_tinker_key
    from workshop_prepare import preflight
    from workshop_reward import SCORER_VERSION
    if not os.environ.get('FINANCEBENCH_TARGETS'): raise ValueError('Set FINANCEBENCH_TARGETS')
    fingerprint = preflight(os.environ.get('FINANCEBENCH_SPLIT', str(fe.hb.BASE/'split.json')), os.environ['FINANCEBENCH_TARGETS'], corpus=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prediction_path=out/'predictions.jsonl'
    if prediction_path.exists(): raise FileExistsError('Use a new run directory; evaluation is append-only within a run')
    rows = fe.load_financebench(args.split)
    index = await fe.Bm25Tool.build(doc_names=sorted({row['doc'] for row in rows}))
    key = os.environ.get('TINKER_API_KEY') or load_tinker_key()
    client = tinker.ServiceClient(api_key=key)
    sampler = client.create_sampling_client(model_path=args.checkpoint, base_model=None if args.checkpoint else args.model)
    fields = getattr(tinker.SamplingParams, 'model_fields', {})
    if args.sampling_seed is not None and 'seed' not in fields:
        raise ValueError('Installed SDK has no SamplingParams.seed; omit --sampling-seed and disclose unseeded sampling')
    manifest = {**vars(args), **fingerprint, 'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'question_ids': [r['financebench_id'] for r in rows], 'scorer_version': 'workshop-v1',
                'max_trajectory_tokens': int(os.environ.get('MAX_TRAJECTORY_TOKENS','32768'))}
    fingerprint_payload = {**fingerprint, 'renderer': args.renderer, 'max_turns': args.max_turns, 'max_tokens': args.max_tokens, 'temperature': args.temperature, 'max_trajectory_tokens': manifest['max_trajectory_tokens'], 'judge': fe.RewardConfig.from_env().summary(), 'source_hashes': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('finance_env.py','financebench_harness.py','workshop_reward.py','observations.py','reward_calculation.py')}}
    evaluation_signature = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    manifest['evaluation_signature'] = evaluation_signature
    manifest['evaluator_fingerprint'] = fingerprint_payload
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    for row in rows:
        builder = fe.FinanceSearchEnvGroupBuilder(row,args.model,args.renderer,args.max_turns,1,index,
            max_trajectory_tokens=manifest['max_trajectory_tokens'], judge=fe.build_judge(fe.RewardConfig.from_env()))
        env=(await builder.make_envs())[0]
        state=env.tool_obj.state
        call_counter=0
        async def policy(model_input, stop, *, max_tokens=None):
            nonlocal call_counter
            params=dict(stop=stop,max_tokens=min(args.max_tokens,max_tokens or args.max_tokens),temperature=args.temperature)
            if args.sampling_seed is not None:
                key=f'{args.sampling_seed}:{row["financebench_id"]}:{call_counter}'
                params['seed']=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2147483647
            call_counter+=1
            response=await sampler.sample_async(prompt=model_input,num_samples=1,sampling_params=tinker.SamplingParams(**params))
            seq=response.sequences[0]
            return TokensWithLogprobs(tokens=seq.tokens,maybe_logprobs=seq.logprobs,stop_reason=seq.stop_reason)
        started=time.monotonic()
        record={'condition': args.condition,'run_id':args.run_id,'checkpoint':args.checkpoint or args.model,
            'question_id':row['financebench_id'],'document_id':row['doc'],'split':args.split,
            'training_seed':args.training_seed,'sampling_seed':args.sampling_seed,'derived':bool(row['target'].get('derived')),
            'evaluation_signature':evaluation_signature,'F':0,'A':0,'G':0,'Ret':0,'correct':0,'grounded_success':0,'reward':0,'unresolved':False,'grader_version':SCORER_VERSION}
        try:
            trajectory=await do_single_rollout(policy,env)
            record.update(state.last_score)
            record.update(stop_reason=terminal_reason(trajectory.stop_reason,state),
                output_tokens=sum(len(t.ac.tokens) for t in trajectory.transitions),
                total_reward=sum(t.reward for t in trajectory.transitions),turns=len(trajectory.transitions))
        except Exception as exc:
            record.update(stop_reason='infrastructure',error=type(exc).__name__,output_tokens=None,turns=state.turn)
        citations=(state.accepted or {}).get('citations',[])
        observations=state.observations
        gold_pages={(e.get('evidence_doc_name',row['doc']),int(e['evidence_page_num'])+1) for e in row.get('evidence',[]) if 'evidence_page_num' in e}
        receipt_pages={(r.get('document_id'),r.get('page')) for r in state.receipts.values()}
        record.update(submission=state.accepted,latency_s=time.monotonic()-started,
            tool_calls=len(observations),calculator_used=bool(state.calculations),
            gold_page_read=bool(gold_pages & receipt_pages),citation_count=len(citations),
            visible_citations=sum(c.get('receipt_id') in state.receipts and state.receipts[c['receipt_id']].get('document_id')==c.get('document_id') and state.receipts[c['receipt_id']].get('page')==c.get('page') for c in citations),
            raw_tool_chars=[o['payload'].get('observation',{}).get('raw_chars',0) for o in observations],
            visible_tool_chars=[o['visible_chars'] for o in observations],
            truncated_observations=sum(bool(o['payload'].get('observation',{}).get('truncated')) for o in observations),
            cost_usd=None)
        trace_path=out/'rollouts'/f'{row["financebench_id"]}.json'
        trace_path.parent.mkdir(exist_ok=True)
        trace_path.write_text(json.dumps({'history':env.env.message_env._history if hasattr(env.env.message_env,'_history') else env.env.message_env.history,
            'receipts':state.receipts,'calculations':state.calculations,'observations':observations,'score':state.last_score},default=json_default,indent=2))
        record['trace_path']=str(trace_path)
        with prediction_path.open('a') as f: f.write(json.dumps(record,default=json_default)+'\n')
        print(json.dumps({k:record.get(k) for k in ('question_id','correct','grounded_success','stop_reason','unresolved')}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition',choices=['B1','R1','R2'],required=True); ap.add_argument('--run-id',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--split',choices=['dev','eval'],default='eval')
    ap.add_argument('--model',default='nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16')
    ap.add_argument('--renderer',default='nemotron3_ultra'); ap.add_argument('--checkpoint')
    ap.add_argument('--training-seed',type=int); ap.add_argument('--sampling-seed',type=int)
    ap.add_argument('--max-turns',type=int,default=8); ap.add_argument('--max-tokens',type=int,default=1024)
    ap.add_argument('--temperature',type=float,default=0.2)
    args=ap.parse_args()
    if args.condition=='B1' and args.checkpoint: ap.error('B1 must use base weights')
    if args.condition!='B1' and not args.checkpoint: ap.error('RL conditions require a checkpoint')
    asyncio.run(evaluate(args))

if __name__=='__main__': main()
