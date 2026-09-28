"""Generate paper tables from measured JSONL records; reject missing/duplicate questions."""
import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

STOP_REASONS = ('accepted_finish','plain_text','parse','validation','length','context','turn_limit','infrastructure','other')


def mean(values):
    return statistics.mean(values) if values else None


def quantile(values, q):
    if not values: return None
    values=sorted(values)
    return values[min(len(values)-1, int((len(values)-1)*q))]


def paired_interval(a,b,seed=0,resamples=10000,clusters=None):
    ids=sorted(a)
    if set(ids)!=set(b): raise ValueError('Paired conditions must contain identical question IDs')
    if not ids: raise ValueError('Cannot bootstrap an empty evaluation')
    differences={qid:b[qid]-a[qid] for qid in ids}
    units=defaultdict(list)
    for qid in ids: units[clusters[qid] if clusters else qid].append(qid)
    keys=sorted(units)
    rng=random.Random(seed)
    draws=[]
    for _ in range(resamples):
        sample=[differences[qid] for _ in keys for qid in units[rng.choice(keys)]]
        draws.append(statistics.mean(sample))
    return statistics.mean(differences.values()), quantile(draws,.025), quantile(draws,.975)


def validate_records(records, expected_ids):
    groups=defaultdict(list)
    seen=set()
    for row in records:
        for key in ('run_id','question_id','condition','F','A','G','correct','grounded_success','stop_reason','unresolved','evaluation_signature'):
            if key not in row: raise ValueError(f'Missing {key} in prediction')
        identity=(row['run_id'],row.get('sampling_seed'),row['question_id'])
        if identity in seen: raise ValueError(f'Duplicate evaluation row: {identity}')
        seen.add(identity)
        if row['stop_reason'] not in STOP_REASONS: raise ValueError('Unknown stop reason')
        for field in ('F','A','G','correct','grounded_success'):
            if not 0 <= row[field] <= 1: raise ValueError(f'Invalid {field}')
        if row['correct'] and not row['F']: raise ValueError('Correct-and-finished requires F')
        if row['grounded_success'] and not row['correct']: raise ValueError('Grounded success requires correctness')
        groups[row['run_id']].append(row)
    if len({r['evaluation_signature'] for r in records}) != 1:
        raise ValueError('Conditions use different frozen evaluator/harness settings')
    for name, rows in groups.items():
        for seed in {r.get('sampling_seed') for r in rows}:
            ids={r['question_id'] for r in rows if r.get('sampling_seed')==seed}
            if ids!=set(expected_ids): raise ValueError(f'{name}: missing/extra scheduled evaluation IDs')
    return groups


def write_table(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def display(v):
        if v is None: return 'unmeasured'
        if isinstance(v,float): return f'{v:.4f}'
        return str(v).replace('|','\\|').replace('\n',' ')
    path.with_suffix('.md').write_text('| '+' | '.join(fields)+' |\n| '+' | '.join('---' for _ in fields)+' |\n'+''.join('| '+' | '.join(display(row.get(k)) for k in fields)+' |\n' for row in rows))


def build_report(records, expected_ids, out, resamples=10000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    groups=validate_records(records,expected_ids)
    main, failures, paired=[],[],[]
    for run, rows in groups.items():
        n=len(rows)
        entry={'run_id':run,'condition':rows[0]['condition'],'training_seed':rows[0].get('training_seed'),
            'N_questions':len(expected_ids),'N_rollouts':n,'valid_finish_pct':100*mean([r['F'] for r in rows]),
            'correct_and_finished_pct':100*mean([r['correct'] for r in rows]),
            'grounded_success_pct':100*mean([r['grounded_success'] for r in rows]),
            'mean_A':mean([r['A'] for r in rows]),'mean_G':mean([r['G'] for r in rows]),
            'unresolved':sum(bool(r['unresolved']) for r in rows),'infrastructure_failures':sum(r['stop_reason']=='infrastructure' for r in rows),
            'correct_upper_bound_pct':100*mean([min(1,r['correct']+int(r['unresolved'] and r['F'])) for r in rows]),
            'grounded_upper_bound_pct':100*mean([min(1,r['grounded_success']+int(r['unresolved'] and r['F'] and r['G']==1)) for r in rows])}
        for source,key in [('output_tokens','output_tokens_per_question'),('tool_calls','tool_calls_per_question'),('latency_s','latency_s_per_question')]:
            vals=[r.get(source) for r in rows]
            entry[key]=mean(vals) if all(v is not None for v in vals) else None
        costs=[r.get('cost_usd') for r in rows]
        entry['cost_usd']=sum(costs) if all(c is not None for c in costs) else None
        main.append(entry)
        f={'run_id':run,**{stop:sum(r['stop_reason']==stop for r in rows) for stop in STOP_REASONS}}
        derived=[r for r in rows if r.get('derived')]
        f.update(gold_page_read_pct=100*mean([float(r.get('gold_page_read',False)) for r in rows]),
                 derived_calculator_pct=100*mean([float(r.get('calculator_used',False)) for r in derived]) if derived else None,
                 zero_citation_episodes=sum(not r.get('citation_count',0) for r in rows),
                 visible_citation_pct=100*sum(r.get('visible_citations',0) for r in rows)/sum(r.get('citation_count',0) for r in rows) if sum(r.get('citation_count',0) for r in rows) else None,
                 median_turns=statistics.median([r.get('turns',0) for r in rows]))
        for source in ('raw_tool_chars','visible_tool_chars'):
            vals=[v for r in rows for v in r.get(source,[])]
            f['median_'+source]=quantile(vals,.5);f['p95_'+source]=quantile(vals,.95)
        f['truncation_pct']=100*sum(r.get('truncated_observations',0) for r in rows)/max(1,sum(r.get('tool_calls',0) for r in rows))
        failures.append(f)
    bases=[run for run,rows in groups.items() if rows[0]['condition']=='B1']
    for baseline in bases:
        for run, rows in groups.items():
            if rows[0]['condition'] not in ('R1','R2'): continue
            if {r.get('sampling_seed') for r in rows}!={r.get('sampling_seed') for r in groups[baseline]}:
                raise ValueError('Paired arms must use the same declared evaluation sampling seeds')
            for metric in ('correct','grounded_success'):
                def aggregate(rr):
                    return {qid:mean([r[metric] for r in rr if r['question_id']==qid]) for qid in expected_ids}
                a,b=aggregate(groups[baseline]),aggregate(rows)
                delta,lo,hi=paired_interval(a,b,resamples=resamples)
                cluster={r['question_id']:r['document_id'] for r in rows}
                _,clo,chi=paired_interval(a,b,resamples=resamples,clusters=cluster)
                paired.append({'comparison':run+' minus '+baseline,'metric':metric,'N_questions':len(a),
                    'delta_pp':100*delta,'ci95_low_pp':100*lo,'ci95_high_pp':100*hi,
                    'document_cluster_low_pp':100*clo,'document_cluster_high_pp':100*chi,
                    'wins':sum(b[q]>a[q] for q in a),'losses':sum(b[q]<a[q] for q in a),
                    'both_correct':sum(a[q]==b[q]==1 for q in a),'both_wrong':sum(a[q]==b[q]==0 for q in a),
                    'training_seed':rows[0].get('training_seed'),'bootstrap_seed':0,'resamples':resamples})
    write_table(out/'table1_main',main);write_table(out/'table2_failures',failures);write_table(out/'table4_paired',paired)
    status={'table1_main':'complete','table2_failures':'partial: citation semantics and repeated-call audits require trace review',
            'table3_scorer':'run workshop_scorer_audit.py','table4_paired':'complete' if paired else 'blocked: need B1 and R1',
            'figure1_learning':'blocked until dev checkpoint evaluations are supplied','figure2_failures':'run with --figures',
            'B0':'diagnostic historical control requires separate legacy adapter and evidence audit',
            'human_audit':'agent must provide actual review records, no automatic human labels'}
    (out/'completion_report.json').write_text(json.dumps(status,indent=2))
    sections=['# Measured paper results\n','Only supplied, validated predictions are summarized. Unmeasured fields remain explicit.\n']
    for name in ('table1_main','table2_failures','table4_paired'):
        if (out/(name+'.md')).exists(): sections += ['\n## '+name+'\n',(out/(name+'.md')).read_text()]
    sections += ['\n## Limitations\nPreviously inspected evaluation split; small sample; document overlap must be reported from protocol.json; public-data pretraining contamination is unknown. Training seeds and question-level uncertainty measure different sources of variation. These outputs do not establish causality for observational tool-use correlations.\n']
    (out/'paper_results.md').write_text('\n'.join(sections))
    return main,failures,paired


def figures(failures,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,3.8));bottom=[0]*len(failures)
    for reason in STOP_REASONS:
        vals=[100*r[reason]/max(1,sum(r[s] for s in STOP_REASONS)) for r in failures]
        ax.bar([r['run_id'] for r in failures],vals,bottom=bottom,label=reason)
        bottom=[a+b for a,b in zip(bottom,vals)]
    ax.set_ylabel('Scheduled rollouts (%)');ax.legend(fontsize=6,bbox_to_anchor=(1.01,1));fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(Path(out)/('figure2_failures.'+ext),dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--predictions',nargs='+',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True);ap.add_argument('--figures',action='store_true')
    args=ap.parse_args();records=[json.loads(line) for path in args.predictions for line in Path(path).read_text().splitlines() if line.strip()]
    protocol=json.loads(Path(args.protocol).read_text());_,f,_=build_report(records,protocol['question_ids']['eval'],args.out)
    if args.figures: figures(f,args.out)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(args.out).glob('*') if p.is_file() and p.name!='checksums.json'}
    (Path(args.out)/'checksums.json').write_text(json.dumps(hashes,indent=2))

if __name__=='__main__': main()
