"""Compatibility draft generator using the authoritative reviewed-target schema.

Output is a question-ID-keyed JSON object; it is never source-approved automatically.
Use workshop_prepare.py to preserve the recorded train/dev/eval manifest.
"""
import argparse
import json
from pathlib import Path
import financebench_harness as hb
from workshop_prepare import draft_target

infer_target = draft_target

def write_targets(split_name, output):
    rows=json.loads((hb.BASE/'split.json').read_text())[split_name]
    targets={r['financebench_id']:draft_target(r) for r in rows}
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(targets,indent=2)+'\n')
    return len(targets)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',default='train');parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps({'rows':write_targets(args.split,args.output)}))
