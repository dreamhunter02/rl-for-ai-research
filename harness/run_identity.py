"""Secret-free identities shared by evaluation and teacher-generation resumption."""
import hashlib
import json
import os
from pathlib import Path


def require_current_judge(judge):
    if os.environ.get('FINANCEBENCH_SCORER','components')!='legacy' and not hasattr(judge,'judge_components'):
        raise ValueError('Component judge must be configured before generating paid rollouts; set COMPONENT_JUDGE_*')


def judging_identity():
    from component_reward import SCORER_VERSION, PROMPT_VERSION
    return dict(scorer_mode=os.environ.get('FINANCEBENCH_SCORER','components'),
                scorer_version=SCORER_VERSION,prompt_version=PROMPT_VERSION,
                judge_endpoint=os.environ.get('COMPONENT_JUDGE_ENDPOINT',''),
                judge_model=os.environ.get('COMPONENT_JUDGE_MODEL',''),
                legacy_judge_backend=os.environ.get('JUDGE_BACKEND','none'),
                legacy_judge_model=os.environ.get('JUDGE_MODEL',''),
                retrieval_weight=os.environ.get('RETRIEVAL_WEIGHT','0'))


def verify_run_config(out, config):
    out=Path(out);path=out.with_suffix(out.suffix+'.config.json')
    identity=dict(config=config,judging=judging_identity())
    if path.exists():
        if json.loads(path.read_text())!=identity: raise ValueError('Resume configuration differs; use a new output file')
    elif out.exists() and out.stat().st_size:
        raise ValueError('Existing run lacks configuration fingerprint; use a new output file')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(identity,sort_keys=True,indent=2))
