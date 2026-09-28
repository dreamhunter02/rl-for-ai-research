"""Load a declared experiment config into the environment-driven trainer."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

FIELDS={'model':'MODEL','renderer':'RENDERER','split':'SPLIT_NAME','steps':'STEPS',
    'epochs':'EPOCHS','batch':'BATCH','group':'GROUP','learning_rate':'LR',
    'max_turns':'MAX_TURNS','max_tokens':'MAX_TOKENS','lora_rank':'LORA_RANK',
    'temperature':'TEMPERATURE','seed':'SEED','eval_every':'EVAL_EVERY','save_every':'SAVE_EVERY',
    'remove_constant_reward_groups':'REMOVE_CONSTANT_REWARD_GROUPS',
    'zero_reward_on_limit':'ZERO_REWARD_ON_LIMIT','require_finish':'REQUIRE_FINISH',
    'judge_backend':'JUDGE_BACKEND','judge_model':'JUDGE_MODEL',
    'judge_confidence_threshold':'JUDGE_CONFIDENCE_THRESHOLD'}


def config_environment(config, lr=None, seed=None):
    config=dict(config)
    if lr is not None:config['learning_rate']=lr
    if seed is not None:config['seed']=seed
    try:rate=float(config['learning_rate'])
    except (TypeError,ValueError,KeyError):raise ValueError('Choose a numeric learning rate from dev pilots using --lr')
    if not math.isfinite(rate) or rate<=0:raise ValueError('Learning rate must be finite and positive')
    if 'seed' not in config:raise ValueError('Choose one training seed with --seed; run each seed separately')
    if config.get('eval_split','dev')!='dev' or config.get('dev_split','dev')!='dev':
        raise ValueError('Checkpoint selection must use dev, never eval')
    result={env: str(config[key]).lower() if isinstance(config[key],bool) else str(config[key])
        for key,env in FIELDS.items() if key in config}
    result['DEV_SPLIT']='dev'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--log-path',required=True)
    parser.add_argument('--lr',type=float);parser.add_argument('--seed',type=int);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    env=config_environment(json.loads(Path(args.config).read_text()),args.lr,args.seed)
    env['LOG_PATH']=args.log_path
    if args.dry_run:
        print(json.dumps(env,indent=2));return
    raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).with_name('train_financebench.py'))],env={**os.environ,**env}))

if __name__=='__main__':main()
