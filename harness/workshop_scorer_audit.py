"""Generate scalar scorer regressions from an immutable original git commit."""
import argparse
import subprocess
import types
from pathlib import Path
from workshop_reward import numeric_equal
from workshop_report import write_table

CASES=[
    ('percent-point error','3.2%','4.1%',False),('sign flip','-5.0%','5.0%',False),
    ('small decimal error','0.05','0.06',False),('accounting negative','$(1,234) million','$1,234 million',False),
    ('number dump','$1577.00','1,200, 1,577 and 1,890',False),
    ('plural scale','$1577 million','$1,577 millions',True),
    ('scale mismatch','4.2 million','4.2 billion',False),('currency mismatch','USD 100','EUR 100',False),
    ('zero percent','0.00%','0.4%',False),('amount in year range','$2000','$2001',False),
    ('rounding interior','1.2','1.249',True),('rounding tie half-up','1.2','1.25',False),
    ('equivalent scale','4.2 billion','4200 million',True),
    ('verbose wrong answer','3.2%','The requested margin was 4.1%, reflecting operating performance, financing, expenses and other effects over the period under review.',False),
    ('empty finish','3.2%','',False),
]


def audit(out, legacy_ref='361d3f2337ed55576c241866353e4a9b72a9e9b6'):
    source=subprocess.check_output(['git','show',legacy_ref+':harness/financebench_harness.py'],text=True)
    old=types.ModuleType('legacy_scorer')
    old.__file__=str(Path(__file__).with_name('financebench_harness.py'))
    exec(compile(source,'<pinned legacy scorer>','exec'),old.__dict__)
    rows=[]
    for name,gold,candidate,expected in CASES:
        new=float(numeric_equal(gold,candidate))
        rows.append(dict(case=name,gold=gold,candidate=candidate,expected=int(expected),
            rationale='One scalar; exact unit/sign/scale; round-half-up at gold precision',
            old_score=old.score_answer(gold,candidate)[0],new_score=new,passed=new==float(expected)))
    Path(out).mkdir(parents=True,exist_ok=True)
    write_table(Path(out)/'table3_scorer',rows)
    if not all(r['passed'] for r in rows): raise RuntimeError('Scorer regressions failed')
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True);ap.add_argument('--legacy-ref',default='361d3f2337ed55576c241866353e4a9b72a9e9b6')
    args=ap.parse_args();audit(args.out,args.legacy_ref)

if __name__=='__main__':main()
