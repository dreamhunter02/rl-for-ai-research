"""Plot dev evaluations at explicit actual-update counts, including update-zero base."""
import argparse
import csv
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--csv',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.csv).open()))
    required={'run_id','training_seed','actual_optimizer_updates','dev_correct_pct','dev_grounded_pct','checkpoint'}
    if not rows or not required.issubset(rows[0]): raise ValueError('CSV requires: '+','.join(sorted(required)))
    if not any(int(r['actual_optimizer_updates'])==0 for r in rows): raise ValueError('Include the measured base checkpoint at update zero')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),sharex=True)
    for run in sorted({r['run_id'] for r in rows}):
        points=sorted([r for r in rows if r['run_id']==run],key=lambda r:int(r['actual_optimizer_updates']))
        for ax,metric,label in zip(axes,('dev_correct_pct','dev_grounded_pct'),('Correct and finished (%)','Grounded success (%)')):
            ax.plot([int(r['actual_optimizer_updates']) for r in points],[float(r[metric]) for r in points],marker='o',label=run)
            ax.set_xlabel('Actual optimizer updates');ax.set_ylabel(label);ax.set_ylim(0,100);ax.legend(fontsize=7)
    fig.tight_layout();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'): fig.savefig(out/('figure1_learning.'+ext),dpi=180)

if __name__=='__main__':main()
