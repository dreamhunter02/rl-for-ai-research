"""Fast FinanceBench eval using NVIDIA Inference Hub Nemotron."""
from __future__ import annotations
import json,time
from pathlib import Path
import financebench_harness as hb
from generate_teacher_traces import _rows
from generate_targeted_traces import run_hub

idx=hb.build_index(); rows=_rows("eval")
out=Path("results/local_eval/nemotron35_lightning_eval_10turn.jsonl"); out.parent.mkdir(parents=True,exist_ok=True)
model="nvidia/nvidia/nemotron-3.5-lightning"
for i,row in enumerate(rows,1):
 t=time.time()
 try:
  rec=run_hub(idx,row,model,max_turns=10)
  reward=hb.reward(row["answer"][0],rec.get("answer_text",""))
  r={"financebench_id":row["financebench_id"],"question":row["question"],"gold":row["answer"][0],"model":model,"answer_text":rec.get("answer_text",""),"tool_calls":rec.get("tool_calls",[]),"trace":rec.get("trace",[]),"deterministic_reward":reward,"wall_s":round(time.time()-t,2)}
 except Exception as exc:
  r={"financebench_id":row["financebench_id"],"question":row["question"],"gold":row["answer"][0],"model":model,"error":repr(exc),"wall_s":round(time.time()-t,2)}
 with out.open("a") as f:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 print(f"[{i}/{len(rows)}] {row['financebench_id']} reward={r.get('deterministic_reward')} tools={len(r.get('tool_calls',[]))} wall={r['wall_s']}s",flush=True)
rs=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
v=[r for r in rs if "deterministic_reward" in r]
print(json.dumps({"n":len(rs),"valid":len(v),"exact_or_numeric":sum(r["deterministic_reward"]>=.9 for r in v),"mean_reward":sum(r["deterministic_reward"] for r in v)/max(1,len(v)),"out":str(out)}))
