"""Local vLLM baseline on the untouched FinanceBench eval split."""
from __future__ import annotations
import json, time
from pathlib import Path
import financebench_harness as hb
from generate_teacher_traces import _rows, run_teacher
from eval_agent import SYSTEM

INDEX=hb.build_index()
rows=_rows("eval")
out=Path("results/local_eval/qwen35_9b_base_eval.jsonl")
out.parent.mkdir(parents=True,exist_ok=True)
for i,row in enumerate(rows,1):
    started=time.time()
    try:
        rec=run_teacher(INDEX,row,"qwen3.5-9b",6,base_url="http://spark-a16b.local:18300/v1",system_prompt=SYSTEM,temperature=0.0,max_tokens=512)
        reward=hb.reward(row["answer"][0],rec.get("answer_text",""))
        record={"financebench_id":row["financebench_id"],"question":row["question"],"gold":row["answer"][0],"model":"Qwen/Qwen3.5-9B","answer_text":rec.get("answer_text",""),"tool_calls":rec.get("tool_calls",[]),"trace":rec.get("trace",[]),"deterministic_reward":reward,"wall_s":round(time.time()-started,2)}
    except Exception as exc:
        record={"financebench_id":row["financebench_id"],"question":row["question"],"gold":row["answer"][0],"model":"Qwen/Qwen3.5-9B","error":repr(exc)}
    with out.open("a") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
    print(f"[{i}/{len(rows)}] {row['financebench_id']} reward={record.get('deterministic_reward')} tools={len(record.get('tool_calls',[]))}",flush=True)
rows2=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
valid=[r for r in rows2 if "deterministic_reward" in r]
print(json.dumps({"n":len(rows2),"valid":len(valid),"exact":sum(r["deterministic_reward"]>=0.9 for r in valid),"mean":sum(r["deterministic_reward"] for r in valid)/max(1,len(valid)),"out":str(out)}))
