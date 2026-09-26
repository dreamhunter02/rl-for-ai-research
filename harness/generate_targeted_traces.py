"""Target missing exact-1.0 FinanceBench teacher traces with provider fallback.

Selection is based on the current local trace files. For each training question
without a non-empty deterministic reward==1.0 trace, try GPT first, then the
DeepInfra teachers, and finally NVIDIA Inference Hub. Every candidate is saved;
the best exact candidate is also written to targeted_exact.jsonl.

The Inference Hub helper is used without exposing its credential.
"""
from __future__ import annotations
import argparse, json, re, subprocess, tempfile, time
from pathlib import Path
import financebench_harness as hb
from generate_teacher_traces import run_teacher, _rows

ROOT = hb.BASE


def load_rows(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["financebench_id"]] = r
    return out


def exact_ids() -> set[str]:
    good = set()
    for name in ["gpt56", "qwen38flashnext", "deepseek41", "glm53"]:
        for r in load_rows(ROOT / "results" / "teacher_traces" / f"train_{name}.jsonl").values():
            if r.get("answer_text", "").strip() and float(r.get("reward", 0)) >= 1.0:
                good.add(r["financebench_id"])
    return good


def hub_chat(model: str, messages: list[dict], tools: list[dict], max_tokens: int = 1024) -> dict:
    payload = {"model": model, "messages": messages, "tools": tools, "temperature": 0.0, "max_tokens": max_tokens, "reasoning": {"effort": "medium"}, "stream": False}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f, ensure_ascii=False)
        path = f.name
    try:
        cmd = [str(Path.home() / ".hermes/skills/research/inference-hub/scripts/inference-hub"), "request", "--endpoint", "chat/completions", "--input-file", path]
        out = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
        if out.returncode:
            raise RuntimeError(out.stderr.strip() or out.stdout.strip())
        return json.loads(out.stdout)
    finally:
        Path(path).unlink(missing_ok=True)


def _text_tool_calls(message: dict, start_id: int) -> list[dict]:
    """Parse Qwen-style XML tool calls emitted in reasoning_content by Hub."""
    text = "\n".join(str(message.get(k) or "") for k in ("reasoning_content", "content"))
    parsed = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.S):
        fn_match = re.search(
            r"<function=([A-Za-z0-9_.-]+)>(.*?)(?:</function=\1>|</function>)",
            block,
            flags=re.S,
        )
        if not fn_match:
            continue
        name, body = fn_match.groups()
        args = {}
        for param, raw in re.findall(
            r"<parameter=([A-Za-z0-9_.-]+)>\s*(.*?)\s*</parameter>",
            body,
            flags=re.S,
        ):
            try:
                args[param] = json.loads(raw)
            except json.JSONDecodeError:
                args[param] = raw.strip()
        parsed.append({
            "id": f"hub-text-call-{start_id + len(parsed)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    return parsed


def run_hub(index, row, model: str, max_turns: int) -> dict:
    from eval_agent import tool_specs, execute_local
    import finance_env as fe
    system = ("You are a careful financial research agent. Use typed tools to retrieve filing evidence, then call finish(answer=...) exactly once. Include evidence_document and evidence_page when known, use requested units and precision, and do not continue after finish.")
    specs = tool_specs(fe.Bm25Tool(index))
    tools = [{"type": "function", "function": {"name": s["name"], "description": s.get("description", ""), "parameters": s["parameters"]}} for s in specs]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}]
    calls=[]; trace=[]; final=""; termination_reason="max_turns"
    for _ in range(max_turns):
        response = hub_chat(model, messages, tools, max_tokens=4096)
        choices=response.get("choices") or []
        if not choices: break
        msg=choices[0].get("message") or {}
        tcs=msg.get("tool_calls") or []
        if not tcs:
            tcs = _text_tool_calls(msg, len(calls))
        if not tcs:
            final=msg.get("content") or ""
            if not final:
                answers = re.findall(r"(?im)^\s*Answer:\s*(.+?)\s*$", str(msg.get("reasoning_content") or ""))
                if answers:
                    final = "Answer: " + answers[-1].strip()
            trace.append({"role":"assistant","content":final})
            break
        turn=[]
        finish_answer=""
        assistant={"role":"assistant","content":msg.get("content") or msg.get("reasoning_content") or "","tool_calls":[]}
        for tc in tcs:
            fn=tc.get("function") or {}
            args=json.loads(fn.get("arguments") or "{}")
            cid=tc.get("id", "hub-call")
            call={"name":fn.get("name"),"arguments":args,"call_id":cid}
            calls.append(call); turn.append(call)
            assistant["tool_calls"].append({"id":cid,"type":"function","function":{"name":fn.get("name"),"arguments":fn.get("arguments","{}")}})
        messages.append(assistant)
        for tc in tcs:
            fn=tc.get("function") or {}; cid=tc.get("id", "hub-call")
            args=json.loads(fn.get("arguments") or "{}")
            obs=execute_local(index, fn.get("name"), args)[:4000]
            messages.append({"role":"tool","tool_call_id":cid,"content":obs})
            trace.append({"role":"tool","call_id":cid,"content":obs})
            if fn.get("name") == "finish":
                finish_answer = str(args.get("answer", "")).strip()
        trace.append({"role":"assistant","tool_calls":turn})
        if finish_answer:
            final = f"Answer: {finish_answer}"
            termination_reason = "finish"
            break
    return {"answer_text":final,"tool_calls":calls,"trace":trace,"termination_reason":termination_reason,"openai_response_id":None}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-turns",type=int,default=8)
    ap.add_argument("--out",default="results/teacher_traces/targeted_candidates.jsonl")
    ap.add_argument("--exact-out",default="results/teacher_traces/targeted_exact.jsonl")
    ap.add_argument("--deepinfra-key",default="")
    ap.add_argument("--deepinfra-key-file",default="")
    args=ap.parse_args()
    if not args.deepinfra_key and args.deepinfra_key_file:
        args.deepinfra_key=Path(args.deepinfra_key_file).read_text().strip()
    rows={r["financebench_id"]:r for r in _rows("train")}
    missing=sorted(set(rows)-exact_ids())
    index=hb.build_index()
    out=Path(args.out); exact=Path(args.exact_out)
    out.parent.mkdir(parents=True,exist_ok=True)
    exact.parent.mkdir(parents=True,exist_ok=True)
    existing=load_rows(out); exact_existing=load_rows(exact)
    providers=[
      ("openai","gpt-5.6-terra","", ""),
      ("deepinfra","deepseek-ai/DeepSeek-V4.1-Flash","https://api.deepinfra.com/v1",args.deepinfra_key),
      ("deepinfra","zai-org/GLM-5.3","https://api.deepinfra.com/v1",args.deepinfra_key),
      ("inference-hub","nvidia/qwen/qwen3.8-flash-next","hub", ""),
    ]
    done=0
    for i,fid in enumerate(missing,1):
        if fid in exact_existing: continue
        row=rows[fid]; best=None
        for provider,model,base,key in providers:
            try:
                started=time.time()
                if base=="hub": rec=run_hub(index,row,model,args.max_turns)
                else: rec=run_teacher(index,row,model,args.max_turns,base,key)
                reward=hb.reward(row["answer"][0],rec.get("answer_text",""))
                record={"financebench_id":fid,"question":row["question"],"gold":row["answer"][0],"teacher_model":model,"provider":provider,"tool_calls":rec.get("tool_calls",[]),"trace":rec.get("trace",[]),"answer_text":rec.get("answer_text",""),"reward":reward,"wall_s":round(time.time()-started,2),"targeted":True}
                with out.open("a") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
                if best is None or reward>best["reward"] or (record["answer_text"].strip() and not best["answer_text"].strip()): best=record
                if reward>=1.0 and record["answer_text"].strip():
                    with exact.open("a") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
                    exact_existing[fid]=record; break
            except Exception as exc:
                with out.open("a") as f: f.write(json.dumps({"financebench_id":fid,"provider":provider,"teacher_model":model,"error":repr(exc),"targeted":True},ensure_ascii=False)+"\n")
                continue
        done+=1
        if done%5==0: print(f"[{done}/{len(missing)}] targeted; exact={len(exact_existing)}",flush=True)
    print(json.dumps({"targeted":len(missing),"exact":len(exact_existing),"out":str(out),"exact_out":str(exact)}))

if __name__=="__main__": main()
