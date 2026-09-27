#!/usr/bin/env python3
"""Exact local-policy, stateful FinanceBench GRPO pilot.

The policy that samples each assistant action is the same Unsloth model whose
LoRA parameters are updated.  Tool observations are inserted between assistant
actions, and each action stores exact token IDs plus generation-time log-probs.
This is intentionally a conservative Stage-0/Stage-1 implementation: it logs
all rollout and optimizer artifacts and defaults to teacher guidance disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401; must precede transformers/trl imports
import torch
from unsloth import FastLanguageModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import financebench_harness as hb

if Path("/workspace/split.json").exists():
    hb.BASE = Path("/workspace")
    hb.FILINGS = hb.BASE / "filings"
    hb.TEXTDIR = hb.BASE / "text"
    hb.DATA = hb.BASE / "data"
    hb.CACHE = hb.BASE / "artifacts" / "page_cache"

SYSTEM = """You are a financial-filings retrieval agent.

Answer the user's question from the SEC filing corpus. Search before answering.
Start with bm25_search, then use grep_document and read/read_table for bounded,
page-aware evidence. Use calculate only for arithmetic. Do not invent values.
When you have enough evidence, call finish(answer, evidence_document,
evidence_page) exactly once. Put the complete answer, units, and conclusion in
answer. Do not stop after a search hit unless it resolves the question.
"""

TOOLS = [
    {"type": "function", "function": {"name": "bm25_search", "description": "Ranked keyword search over prose and tables.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "items": {"type": "string"}}, "company": {"type": "string"}, "year": {"type": "integer"}, "filing_type": {"type": "string"}, "document_id": {"type": "string"}, "scope": {"type": "string", "enum": ["prose", "tables", "both"]}, "top_k": {"type": "integer"}}, "required": ["query_list"]}}},
    {"type": "function", "function": {"name": "grep_document", "description": "Search one filing for exact terms or regexes.", "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}, "patterns": {"type": "array", "items": {"type": "string"}}, "page_start": {"type": "integer"}, "page_end": {"type": "integer"}, "context_lines": {"type": "integer"}, "grep_type": {"type": "string"}}, "required": ["document_id", "patterns"]}}},
    {"type": "function", "function": {"name": "search_tables", "description": "Search table-like filing pages.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "items": {"type": "string"}}, "document_id": {"type": "string"}, "company": {"type": "string"}, "year": {"type": "integer"}, "top_k": {"type": "integer"}}, "required": ["query_list"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a bounded page or passage with provenance.", "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}, "page": {"type": "integer"}, "start": {"type": "integer"}, "end": {"type": "integer"}, "passage_id": {"type": "string"}}, "required": ["document_id"]}}},
    {"type": "function", "function": {"name": "read_table", "description": "Read a table candidate and neighboring context.", "parameters": {"type": "object", "properties": {"table_id": {"type": "string"}, "include_neighbors": {"type": "boolean"}}, "required": ["table_id"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate safe basic arithmetic.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "finish", "description": "Submit final answer and primary evidence provenance.", "parameters": {"type": "object", "properties": {"answer": {"type": "string"}, "evidence_document": {"type": "string"}, "evidence_page": {"type": "integer"}}, "required": ["answer"]}}},
]


def _arg_list(args: dict[str, Any], key: str) -> list[str]:
    value = args.get(key, [])
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [value]
    return [str(value)] if value else []


def execute_local(index: hb.StructuredIndex, name: str, args: dict[str, Any]) -> str:
    if name == "bm25_search":
        filters = {"company": args.get("company", ""), "year": args.get("year", -1), "filing_type": args.get("filing_type", ""), "document_id": args.get("document_id", "")}
        top_k = max(1, min(int(args.get("top_k", 3)), 8))
        queries = _arg_list(args, "query_list")
        out: dict[str, Any] = {"queries": queries, "filters": filters}
        scope = str(args.get("scope", "both")).lower()
        if scope in ("prose", "both"):
            out["prose_hits"] = index.search_prose(queries, filters, top_k)
        if scope in ("tables", "table", "both"):
            out["table_hits"] = index.search_tables(queries, filters, top_k)
        return json.dumps(out, ensure_ascii=False)
    if name == "grep_document":
        out = index.grep_document(args["document_id"], _arg_list(args, "patterns"), int(args.get("page_start", -1)), int(args.get("page_end", -1)), int(args.get("context_lines", 2)))
        return json.dumps({"matches": out, "backend_used": "page_text"}, ensure_ascii=False)
    if name == "search_tables":
        filters = {"document_id": args.get("document_id", ""), "company": args.get("company", ""), "year": args.get("year", -1)}
        queries = _arg_list(args, "query_list")
        out = index.search_tables(queries, filters, max(1, min(int(args.get("top_k", 3)), 8)))
        return json.dumps({"queries": queries, "table_hits": out}, ensure_ascii=False)
    if name == "read":
        return json.dumps(index.read(args["document_id"], int(args.get("page", -1)), int(args.get("start", 0)), int(args.get("end", hb.MAX_READ)), args.get("passage_id", "")), ensure_ascii=False)
    if name == "read_table":
        return json.dumps(index.read_table(args["table_id"], bool(args.get("include_neighbors", True))), ensure_ascii=False)
    if name == "calculate":
        try:
            return json.dumps({"expression": args["expression"], "value": hb.calculate(args["expression"])})
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    if name == "finish":
        return json.dumps({"finish": True, "answer": str(args.get("answer", "")), "evidence_document": str(args.get("evidence_document", "")), "evidence_page": int(args.get("evidence_page", -1))})
    return json.dumps({"error": f"Unknown tool {name}"})


def load_rows(ids: list[str]) -> list[dict[str, Any]]:
    rows = json.loads((hb.BASE / "split.json").read_text())["train"]
    wanted = set(ids)
    out = []
    for row in rows:
        if row.get("financebench_id") not in wanted or not str(row.get("answer") or "").strip():
            continue
        company = row.get("company") or row.get("doc_name", "").split("_")[0]
        question = str(row["question"]).strip()
        if company.lower() not in question.lower():
            question = f"About {company}: {question}"
        out.append({"financebench_id": row["financebench_id"], "question": question, "gold": str(row["answer"]).strip()})
    missing = wanted - {r["financebench_id"] for r in out}
    if missing:
        raise ValueError(f"ids not found in train split: {sorted(missing)}")
    return out


def _json_arg(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen JSON/XML tool-call variants without altering generated tokens."""
    calls: list[dict[str, Any]] = []
    patterns = [
        r"<tool_call>\s*(.*?)\s*</tool_call>",
    ]
    for pattern in patterns:
        matches = list(__import__("re").finditer(pattern, text, __import__("re").S))
        if not matches:
            continue
        for match in matches:
            inner = match.group(1).strip()
            obj = _json_arg(inner)
            if obj.get("name"):
                calls.append({"name": str(obj["name"]), "arguments": _json_arg(obj.get("arguments", {}))})
                continue
            fn = __import__("re").search(r"<function=([\w.\-]+)", inner)
            if fn:
                args: dict[str, Any] = {}
                for pm in __import__("re").finditer(r"<parameter=([\w_]+)>(.*?)</parameter>", inner, __import__("re").S):
                    value = pm.group(2).strip()
                    try:
                        value = json.loads(value)
                    except Exception:
                        pass
                    args[pm.group(1)] = value
                calls.append({"name": fn.group(1), "arguments": args})
        break
    if calls:
        return calls
    # Some templates emit a bare tool-call JSON object.
    for match in __import__("re").finditer(r"\{\s*[\"']name[\"']\s*:\s*[\"']([\w.\-]+)[\"'].*?\}", text, __import__("re").S):
        obj = _json_arg(match.group(0))
        if obj.get("name"):
            calls.append({"name": str(obj["name"]), "arguments": _json_arg(obj.get("arguments", {}))})
    return calls


def _inner_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def _template_ids(tokenizer: Any, messages: list[dict[str, Any]], max_length: int, max_new_tokens: int) -> torch.Tensor:
    tokenizer=_inner_tokenizer(tokenizer)
    encoded = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    if isinstance(encoded, dict):
        ids = encoded["input_ids"]
    else:
        ids = encoded
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    keep = max(1, max_length - max_new_tokens)
    if ids.shape[1] > keep:
        ids = ids[:, -keep:]
    return ids


def _generation_logprobs(scores: tuple[torch.Tensor, ...], token_ids: torch.Tensor) -> list[float]:
    values: list[float] = []
    for step, score in enumerate(scores):
        if step >= token_ids.shape[0]:
            break
        values.append(float(torch.log_softmax(score[0].float(), dim=-1)[int(token_ids[step])].detach().cpu()))
    return values


def _decode(tokenizer: Any, ids: torch.Tensor) -> str:
    return _inner_tokenizer(tokenizer).decode(ids.tolist(), skip_special_tokens=False)


def rollout_one(model: torch.nn.Module, tokenizer: Any, index: hb.StructuredIndex, row: dict[str, Any], seed: int, max_turns: int, max_new_tokens: int, max_seq_length: int, temperature: float, top_p: float) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}]
    trace: list[dict[str, Any]] = []
    action_records: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    finish_metadata: dict[str, Any] = {}
    answer = ""
    termination = "max_turns"
    started = time.time()
    for turn in range(max_turns):
        prompt_ids = _template_ids(tokenizer, messages, max_seq_length, max_new_tokens)
        attention = torch.ones_like(prompt_ids)
        device = next(model.parameters()).device
        prompt_ids = prompt_ids.to(device)
        attention = attention.to(device)
        with torch.no_grad():
            generated = model.generate(input_ids=prompt_ids, attention_mask=attention, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=top_p, return_dict_in_generate=True, output_scores=True, pad_token_id=_inner_tokenizer(tokenizer).eos_token_id)
        action_ids = generated.sequences[0, prompt_ids.shape[1]:].detach().cpu()
        action_text = _decode(tokenizer, action_ids)
        generation_logprobs = _generation_logprobs(generated.scores, action_ids)
        if not action_ids.numel():
            termination = "empty_generation"
            break
        calls = parse_tool_calls(action_text)
        if not calls:
            answer = action_text.strip()
            trace.append({"role": "assistant", "content": answer, "token_ids": action_ids.tolist()})
            action_records.append({"turn": turn, "text": action_text, "token_ids": action_ids.tolist(), "action_mask":[1]*int(action_ids.numel()), "prompt_token_ids": prompt_ids[0].detach().cpu().tolist(), "old_logprobs": [], "sampling_logprobs": generation_logprobs, "tool_calls": []})
            termination = "assistant"
            break
        assistant_calls=[]
        openai_calls=[]
        tool_items=[]
        for idx, call in enumerate(calls):
            call_id=f"local-{seed}-{turn}-{idx}"
            args=call["arguments"]
            call_record={"name":call["name"],"arguments":args,"call_id":call_id,"turn":turn}
            tool_calls.append(call_record)
            assistant_calls.append(call_record)
            openai_calls.append({"id":call_id,"type":"function","function":{"name":call["name"],"arguments":args}})
        messages.append({"role":"assistant","content":None,"tool_calls":openai_calls})
        for call_record in assistant_calls:
            raw_observation=execute_local(index,call_record["name"],call_record["arguments"])
            observation=raw_observation[:4000]
            messages.append({"role":"tool","tool_call_id":call_record["call_id"],"content":observation})
            tool_items.append({"role":"tool","call_id":call_record["call_id"],"content":observation,"observation_chars":len(raw_observation),"observation_truncated":len(raw_observation)>len(observation)})
            trace.append({"role":"tool","call_id":call_record["call_id"],"content":observation,"observation_chars":len(raw_observation),"observation_truncated":len(raw_observation)>len(observation)})
            if call_record["name"]=="finish":
                finish_args=call_record["arguments"]
                answer=str(finish_args.get("answer","")).strip()
                raw_page=finish_args.get("evidence_page",-1)
                finish_metadata={"evidence_document":str(finish_args.get("evidence_document","")),"evidence_page":int(raw_page) if str(raw_page).lstrip("-").isdigit() else -1}
        trace.insert(len(trace)-len(tool_items),{"role":"assistant","tool_calls":assistant_calls,"token_ids":action_ids.tolist()})
        action_records.append({"turn":turn,"text":action_text,"token_ids":action_ids.tolist(),"action_mask":[1]*int(action_ids.numel()),"prompt_token_ids":prompt_ids[0].detach().cpu().tolist(),"old_logprobs":[],"sampling_logprobs":generation_logprobs,"tool_calls":assistant_calls})
        if answer:
            termination="finish"
            break
    answer_reward, answer_parts = hb.score_answer(row["gold"], answer)
    evidence_reward, evidence_parts = hb.score_evidence(row["gold"], answer, trace)
    reward = hb.grounded_reward(answer_reward, evidence_reward)
    return {"financebench_id":row["financebench_id"],"question":row["question"],"gold":row["gold"],"answer_text":answer,"trace":trace,"tool_calls":tool_calls,"termination_reason":termination,"finish_ok":float(termination=="finish"),"finish_metadata":finish_metadata,"answer_reward":answer_reward,"evidence_reward":evidence_reward,"reward":reward,"reward_parts":{**answer_parts,**evidence_parts},"seed":seed,"sampling":{"temperature":temperature,"top_p":top_p,"top_k":None,"max_new_tokens":max_new_tokens},"action_records":action_records,"wall_s":time.time()-started}


def action_logprobs(model: torch.nn.Module, prompt_ids: list[int], action_ids: list[int], max_seq_length: int) -> torch.Tensor:
    device=next(model.parameters()).device
    full=torch.tensor([prompt_ids+action_ids],device=device,dtype=torch.long)
    if full.shape[1]>max_seq_length:
        raise RuntimeError(f"action sequence exceeds max_seq_length: {full.shape[1]} > {max_seq_length}")
    out=model(input_ids=full,attention_mask=torch.ones_like(full))
    logits=out.logits[:,:-1,:].float()
    labels=full[:,1:]
    logps=torch.log_softmax(logits,dim=-1).gather(-1,labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
    start=max(0,len(prompt_ids)-1)
    return logps[start:start+len(action_ids)]


def reference_logprobs(model: torch.nn.Module, records: list[dict[str,Any]], max_seq_length: int) -> None:
    model.eval()
    with torch.no_grad():
        for record in records:
            for action in record.get("action_records",[]):
                vals=action_logprobs(model,action["prompt_token_ids"],action["token_ids"],max_seq_length)
                action["reference_logprobs"]=[float(x) for x in vals.detach().cpu()]


def teacher_signal(record: dict[str,Any], teacher: dict[str,Any] | None) -> float:
    if not teacher or not record.get("answer_text"):
        return 0.0
    try:
        return float(hb.score_answer(str(teacher.get("answer_text",teacher.get("gold",""))),record["answer_text"])[0])
    except Exception:
        return 0.0


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args,stderr=subprocess.STDOUT,text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def file_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    manifest=[]
    for root in paths:
        if not root.exists():
            manifest.append({"path":str(root),"missing":True})
            continue
        candidates=[root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in sorted(candidates):
            item={"path":str(path.relative_to(hb.BASE) if path.is_relative_to(hb.BASE) else path),"bytes":path.stat().st_size}
            if path.stat().st_size <= 64*1024*1024:
                item["sha256"]=sha256(path)
            manifest.append(item)
    return manifest


def environment_metadata() -> dict[str, Any]:
    packages={}
    for name in ["unsloth","transformers","trl","torch","accelerate","peft","datasets"]:
        try: packages[name]=importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError: packages[name]=None
    gpu=[]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props=torch.cuda.get_device_properties(i)
            gpu.append({"index":i,"name":props.name,"total_memory":props.total_memory,"capability":[props.major,props.minor]})
    return {"python":sys.version,"platform":platform.platform(),"hostname":platform.node(),"packages":packages,"torch_cuda":torch.version.cuda,"gpu":gpu}


def save_rng_state(path: Path) -> None:
    torch.save({"python":random.getstate(),"torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},path)


def log_mlflow_if_requested(out: Path, args: argparse.Namespace, metrics: dict[str, Any]) -> dict[str, Any]:
    if not args.mlflow_uri:
        return {"status":"not_requested"}
    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=metrics["run_id"]) as run:
            mlflow.log_params({k:str(v) for k,v in vars(args).items() if k != "teacher_traces"})
            mlflow.log_metrics({k:float(v) for k,v in metrics.items() if isinstance(v,(int,float)) and math.isfinite(float(v))})
            for name in ["config.json","environment.json","corpus_manifest.json","teacher_trace_hashes.json","rollouts.jsonl","optimizer_metrics.jsonl","summary.json"]:
                path=out/name
                if path.exists(): mlflow.log_artifact(str(path))
            return {"status":"logged","tracking_uri":args.mlflow_uri,"experiment":args.mlflow_experiment,"run_id":run.info.run_id}
    except Exception as exc:
        return {"status":"error","error":repr(exc),"tracking_uri":args.mlflow_uri}


def group_advantages(records: list[dict[str,Any]], teacher_by_id: dict[str,dict[str,Any]], coef: float) -> dict[int,float]:
    groups: dict[str,list[dict[str,Any]]]={}
    for r in records: groups.setdefault(r["financebench_id"],[]).append(r)
    advantages={}
    for fid,group in groups.items():
        primary=[float(r["reward"]) for r in group]
        tie_signals=[teacher_signal(r,teacher_by_id.get(fid)) for r in group]
        tie_applied=[0.0]*len(group)
        if coef and max(primary)-min(primary)<=1e-8:
            tie_applied=[coef*x for x in tie_signals]
        final=[p+t for p,t in zip(primary,tie_applied)]
        mean=sum(final)/len(final)
        std=math.sqrt(sum((x-mean)**2 for x in final)/len(final))
        denom=std if std>1e-8 else 1.0
        for r,value,tie,signal in zip(group,final,tie_applied,tie_signals):
            r["teacher_signal"]=signal
            r["teacher_tiebreak_reward"]=tie
            r["final_reward"]=value
            r["group_mean"]=mean
            r["group_std"]=std
            r["advantage"]=(value-mean)/denom
            advantages[id(r)]=r["advantage"]
    return advantages


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="/hf/models--Qwen--Qwen3.5-4B")
    ap.add_argument("--ids",default="financebench_id_01226,financebench_id_01936")
    ap.add_argument("--ids-file",default="")
    ap.add_argument("--teacher-traces",default="")
    ap.add_argument("--teacher-tie-coef",type=float,default=0.0)
    ap.add_argument("--group-size",type=int,default=2)
    ap.add_argument("--seed",type=int,default=101)
    ap.add_argument("--temperature",type=float,default=0.3)
    ap.add_argument("--top-p",type=float,default=0.95)
    ap.add_argument("--max-turns",type=int,default=6)
    ap.add_argument("--max-new-tokens",type=int,default=256)
    ap.add_argument("--max-seq-length",type=int,default=16384)
    ap.add_argument("--lora-rank",type=int,default=32)
    ap.add_argument("--learning-rate",type=float,default=1e-6)
    ap.add_argument("--clip-epsilon",type=float,default=0.2)
    ap.add_argument("--max-grad-norm",type=float,default=1.0)
    ap.add_argument("--output-dir",default="")
    ap.add_argument("--mlflow-uri",default="")
    ap.add_argument("--mlflow-experiment",default="financebench-grpo")
    ap.add_argument("--reference-kl",action=argparse.BooleanOptionalAction,default=True)
    args=ap.parse_args()
    if args.ids_file:
        args.ids=[x.strip() for x in Path(args.ids_file).read_text().splitlines() if x.strip()]
    else:
        args.ids=[x.strip() for x in args.ids.split(",") if x.strip()]
    run_id=time.strftime("%Y%m%d_%H%M%S")+f"_qwen4b_g{args.group_size}_s{args.seed}"
    out=Path(args.output_dir or (hb.BASE/"results"/"grpo_runs"/run_id))
    out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(vars(args)|{"run_id":run_id,"system_prompt":SYSTEM,"tool_schema":TOOLS},indent=2,default=str))
    (out/"git_commit.txt").write_text(git_text(["git","rev-parse","HEAD"]))
    (out/"git_status.txt").write_text(git_text(["git","status","--short"]))
    (out/"git_diff.patch").write_text(git_text(["git","diff","HEAD"]))
    (out/"environment.json").write_text(json.dumps(environment_metadata(),indent=2,default=str))
    save_rng_state(out/"rng_state_initial.pt")
    (out/"corpus_manifest.json").write_text(json.dumps(file_manifest([hb.BASE/"split.json",hb.DATA,hb.CACHE,Path(__file__),hb.BASE/"GRPO_PLAN.md"]),indent=2))
    rows=load_rows(args.ids)
    teacher_by_id={}
    if args.teacher_traces:
        teacher_by_id={r["financebench_id"]:r for r in (json.loads(x) for x in Path(args.teacher_traces).read_text().splitlines() if x.strip())}
        (out/"teacher_trace_hashes.json").write_text(json.dumps({k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in teacher_by_id.items()},indent=2))
    index=hb.build_index()
    print(json.dumps({"stage":"model_load","run_id":run_id,"output":str(out),"questions":len(rows)}),flush=True)
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=args.model,max_seq_length=args.max_seq_length,load_in_4bit=False,load_in_16bit=True,full_finetuning=False,fast_inference=False)
    reference=None
    if args.reference_kl:
        reference,_=FastLanguageModel.from_pretrained(model_name=args.model,max_seq_length=args.max_seq_length,load_in_4bit=False,load_in_16bit=True,full_finetuning=False,fast_inference=False)
        reference.eval()
    model=FastLanguageModel.get_peft_model(model,r=args.lora_rank,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],lora_alpha=args.lora_rank,lora_dropout=0,bias="none",use_gradient_checkpointing="unsloth",random_state=3407,max_seq_length=args.max_seq_length)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    records=[]
    with (out/"rollouts.jsonl").open("a") as fh:
        for row_index,row in enumerate(rows):
            for generation in range(args.group_size):
                seed=args.seed+row_index*1000+generation
                record=rollout_one(model,tokenizer,index,row,seed,args.max_turns,args.max_new_tokens,args.max_seq_length,args.temperature,args.top_p)
                with torch.no_grad():
                    for action in record["action_records"]:
                        exact_old=action_logprobs(model,action["prompt_token_ids"],action["token_ids"],args.max_seq_length)
                        action["old_logprobs"]=[float(x) for x in exact_old.detach().cpu()]
                if reference is not None:
                    reference_logprobs(reference,[record],args.max_seq_length)
                    for action in record["action_records"]:
                        old=action["old_logprobs"]; ref=action.get("reference_logprobs",[])
                        action["reference_kl_mean"]=sum(o-r for o,r in zip(old,ref))/max(1,len(ref))
                records.append(record)
                fh.write(json.dumps(record,ensure_ascii=False)+"\n");fh.flush()
                print(json.dumps({"stage":"rollout","id":record["financebench_id"],"generation":generation,"reward":record["reward"],"answer_reward":record["answer_reward"],"evidence_reward":record["evidence_reward"],"termination":record["termination_reason"],"actions":len(record["action_records"]),"tools":len(record["tool_calls"])}),flush=True)
    advantages=group_advantages(records,teacher_by_id,args.teacher_tie_coef)
    model.train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    losses=[]; ratio_values=[]; clipped=0; total_tokens=0
    for record in records:
        adv=float(advantages[id(record)])
        for action in record["action_records"]:
            new=action_logprobs(model,action["prompt_token_ids"],action["token_ids"],args.max_seq_length)
            old=torch.tensor(action["old_logprobs"],device=new.device,dtype=new.dtype)
            n=min(new.numel(),old.numel(),len(action["token_ids"]))
            new=new[:n]; old=old[:n]
            action["new_logprobs_before_update"]=[float(x) for x in new.detach().cpu()]
            ratio=torch.exp(new-old.detach())
            clipped_ratio=torch.clamp(ratio,1.0-args.clip_epsilon,1.0+args.clip_epsilon)
            objective=torch.minimum(ratio*adv,clipped_ratio*adv)
            loss=-objective.mean()/max(1,len(records))
            loss.backward()
            losses.append(float(loss.detach().cpu()));ratio_values.extend(ratio.detach().float().cpu().tolist());clipped+=int((ratio!=clipped_ratio).sum().item());total_tokens+=n
    grad_norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.max_grad_norm)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        for record in records:
            for action in record["action_records"]:
                post=action_logprobs(model,action["prompt_token_ids"],action["token_ids"],args.max_seq_length)
                action["new_logprobs_after_update"]=[float(x) for x in post.detach().cpu()]
    (out/"rollouts.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    adapter=out/"adapter-step-1";model.save_pretrained(adapter);tokenizer.save_pretrained(adapter)
    metrics={"stage":"exact_on_policy_optimizer_step_complete","run_id":run_id,"questions":args.ids,"records":len(records),"group_size":args.group_size,"groups_processed":len(records)//max(1,args.group_size),"optimizer_steps":1,"mean_reward":sum(r["reward"] for r in records)/max(1,len(records)),"mean_answer_reward":sum(r["answer_reward"] for r in records)/max(1,len(records)),"mean_evidence_reward":sum(r["evidence_reward"] for r in records)/max(1,len(records)),"zero_variance_groups":sum(1 for fid in set(r["financebench_id"] for r in records) if max(r2["reward"] for r2 in records if r2["financebench_id"]==fid)-min(r2["reward"] for r2 in records if r2["financebench_id"]==fid)<=1e-8),"loss_mean":sum(losses)/max(1,len(losses)),"grad_norm":float(grad_norm),"ratio_mean":sum(ratio_values)/max(1,len(ratio_values)),"ratio_min":min(ratio_values or [0]),"ratio_max":max(ratio_values or [0]),"clipped_fraction":clipped/max(1,total_tokens),"action_tokens":total_tokens,"teacher_tie_coef":args.teacher_tie_coef,"reference_kl_mean":sum(float(a.get("reference_kl_mean",0)) for r in records for a in r["action_records"])/max(1,sum(len(r["action_records"]) for r in records)),"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,"learning_rate":args.learning_rate,"clip_epsilon":args.clip_epsilon,"adapter":str(adapter),"adapter_sha256":sha256(adapter/"adapter_model.safetensors") if (adapter/"adapter_model.safetensors").exists() else None}
    (out/"optimizer_metrics.jsonl").write_text(json.dumps(metrics,default=float)+"\n")
    (out/"summary.json").write_text(json.dumps(metrics,indent=2,default=float))
    tracking=log_mlflow_if_requested(out,args,metrics)
    (out/"mlflow_status.json").write_text(json.dumps(tracking,indent=2))
    print(json.dumps(metrics|{"mlflow":tracking},indent=2,default=float),flush=True)

if __name__=="__main__":
    main()
