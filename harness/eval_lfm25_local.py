"""Local Transformers FinanceBench baseline for LiquidAI LFM2.5-8B-A1B."""
from __future__ import annotations

import ast
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import financebench_harness as hb
from eval_agent import execute_local, tool_specs
from generate_teacher_traces import _rows

MODEL_ID = "LiquidAI/LFM2.5-8B-A1B"
MODEL_PATH = "/hf/models--LiquidAI--LFM2.5-8B-A1B/snapshots"
OUT = Path("/workspace/results/local_eval/lfm25_8b_a1b_eval_10turn.jsonl")
MAX_TURNS = 10
MAX_NEW_TOKENS = 1024


def resolve_model_path() -> str:
    roots = sorted(Path(MODEL_PATH).glob("*"))
    if not roots:
        return MODEL_ID
    return str(roots[-1])


def parse_python_calls(text: str) -> list[dict]:
    blocks = re.findall(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", text, flags=re.S)
    calls = []
    for block in blocks:
        body = block.strip()
        try:
            expr = ast.parse(body, mode="eval").body
            nodes = expr.elts if isinstance(expr, (ast.List, ast.Tuple)) else [expr]
            for node in nodes:
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                args = {}
                for kw in node.keywords:
                    if kw.arg is not None:
                        try:
                            args[kw.arg] = ast.literal_eval(kw.value)
                        except Exception:
                            args[kw.arg] = ast.unparse(kw.value)
                calls.append({"name": node.func.id, "arguments": args})
        except Exception:
            continue
    if calls:
        return calls
    # Fallback for JSON-style tool calls if the model follows the instruction literally.
    for obj in re.findall(r"\{.*?\}", text, flags=re.S):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict) and parsed.get("name"):
                calls.append({"name": parsed["name"], "arguments": parsed.get("arguments", {})})
        except Exception:
            pass
    return calls


def generate(model, tokenizer, messages: list[dict], device) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=1.05,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def clean_answer(text: str) -> str:
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text).strip()
    return text


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    index = hb.build_index()
    rows = _rows("eval")[: int(os.environ.get("LFM_LIMIT", "42"))]
    specs = tool_specs(__import__("finance_env").Bm25Tool(index))
    tool_text = json.dumps(specs, ensure_ascii=False)
    system = (
        "You are a careful financial research agent. Use the typed tools to retrieve filing "
        "evidence, then answer the exact question. Stop searching once sufficient evidence "
        "is found. End with exactly one line beginning Answer: and include requested units "
        "and precision. Tool calls must use the model's <|tool_call_start|> format.\n"
        "Available tools:\n" + tool_text
    )
    path = resolve_model_path()
    print(f"loading {path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    for i, row in enumerate(rows, 1):
        started = time.time()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}]
        calls = []
        trace = []
        answer = ""
        error = None
        try:
            for _ in range(MAX_TURNS):
                raw = generate(model, tokenizer, messages, device)
                parsed = parse_python_calls(raw)
                if not parsed:
                    answer = clean_answer(raw)
                    trace.append({"role": "assistant", "content": answer})
                    break
                assistant = clean_answer(raw)
                messages.append({"role": "assistant", "content": assistant})
                trace.append({"role": "assistant", "content": assistant})
                for call in parsed:
                    calls.append(call)
                    obs = execute_local(index, call["name"], call["arguments"])
                    messages.append({"role": "tool", "content": obs})
                    trace.append({"role": "tool", "content": obs, "name": call["name"]})
        except Exception as exc:
            error = repr(exc)
        result = {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "gold": row["answer"][0],
            "model": MODEL_ID,
            "answer_text": answer,
            "tool_calls": calls,
            "trace": trace,
            "deterministic_reward": hb.reward(row["answer"][0], answer) if error is None else None,
            "wall_s": round(time.time() - started, 2),
        }
        if error:
            result["error"] = error
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(rows)}] {row['financebench_id']} reward={result['deterministic_reward']} tools={len(calls)} wall={result['wall_s']}s", flush=True)
    records = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    valid = [r for r in records if r.get("deterministic_reward") is not None]
    print(json.dumps({"n": len(records), "valid": len(valid), "exact_or_numeric": sum(r["deterministic_reward"] >= .9 for r in valid), "mean_reward": sum(r["deterministic_reward"] for r in valid) / max(1, len(valid)), "out": str(OUT)}))


if __name__ == "__main__":
    main()
