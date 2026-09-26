from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

MODEL_ID = "LiquidAI/LFM2.5-8B-A1B"
MODEL_PATH = "/hf/models--LiquidAI--LFM2.5-8B-A1B/snapshots/5dd22602c2e9f6a097b1de4c4efe0658b605015c"
MAX_NEW_TOKENS = 1024

print(f"loading {MODEL_ID} from {MODEL_PATH}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()
device = next(model.parameters()).device
print(f"ready on {device}", flush=True)
app = FastAPI()


def python_value(value):
    return repr(value)


def assistant_tool_text(tool_calls):
    calls = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        call = fn.get("name", "unknown") + "(" + ", ".join(
            f"{k}={python_value(v)}" for k, v in args.items()
        ) + ")"
        calls.append(call)
    return "<|tool_call_start|>[" + ", ".join(calls) + "]<|tool_call_end|>"


def to_lfm_messages(messages, tools):
    tool_json = json.dumps(
        [x.get("function", x) for x in (tools or [])], ensure_ascii=False
    )
    result = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            content = (content or "") + "\n\nTool schemas (JSON):\n" + tool_json
            result.append({"role": "system", "content": content})
        elif role == "assistant" and message.get("tool_calls"):
            result.append({"role": "assistant", "content": message.get("content") or assistant_tool_text(message["tool_calls"])})
        else:
            result.append({"role": role, "content": content or ""})
    return result


def parse_python_calls(text):
    blocks = re.findall(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", text, flags=re.S)
    calls = []
    for block in blocks:
        try:
            expr = ast.parse(block.strip(), mode="eval").body
            nodes = expr.elts if isinstance(expr, (ast.List, ast.Tuple)) else [expr]
            for node in nodes:
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                args = {}
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    try:
                        args[kw.arg] = ast.literal_eval(kw.value)
                    except Exception:
                        args[kw.arg] = ast.unparse(kw.value)
                calls.append({"name": node.func.id, "arguments": args})
        except Exception as exc:
            print(f"tool parse warning: {exc}", flush=True)
    return calls


def generate(messages, tools, max_tokens):
    prompt = tokenizer.apply_chat_template(
        to_lfm_messages(messages, tools),
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    prompt = {k: v.to(device) for k, v in prompt.items()}
    with torch.inference_mode():
        output = model.generate(
            **prompt,
            do_sample=False,
            max_new_tokens=min(int(max_tokens or MAX_NEW_TOKENS), MAX_NEW_TOKENS),
            repetition_penalty=1.05,
        )
    new_tokens = output[0, prompt["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


@app.get("/health")
def health():
    return {"status": "ready", "model": MODEL_ID}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "LiquidAI"}]}


@app.post("/v1/chat/completions")
async def completions(request: Request):
    payload = await request.json()
    started = time.time()
    raw = generate(payload.get("messages", []), payload.get("tools", []), payload.get("max_tokens", MAX_NEW_TOKENS))
    calls = parse_python_calls(raw)
    if calls:
        tool_calls = []
        for i, call in enumerate(calls):
            tool_calls.append({
                "id": f"lfm-tool-{int(started)}-{i}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)},
            })
        message = {"role": "assistant", "content": raw, "tool_calls": tool_calls}
        finish = "tool_calls"
    else:
        clean = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", raw).strip()
        message = {"role": "assistant", "content": clean}
        finish = "stop"
    return JSONResponse({
        "id": f"lfm-chat-{int(started * 1000)}",
        "object": "chat.completion",
        "created": int(started),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18350, log_level="info")
