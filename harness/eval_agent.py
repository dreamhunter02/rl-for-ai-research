"""Comparable smoke evaluation for OpenAI and Tinker FinanceBench agents.

This is intentionally inference-only: it saves full trajectories and metrics but
does not start RL. Use --limit 5 for a cheap check and --limit 42 for the held-out
run after the tool loop is stable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import financebench_harness as hb
import finance_env as fe

BASE = hb.BASE
SYSTEM = fe.FINANCE_TASK_INSTRUCTIONS


def openai_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for env_path in (Path.home() / ".hermes" / ".env", Path.home() / ".hermes" / "hermes-agent" / ".env"):
        if env_path.exists():
            for line in env_path.read_text(errors="ignore").splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def rows_for_eval(limit: int) -> list[dict[str, Any]]:
    rows = fe.load_financebench("eval")
    return rows[:limit] if limit else rows


def tool_specs(tool_obj: fe.Bm25Tool) -> list[dict[str, Any]]:
    names = [tool_obj.bm25_search, tool_obj.grep_document, tool_obj.search_tables, tool_obj.read, tool_obj.read_table, tool_obj.calculate, tool_obj.finish]
    return [x.to_spec() for x in names]


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
        query_list = _arg_list(args, "query_list")
        filters = {"company": args.get("company", ""), "year": args.get("year", -1), "filing_type": args.get("filing_type", ""), "document_id": args.get("document_id", "")}
        scope = str(args.get("scope", "both")).lower()
        top_k = max(1, min(int(args.get("top_k", 3)), 8))
        out: dict[str, Any] = {"queries": query_list, "filters": filters}
        if scope in ("prose", "both"):
            out["prose_hits"] = index.search_prose(query_list, filters, top_k)
        if scope in ("tables", "table", "both"):
            out["table_hits"] = index.search_tables(query_list, filters, top_k)
        return json.dumps(out, ensure_ascii=False)
    if name == "grep_document":
        out = index.grep_document(args["document_id"], _arg_list(args, "patterns"), int(args.get("page_start", -1)), int(args.get("page_end", -1)), int(args.get("context_lines", 2)))
        return json.dumps({"grep_type_requested": args.get("grep_type", "text"), "backend_used": "page_text", "matches": out}, ensure_ascii=False)
    if name == "search_tables":
        filters = {"document_id": args.get("document_id", ""), "company": args.get("company", ""), "year": args.get("year", -1)}
        out = index.search_tables(_arg_list(args, "query_list"), filters, max(1, min(int(args.get("top_k", 3)), 8)))
        return json.dumps({"queries": _arg_list(args, "query_list"), "table_hits": out}, ensure_ascii=False)
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
        return json.dumps({"status": "finished", "answer": str(args.get("answer", "")), "evidence_document": str(args.get("evidence_document", "")), "evidence_page": int(args.get("evidence_page", -1))})
    return json.dumps({"error": f"Unknown tool {name}"})


def answer_reward(text: str, gold: str) -> float:
    return hb.reward(gold, text)


def extract_call(text: str) -> tuple[str, dict[str, Any]] | None:
    block = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text or "", re.S)
    body = block.group(1).strip() if block else (text or "")
    if block:
        try:
            obj = json.loads(body)
            if isinstance(obj, dict) and obj.get("name"):
                return obj["name"], obj.get("arguments", {})
        except Exception:
            pass
    fn = re.search(r"<function=([\w.-]+)", body)
    if not fn:
        return None
    args: dict[str, Any] = {}
    for m in re.finditer(r"<parameter=([\w_]+)>(.*?)</parameter>", body, re.S):
        value = m.group(2).strip()
        try:
            args[m.group(1)] = json.loads(value)
        except Exception:
            if value.lower() in ("true", "false"):
                args[m.group(1)] = value.lower() == "true"
            else:
                try:
                    args[m.group(1)] = int(value)
                except ValueError:
                    args[m.group(1)] = value
    return fn.group(1), args


def assistant_text(message: Any) -> str:
    return getattr(message, "content", None) or ""


def _openai_tools() -> list[dict[str, Any]]:
    return [
        {"type": "function", "name": s["name"], "description": s.get("description", ""), "parameters": s["parameters"]}
        for s in tool_specs(fe.Bm25Tool(None))  # type: ignore[arg-type]
    ]


def run_openai(index: hb.StructuredIndex, row: dict[str, Any], model: str, max_turns: int) -> dict[str, Any]:
    """Teacher-style OpenAI evaluation using the Responses API.

    The first request receives the task and tools. Subsequent requests resend
    only the prior model output plus the local tool results, which is the
    verified supported pattern for gpt-5.6-terra.
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_key())
    tools = _openai_tools()
    response = client.responses.create(model=model, input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}], tools=tools, tool_choice="auto", reasoning={"effort": "medium"}, max_output_tokens=2048, store=False)
    calls = []
    final = ""
    for _ in range(max_turns):
        output = response.output
        if not output:
            break
        if output[-1].type == "message":
            if output[-1].content:
                final = " ".join(getattr(part, "text", "") for part in output[-1].content if getattr(part, "type", "") == "output_text")
            break
        tool_outputs: list[dict[str, Any]] = []
        for item in output:
            if item.type == "function_call":
                args = json.loads(item.arguments or "{}")
                calls.append({"turn": len(calls), "name": item.name, "arguments": args, "latency_s": 0.0})
                tool_outputs.append({"type": "function_call_output", "call_id": item.call_id, "output": execute_local(index, item.name, args)})
        if not tool_outputs:
            break
        previous_output = [item.model_dump(exclude_none=True) for item in output]
        response = client.responses.create(model=model, input=previous_output + tool_outputs, tools=tools, tool_choice="auto", reasoning={"effort": "medium"}, max_output_tokens=2048, store=False)
    return {"backend": "openai", "model": model, "answer_text": final, "tool_calls": calls, "messages": [{"role": "assistant", "content": final, "tool_calls": calls}], "openai_response_id": response.id}


def run_tinker(index: hb.StructuredIndex, row: dict[str, Any], model: str, project: str, max_turns: int, model_path: str | None = None, renderer_name: str | None = None) -> dict[str, Any]:
    from tinker import SamplingParams, ServiceClient
    from tinker_cookbook import tokenizer_utils
    from tinker_cookbook.renderers import get_renderer
    tokenizer = tokenizer_utils.get_tokenizer(model)
    renderer = get_renderer(renderer_name or ("nemotron3_ultra" if model.startswith("nvidia/NVIDIA-Nemotron-3.5") else "qwen3_5"), tokenizer)
    tool_obj = fe.Bm25Tool(index)
    messages = fe._initial_messages(row, renderer, tool_obj)
    client = ServiceClient(project_id=project, api_key=load_tinker_key()).create_sampling_client(model_path=model_path, base_model=None if model_path else model)
    params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=1024)
    calls = []
    final = ""
    for turn in range(max_turns):
        prompt = renderer.build_generation_prompt(messages)
        response = client.sample(prompt=prompt, num_samples=1, sampling_params=params, include_prompt_logprobs=False)
        if hasattr(response, "result"):
            response = response.result()
        seq = response.sequences[0]
        text = tokenizer.decode(list(seq.tokens_np), skip_special_tokens=True)
        parsed = extract_call(text)
        if not parsed:
            final = text
            messages.append({"role": "assistant", "content": text})
            break
        name, args = parsed
        messages.append({"role": "assistant", "content": text})
        started = time.time()
        try:
            result = execute_local(index, name, args)
        except Exception as exc:
            result = json.dumps({"error": f"tool execution failed: {type(exc).__name__}: {exc}"}, ensure_ascii=False)
        calls.append({"turn": turn, "name": name, "arguments": args, "latency_s": time.time() - started})
        messages.append({"role": "tool", "content": result})
    return {"backend": "tinker", "model": model, "answer_text": final, "tool_calls": calls, "messages": messages}


def load_tinker_key() -> str:
    value = Path.home().joinpath(".config/tinker/key").read_text().strip()
    if "=" in value:
        value = value.split("=", 1)[1].strip()
    if value.startswith("tinker-"):
        value = "tml-" + value[len("tinker-"):]
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "tinker"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", help="Tinker sampling-client model_path (trained LoRA checkpoint).")
    ap.add_argument("--renderer", default="", help="Renderer override, for example nemotron3_ultra.")
    ap.add_argument("--project", default="5485278b-9573-47cd-816c-9e380e84461f")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.backend == "openai" and not openai_key():
        raise SystemExit("OPENAI_API_KEY is not available in the environment or Hermes .env")
    rows = rows_for_eval(args.limit)
    index = hb.build_index()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    records = []
    runner = run_openai if args.backend == "openai" else run_tinker
    for i, row in enumerate(rows, 1):
        try:
            rec = runner(index, row, args.model, args.max_turns) if args.backend == "openai" else runner(index, row, args.model, args.project, args.max_turns, model_path=args.model_path, renderer_name=args.renderer or None)
            rec.update({"financebench_id": row["financebench_id"], "question": row["question"], "gold": row["answer"][0], "reward": answer_reward(rec.get("answer_text", ""), row["answer"][0])})
        except Exception as exc:
            rec = {"backend": args.backend, "model": args.model, "financebench_id": row["financebench_id"], "question": row["question"], "gold": row["answer"][0], "error": repr(exc), "reward": 0.0}
        records.append(rec)
        print(json.dumps({"i": i, "id": row["financebench_id"], "reward": rec.get("reward"), "error": rec.get("error")}, ensure_ascii=False), flush=True)
        Path(args.out).write_text(json.dumps(records, indent=2, ensure_ascii=False))
    valid = [x for x in records if "error" not in x]
    print(json.dumps({"n": len(records), "valid": len(valid), "mean_reward": sum(x.get("reward", 0) for x in valid) / max(1, len(valid)), "exact_or_numeric": sum(x.get("reward", 0) >= 0.9 for x in valid)}, indent=2))


if __name__ == "__main__":
    main()
