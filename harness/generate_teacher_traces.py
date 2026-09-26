"""Generate GPT 5.6 teacher traces for SFT warm-up.

Runs the OpenAI teacher on the training split using the structured retrieval
harness and saves complete trajectories in a JSONL format for later SFT.
The model is instructed to end with a single line:

    Answer: <concise final answer>

Only trajectories whose final answer matches the gold (reward >= threshold)
are written to ``--out`` by default; failed runs go to ``--fail-out`` so they
can be inspected and optionally retried with more turns.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import financebench_harness as hb
import finance_env as fe
from eval_agent import SYSTEM, execute_local, openai_key, tool_specs

LEGACY_TEACHER_SYSTEM = SYSTEM + (
    "\n\nIMPORTANT: This is a supervised teacher trajectory. Use the tools to find evidence in the filings, then provide a concise final Answer in plain text. Do not use or expect a finish/submission tool. Use only information from retrieved evidence."
)

TEACHER_SYSTEM = SYSTEM + (
    "\n\nIMPORTANT: This is a supervised teacher trajectory. Use the tools to find "
    "evidence in the filings, then call finish(answer=...) exactly once when done. "
    "The finish answer must be concise and match the requested quantity and units; "
    "also provide evidence_document and evidence_page when known. Do not continue searching after finish. "
    "For compatibility, the saved trace renders the finish answer as an Answer line. "
    "Use only information from the retrieved evidence."
)


def _lfm_text_tool_calls(text: str, start_id: int) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    blocks = re.findall(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", text or "", flags=re.S)
    for block in blocks:
        try:
            root = ast.parse(block.strip(), mode="eval").body
        except SyntaxError:
            continue
        nodes = list(root.elts) if isinstance(root, (ast.List, ast.Tuple)) else [root]
        for node in nodes:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            args: dict[str, Any] = {}
            try:
                for keyword in node.keywords:
                    if keyword.arg is None:
                        raise ValueError("unsupported kwargs")
                    args[keyword.arg] = ast.literal_eval(keyword.value)
                if node.args:
                    raise ValueError("positional arguments unsupported")
            except (ValueError, TypeError, SyntaxError):
                continue
            parsed.append({
                "id": f"lfm-text-call-{start_id + len(parsed)}",
                "type": "function",
                "function": {"name": node.func.id, "arguments": json.dumps(args, ensure_ascii=False)},
            })
    return parsed


def _rows(split_name: str) -> list[dict[str, Any]]:
    split_path = hb.BASE / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        rows = split.get(split_name, [])
    else:
        rows = [json.loads(line) for line in (hb.DATA / "financebench_merged.jsonl").read_text().splitlines()]
    out = []
    for row in rows:
        answer = str(row.get("answer") or "").strip()
        if not answer:
            continue
        out.append({
            "question": fe._format_question(row),
            "answer": [answer],
            "doc": row.get("doc_name", ""),
            "company": row.get("company", ""),
            "financebench_id": row.get("financebench_id", ""),
        })
    return out


def _finish_metadata(args: dict[str, Any]) -> dict[str, Any]:
    page = args.get("evidence_page", -1)
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = -1
    return {"evidence_document": str(args.get("evidence_document", "")), "evidence_page": page}


def run_teacher(index: hb.StructuredIndex, row: dict[str, Any], model: str, max_turns: int, base_url: str = "", api_key: str = "", system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 2048, include_finish: bool = True, seed: int | None = None) -> dict[str, Any]:
    from openai import OpenAI

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = api_key or "none"
    else:
        kwargs["api_key"] = openai_key()
    client = OpenAI(**kwargs)
    # OpenAI uses Responses; OpenAI-compatible gateways use Chat Completions.
    use_responses = not bool(base_url)

    tools_spec = [
        {"name": s["name"], "description": s.get("description", ""), "parameters": s["parameters"]}
        for s in tool_specs(fe.Bm25Tool(index))
        if include_finish or s["name"] != "finish"
    ]
    if use_responses:
        tools = [{"type": "function", **t} for t in tools_spec]
    else:
        tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools_spec]

    initial_input = [
        {"role": "system", "content": system_prompt or (TEACHER_SYSTEM if include_finish else LEGACY_TEACHER_SYSTEM)},
        {"role": "user", "content": row["question"]},
    ]
    calls: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    final = ""
    finish_metadata: dict[str, Any] = {}
    termination_reason = "max_turns"
    OBS_CAP = 4000
    if use_responses:
        response = client.responses.create(
            model=model,
            input=initial_input,
            tools=tools,
            tool_choice="auto",
            reasoning={"effort": "medium"},
            max_output_tokens=2048,
            store=False,
        )
        conversation = list(initial_input)
        for _ in range(max_turns):
            output = response.output
            if not output:
                break
            if output[-1].type == "message":
                if output[-1].content:
                    final = " ".join(
                        getattr(part, "text", "") for part in output[-1].content
                        if getattr(part, "type", "") == "output_text"
                    )
                trace.append({"role": "assistant", "content": final})
                termination_reason = "assistant"
                break
            tool_outputs: list[dict[str, Any]] = []
            turn_calls: list[dict[str, Any]] = []
            finish_answer = ""
            for item in output:
                if item.type == "function_call":
                    args = json.loads(item.arguments or "{}")
                    call = {"name": item.name, "arguments": args, "call_id": item.call_id}
                    calls.append(call)
                    turn_calls.append(call)
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": execute_local(index, item.name, args)[:OBS_CAP],
                        }
                    )
                    if item.name == "finish":
                        finish_answer = str(args.get("answer", "")).strip()
                        finish_metadata = _finish_metadata(args)
            if not tool_outputs:
                break
            previous_output = [item.model_dump(exclude_none=True) for item in output]
            conversation.extend(previous_output)
            conversation.extend(tool_outputs)
            trace.append({"role": "assistant", "tool_calls": turn_calls})
            trace.extend([{"role": "tool", "call_id": item["call_id"], "content": item["output"]} for item in tool_outputs])
            if finish_answer:
                final = f"Answer: {finish_answer}"
                termination_reason = "finish"
                break
            response = client.responses.create(
                model=model,
                input=conversation,
                tools=tools,
                tool_choice="auto",
                reasoning={"effort": "medium"},
                max_output_tokens=max_tokens,
                store=False,
            )
    else:
        # Chat Completions path (DeepInfra, vLLM, etc.)
        messages = list(initial_input)
        for turn_index in range(max_turns):
            tool_choice = "auto"
            if include_finish and turn_index == max_turns - 1:
                tool_choice = {"type": "function", "function": {"name": "finish"}}
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            msg = response.choices[0].message
            text_calls = []
            if not msg.tool_calls:
                text_calls = _lfm_text_tool_calls(
                    chr(10).join(str(getattr(msg, key, "") or "") for key in ("reasoning_content", "content")),
                    len(calls),
                )
            parsed_tool_calls = list(msg.tool_calls or []) or text_calls
            if parsed_tool_calls:
                turn_calls: list[dict[str, Any]] = []
                tool_observations: list[dict[str, Any]] = []
                finish_answer = ""
                for tc in parsed_tool_calls:
                    if hasattr(tc, "function"):
                        call_id = tc.id
                        name = tc.function.name
                        raw_arguments = tc.function.arguments
                    else:
                        call_id = tc["id"]
                        name = tc["function"]["name"]
                        raw_arguments = tc["function"]["arguments"]
                    args = json.loads(raw_arguments or "{}")
                    call = {"name": name, "arguments": args, "call_id": call_id}
                    calls.append(call)
                    turn_calls.append(call)
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": raw_arguments}}],
                    })
                    obs = execute_local(index, name, args)[:OBS_CAP]
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
                    tool_observations.append({"call_id": call_id, "content": obs})
                    if name == "finish":
                        finish_answer = str(args.get("answer", "")).strip()
                        finish_metadata = _finish_metadata(args)
                trace.append({"role": "assistant", "tool_calls": turn_calls})
                trace.extend({"role": "tool", **item} for item in tool_observations)
                if finish_answer:
                    final = f"Answer: {finish_answer}"
                    termination_reason = "finish"
                    break
                continue
            # No tool calls: final answer
            final = msg.content or ""
            trace.append({"role": "assistant", "content": final})
            termination_reason = "assistant"
            break
    return {"answer_text": final, "tool_calls": calls, "trace": trace, "termination_reason": termination_reason, "finish_metadata": finish_metadata, "openai_response_id": getattr(response, "id", None) if hasattr(response, "id") else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--base-url", default="", help="OpenAI-compatible endpoint (e.g. vLLM). If set, the local OPENAI_API_KEY is not used.")
    ap.add_argument("--api-key", default="", help="API key for the --base-url endpoint.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=-1, help="Optional sampling seed for OpenAI-compatible chat endpoints.")
    ap.add_argument("--reward-mode", choices=("answer", "grounded"), default="answer", help="Primary recorded reward; grounded combines answer quality with trajectory evidence support.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail-out", default="")
    ap.add_argument("--resume-from", default="", help="JSONL of previously saved successful records; those ids are skipped.")
    ap.add_argument("--ids-file", default="", help="Optional file with one financebench_id per line.")
    ap.add_argument("--reward-threshold", type=float, default=0.9)
    ap.add_argument("--max-attempts", type=int, default=1)
    ap.add_argument("--no-finish", action="store_true", help="Run matched legacy control without exposing the finish tool.")
    args = ap.parse_args()

    if not args.base_url and not openai_key():
        raise SystemExit("OPENAI_API_KEY is not available")

    rows = _rows(args.split)
    if args.ids_file:
        wanted = {line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip()}
        rows = [row for row in rows if row["financebench_id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    resume_ids: set[str] = set()
    if args.resume_from:
        p = Path(args.resume_from)
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    resume_ids.add(json.loads(line).get("financebench_id", ""))

    index = hb.build_index()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path = Path(args.fail_out) if args.fail_out else out_path.with_suffix(".failed.jsonl")
    fail_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    for i, row in enumerate(rows, 1):
        fid = row["financebench_id"]
        if fid in resume_ids:
            print(f"[{i}/{len(rows)}] {fid} already saved, skipping", flush=True)
            continue
        record = None
        trace: list[dict[str, Any]] = []
        last_error = None
        for attempt in range(1, max(1, args.max_attempts) + 1):
            started = time.time()
            rec = None
            try:
                rec = run_teacher(index, row, args.model, args.max_turns, args.base_url, args.api_key, temperature=args.temperature, max_tokens=args.max_tokens, include_finish=not args.no_finish, seed=(None if args.seed < 0 else args.seed))
                answer_reward, answer_parts = hb.score_answer(row["answer"][0], rec.get("answer_text", ""))
                evidence_reward, evidence_parts = hb.score_evidence(row["answer"][0], rec.get("answer_text", ""), rec.get("trace") or [])
                combined_reward = hb.grounded_reward(answer_reward, evidence_reward)
                reward = combined_reward if args.reward_mode == "grounded" else answer_reward
                trace = rec.get("trace") or []
                finish_metadata = rec.get("finish_metadata") or {}
                record = {
                    "financebench_id": fid,
                    "question": row["question"],
                    "gold": row["answer"][0],
                    "teacher_model": args.model,
                    "sampling_seed": None if args.seed < 0 else args.seed,
                    "temperature": args.temperature,
                    "tool_calls": rec["tool_calls"],
                    "trace": trace,
                    "answer_text": rec["answer_text"],
                    "reward": reward,
                    "answer_reward": answer_reward,
                    "evidence_reward": evidence_reward,
                    "grounded_reward": combined_reward,
                    "reward_mode": args.reward_mode,
                    "reward_parts": {**answer_parts, **evidence_parts},
                    "auto_verified": reward >= args.reward_threshold,
                    "wall_s": round(time.time() - started, 2),
                    "openai_response_id": rec.get("openai_response_id"),
                    "termination_reason": rec.get("termination_reason", "max_turns"),
                    "finish_ok": float(rec.get("termination_reason") == "finish"),
                    "finish_metadata": finish_metadata,
                }
                break
            except Exception as exc:
                last_error = repr(exc)
                if rec is not None:
                    trace = rec.get("trace") or []
            if record is None:
                with fail_path.open("a") as f:
                    f.write(json.dumps({
                        "financebench_id": fid,
                        "question": row["question"],
                        "gold": row["answer"][0],
                        "error": last_error,
                        "trace": trace,
                        "answer_text": rec.get("answer_text") if rec is not None else None,
                    }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] {fid} attempt {attempt} failed: {last_error}", flush=True)
        if record is not None:
            n_ok += 1
            with out_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] {fid} saved reward={record['reward']:.2f} turns={len(record['tool_calls'])}", flush=True)
        else:
            n_fail += 1
            print(f"[{i}/{len(rows)}] {fid} FAILED: {last_error}", flush=True)
    print(json.dumps({"n": len(rows), "ok": n_ok, "failed": n_fail, "out": str(out_path), "fail_out": str(fail_path)}))


if __name__ == "__main__":
    main()
