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
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import financebench_harness as hb
from teacher_runtime import TeacherHarnessSession, submission_answer_text

LEGACY_TEACHER_SYSTEM = (
    "\n\nIMPORTANT: This is a supervised teacher trajectory. Use the tools to find evidence in the filings, then provide a concise final Answer in plain text. Do not use or expect a finish/submission tool. Use only information from retrieved evidence."
)

TEACHER_SYSTEM = (
    "\n\nIMPORTANT: This is a supervised teacher trajectory. Use the tools to find "
    "evidence in the filings, then use the current typed finish contract exactly once. "
    "Citations must use receipt_id, document_id, and page from a strong read or grep result. "
    "For a derived numeric answer, compute using retrieved values and optionally include derivation. "
    "Numeric values must be decimal strings with unit and scale; "
    "text answers must use answer_text. Do not continue searching after an accepted finish."
)


def openai_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "")


def resolve_api_key(explicit: str, env_name: str) -> str:
    return explicit or os.environ.get(env_name, "")


def chat_completion_options(model: str, temperature: float, seed: int | None) -> dict[str, Any]:
    options: dict[str, Any] = {"seed": seed}
    if model != "openai/openai/gpt-5.6-terra":
        options["temperature"] = temperature
    return options


def is_rate_limit_error(exc: BaseException) -> bool:
    return type(exc).__name__ == "RateLimitError" or "429" in str(exc)


def teacher_client(base_url: str, api_key: str, request_timeout_s: float = 120.0, max_retries: int = 1):
    import httpx
    from openai import OpenAI

    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(request_timeout_s, connect=min(5.0, request_timeout_s)),
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = api_key or "none"
    else:
        kwargs["api_key"] = openai_key()
    return OpenAI(**kwargs)


def parse_tool_arguments(raw: str | None) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(raw or "{}")
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be an object")
        return value, ""
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}, "Tool arguments were not valid JSON; correct them and retry."


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


def _rows(split_name: str, split_path: str | Path | None = None) -> list[dict[str, Any]]:
    split_path = Path(split_path) if split_path else hb.BASE / "split.json"
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
        question = str(row["question"]).strip()
        company = row.get("company") or row.get("doc_name", "").split("_")[0]
        if company and company.lower() not in question.lower():
            question = f"About {company}: {question}"
        out.append({
            "question": question,
            "answer": [answer],
            "doc": row.get("doc_name", ""),
            "company": row.get("company", ""),
            "financebench_id": row.get("financebench_id", ""),
        })
    return out


def split_documents(split_path: str | Path) -> list[str]:
    split = json.loads(Path(split_path).read_text())
    return sorted({
        str(row.get("doc_name", "")).strip()
        for rows in split.values()
        if isinstance(rows, list)
        for row in rows
        if str(row.get("doc_name", "")).strip()
    })


def completed_trace_ids(paths: list[str | Path]) -> set[str]:
    completed: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("trace") and record.get("financebench_id"):
                completed.add(record["financebench_id"])
    return completed


def teacher_target_is_resolved(target: dict[str, Any]) -> bool:
    return target.get("adjudication_status", "resolved") == "resolved"


def _finish_metadata(args: dict[str, Any]) -> dict[str, Any]:
    page = args.get("evidence_page", -1)
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = -1
    return {"evidence_document": str(args.get("evidence_document", "")), "evidence_page": page}


def run_teacher(index: hb.StructuredIndex, row: dict[str, Any], model: str, max_turns: int, base_url: str = "", api_key: str = "", system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 2048, include_finish: bool = True, seed: int | None = None, target: dict[str, Any] | None = None, judge: Any = None, judge_confidence_threshold: float = 0.85, request_timeout_s: float = 120.0, max_retries: int = 1) -> dict[str, Any]:
    client = teacher_client(base_url, api_key, request_timeout_s, max_retries)
    # OpenAI uses Responses; OpenAI-compatible gateways use Chat Completions.
    use_responses = not bool(base_url)

    session = TeacherHarnessSession(index, max_turns=max_turns)
    tools_spec = session.tool_specs(include_finish=include_finish)
    if use_responses:
        tools = [{"type": "function", **t} for t in tools_spec]
    else:
        tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools_spec]

    initial_input = [
        {"role": "system", "content": system_prompt or session.system_prompt + ("" if include_finish else LEGACY_TEACHER_SYSTEM)},
        {"role": "user", "content": row["question"]},
    ]
    calls: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    final = ""
    finish_metadata: dict[str, Any] = {}
    generation_metadata = []
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
            session.start_turn()
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
            accepted_finish = False
            for item in output:
                if item.type == "function_call":
                    args, parse_error = parse_tool_arguments(item.arguments)
                    call = {"name": item.name, "arguments": args, "call_id": item.call_id}
                    if parse_error:
                        call["parse_error"] = parse_error
                    calls.append(call)
                    turn_calls.append(call)
                    if parse_error:
                        tool_output = json.dumps({"error": parse_error})
                        should_stop = False
                    else:
                        result = session.execute(item.name, args, item.call_id)
                        tool_output = result.content
                        should_stop = result.should_stop
                    tool_outputs.append({"type": "function_call_output", "call_id": item.call_id, "output": tool_output})
                    if item.name == "finish":
                        accepted_finish = should_stop and session.accepted is not None
            if not tool_outputs:
                break
            previous_output = [item.model_dump(exclude_none=True) for item in output]
            conversation.extend(previous_output)
            conversation.extend(tool_outputs)
            trace.append({"role": "assistant", "tool_calls": turn_calls})
            trace.extend([{"role": "tool", "call_id": item["call_id"], "content": item["output"]} for item in tool_outputs])
            if accepted_finish:
                final = submission_answer_text(session.accepted)
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
            session.start_turn()
            tool_choice = "auto"
            if include_finish and turn_index == max_turns - 1:
                tool_choice = {"type": "function", "function": {"name": "finish"}}
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                **chat_completion_options(model, temperature, seed),
            )
            msg = response.choices[0].message
            usage = getattr(response, 'usage', None)
            generation_metadata.append({'finish_reason':getattr(response.choices[0],'finish_reason',None),
                'usage':usage.model_dump() if hasattr(usage,'model_dump') else vars(usage) if hasattr(usage,'__dict__') else None})
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
                accepted_finish = False
                assistant_tool_calls: list[dict[str, Any]] = []
                for tc in parsed_tool_calls:
                    if hasattr(tc, "function"):
                        call_id = tc.id
                        name = tc.function.name
                        raw_arguments = tc.function.arguments
                    else:
                        call_id = tc["id"]
                        name = tc["function"]["name"]
                        raw_arguments = tc["function"]["arguments"]
                    args, parse_error = parse_tool_arguments(raw_arguments)
                    call = {"name": name, "arguments": args, "call_id": call_id}
                    if parse_error:
                        call["parse_error"] = parse_error
                    calls.append(call)
                    turn_calls.append(call)
                    assistant_tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": raw_arguments}})
                    if parse_error:
                        obs = json.dumps({"error": parse_error})
                        should_stop = False
                    else:
                        result = session.execute(name, args, call_id)
                        obs = result.content[:OBS_CAP]
                        should_stop = result.should_stop
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
                    tool_observations.append({"call_id": call_id, "content": obs})
                    if name == "finish":
                        accepted_finish = should_stop and session.accepted is not None
                messages.insert(len(messages) - len(tool_observations), {
                    "role": "assistant", "content": msg.content or None, "tool_calls": assistant_tool_calls,
                })
                trace.append({"role": "assistant", "content":msg.content or "", "tool_calls": turn_calls})
                trace.extend({"role": "tool", **item} for item in tool_observations)
                if accepted_finish:
                    final = submission_answer_text(session.accepted)
                    termination_reason = "finish"
                    break
                continue
            # No tool calls: final answer
            final = msg.content or ""
            trace.append({"role": "assistant", "content": final})
            termination_reason = "assistant"
            break
    scoring_target = {**target, 'reference_answer':row.get('answer',target.get('reference_answer'))} if target is not None else None
    score = session.score(scoring_target, question=row["question"], judge=judge,
        judge_confidence_threshold=judge_confidence_threshold) if target is not None else None
    return {"answer_text": final, "tool_calls": calls, "trace": trace, "termination_reason": termination_reason,
        "finish_metadata": finish_metadata, "submission": session.accepted, "score": score, "generation_metadata":generation_metadata,
        "openai_response_id": getattr(response, "id", None) if hasattr(response, "id") else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--split-file", default=os.environ.get("FINANCEBENCH_SPLIT", "artifacts/workshop/split.json"))
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--base-url", default="", help="OpenAI-compatible endpoint (e.g. vLLM). If set, the local OPENAI_API_KEY is not used.")
    ap.add_argument("--api-key", default="", help="API key for the --base-url endpoint.")
    ap.add_argument("--api-key-env", default="TEACHER_ENDPOINT_API_KEY", help="Environment variable holding the endpoint key; avoids exposing it in process arguments.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=-1, help="Optional sampling seed for OpenAI-compatible chat endpoints.")
    ap.add_argument("--reward-mode", choices=("answer", "grounded"), default="answer", help="Primary recorded reward; grounded combines answer quality with trajectory evidence support.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail-out", default="")
    ap.add_argument("--unresolved-out", default="")
    ap.add_argument("--resume-from", default="", help="JSONL of previously saved successful records; those ids are skipped.")
    ap.add_argument("--resume-traces-from", action="append", default=[], help="Additional JSONL whose ids are skipped only when a non-empty trajectory is present; repeatable.")
    ap.add_argument("--ids-file", default="", help="Optional file with one financebench_id per line.")
    ap.add_argument("--reward-threshold", type=float, default=0.9)
    ap.add_argument("--max-attempts", type=int, default=1)
    ap.add_argument("--request-timeout", type=float, default=120.0, help="Per-request endpoint timeout in seconds.")
    ap.add_argument("--endpoint-max-retries", type=int, default=1, help="Transport retries performed by the endpoint client.")
    ap.add_argument("--stop-on-rate-limit", action="store_true", help="Exit with status 75 on HTTP 429 so a supervisor can retry the same id after cooldown.")
    ap.add_argument("--targets", default=os.environ.get("FINANCEBENCH_TARGETS", "results/paper_2026_rl4llm/targets_frozen_rubric-v3.json"))
    ap.add_argument("--no-finish", action="store_true", help="Run matched legacy control without exposing the finish tool.")
    args = ap.parse_args()
    args.api_key = resolve_api_key(args.api_key, args.api_key_env)
    from reward_calculation import RewardConfig, build_judge
    reward_config = RewardConfig.from_env()
    judge = build_judge(reward_config)
    from run_identity import require_current_judge
    require_current_judge(judge)

    if not args.base_url and not openai_key():
        raise SystemExit("OPENAI_API_KEY is not available")

    rows = _rows(args.split, args.split_file)
    targets = json.loads(Path(args.targets).read_text())
    if args.ids_file:
        wanted = {line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip()}
        rows = [row for row in rows if row["financebench_id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    resume_paths = ([args.resume_from] if args.resume_from else []) + args.resume_traces_from
    from run_identity import verify_run_config
    from finance_env import FINANCE_TASK_INSTRUCTIONS, Bm25Tool
    from teacher_runtime import TOOL_NAMES
    import hashlib
    identity={k:getattr(args,k) for k in ('model','base_url','max_turns','max_tokens','temperature','seed','no_finish')}
    identity.update(task_prompt=FINANCE_TASK_INSTRUCTIONS,
                    targets_sha256=hashlib.sha256(Path(args.targets).read_bytes()).hexdigest(),
                    split_sha256=hashlib.sha256(Path(args.split_file).read_bytes()).hexdigest(),
                    tool_specs=[getattr(Bm25Tool(None),name).to_spec() for name in TOOL_NAMES])
    for path in resume_paths:
        if Path(path).exists(): verify_run_config(path,identity)
    verify_run_config(args.out,identity)
    resume_ids = completed_trace_ids(resume_paths)

    index = hb.build_index(doc_names=split_documents(args.split_file))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path = Path(args.fail_out) if args.fail_out else out_path.with_suffix(".failed.jsonl")
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    unresolved_path = Path(args.unresolved_out) if args.unresolved_out else out_path.with_suffix(".unresolved.jsonl")
    unresolved_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    n_unresolved = 0
    for i, row in enumerate(rows, 1):
        fid = row["financebench_id"]
        if fid in resume_ids:
            print(f"[{i}/{len(rows)}] {fid} already saved, skipping", flush=True)
            continue
        target = targets[fid]
        if not teacher_target_is_resolved(target):
            record = {
                "financebench_id": fid,
                "question": row["question"],
                "adjudication_status": target.get("adjudication_status"),
                "adjudication_reason": target.get("adjudication_reason", ""),
            }
            with unresolved_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_unresolved += 1
            print(f"[{i}/{len(rows)}] {fid} unresolved target, preserved without generation", flush=True)
            continue
        record = None
        trace: list[dict[str, Any]] = []
        last_error = None
        for attempt in range(1, max(1, args.max_attempts) + 1):
            started = time.time()
            rec = None
            try:
                rec = run_teacher(index, row, args.model, args.max_turns, args.base_url, args.api_key, temperature=args.temperature, max_tokens=args.max_tokens, include_finish=not args.no_finish, seed=(None if args.seed < 0 else args.seed), target=target,
                    judge=judge, judge_confidence_threshold=reward_config.judge_confidence_threshold,
                    request_timeout_s=args.request_timeout, max_retries=args.endpoint_max_retries)
                exact_score = rec.get("score") or {}
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
                    "submission": rec.get("submission"),
                    "score": exact_score,
                    "generation_metadata": rec.get('generation_metadata',[]),
                }
                gate_ok = bool(exact_score.get("F") == 1 and exact_score.get("A") == 1 and exact_score.get("G") == 1)
                if gate_ok:
                    break
                last_error = f"typed gate failed: F={exact_score.get('F', 0)} A={exact_score.get('A', 0)} G={exact_score.get('G', 0)}"
                record = None
            except Exception as exc:
                last_error = repr(exc)
                if args.stop_on_rate_limit and is_rate_limit_error(exc):
                    print(f"[{i}/{len(rows)}] {fid} rate limited; stopping before advancing", flush=True)
                    raise SystemExit(75)
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
    print(json.dumps({"n": len(rows), "ok": n_ok, "failed": n_fail, "unresolved": n_unresolved,
        "out": str(out_path), "fail_out": str(fail_path), "unresolved_out": str(unresolved_path)}))


if __name__ == "__main__":
    main()
