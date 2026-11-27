#!/usr/bin/env python3
"""Unified FinanceBench stateful-harness runner for Muse Glimmer 30B (Fireworks Serverless).

Supports:
  1. Standalone baseline & checkpoint evaluation via FiretitanSamplingClient
     (exact token-in / token-out parity with Fireworks Serverless Training).
  2. Reusable async multi-turn rollout engine returning both episode records
     and per-turn token/logprob/loss_mask segments for GRPO training.
  3. Live v6 additive scoring (R = 0.15E + F(0.20G + 0.45A + 0.15B + 0.05))
     via Fireworks DeepSeek-V4.1-Flash with disk caching.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

# Point FinanceBench corpus/cache at the canonical rl-for-ai-research artifacts if not overridden.
_DEFAULT_CORPUS = Path.home() / "Documents" / "Research" / "rl-for-ai-research"
os.environ.setdefault("FINANCEBENCH_FILINGS", str(_DEFAULT_CORPUS / "filings"))
os.environ.setdefault("FINANCEBENCH_CACHE", str(_DEFAULT_CORPUS / "artifacts" / "page_cache"))

import financebench_harness as hb
from component_reward import ComponentJudge
from eval_stateful_harness import (
    SUBMIT_REMINDER,
    MinimalHarnessSession,
    pack_conversation,
    selected_rows,
    system_prompt,
)
from generate_teacher_traces import (
    TeacherHarnessSession,
    split_documents,
    parse_tool_arguments,
    submission_answer_text,
)
from live_additive_reward import LiveAdditiveJudge
from persistent_state import ResearchMemory
from harness_v2_patches import patch_system_prompt, patch_tool_observation
from financeharness_v3 import run_python_sandbox
from provider_tool_calls import parse_provider_tool_calls

from fireworks.training.sdk import FiretitanServiceClient, types
from transformers import AutoTokenizer
from training.renderer import get_renderer
from training.renderer.base import ToolCall


BASE_MODEL_DEFAULT = "accounts/fireworks/models/muse-glimmer-30b"
TOKENIZER_DEFAULT = "meta-models/Muse-Glimmer-30B"
RENDERER_DEFAULT = "muse_glimmer"
SERVERLESS_TRAINING_URL = "https://api.fireworks.ai/training/v1/serverless"
JUDGE_ENDPOINT_DEFAULT = "https://api.fireworks.ai/inference/v1/chat/completions"
JUDGE_MODEL_DEFAULT = "accounts/fireworks/models/deepseek-v4p1-flash"


def _extract_text_and_thinking(parsed_msg: dict[str, Any]) -> tuple[str, str]:
    """Extract visible text and thinking text from MuseGlimmerRenderer.parse_response output."""
    content = parsed_msg.get("content")
    if isinstance(content, str):
        return content, ""
    if not isinstance(content, list):
        return "", ""
    visible_parts: list[str] = []
    thinking_parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            ptype = part.get("type")
            if ptype == "thinking":
                thinking_parts.append(str(part.get("thinking") or ""))
            elif ptype == "text":
                visible_parts.append(str(part.get("text") or ""))
        elif isinstance(part, str):
            visible_parts.append(part)
    return "\n".join(p for p in visible_parts if p), "\n".join(p for p in thinking_parts if p)


def _to_muse_messages(
    renderer: Any,
    tool_specs: list[dict[str, Any]],
    packed_messages: list[dict[str, Any]],
    cid_to_name: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert OpenAI-style packed messages from ResearchMemory into MuseGlimmerRenderer messages."""
    sys_content = ""
    rest = packed_messages
    if packed_messages and packed_messages[0].get("role") == "system":
        sys_content = str(packed_messages[0].get("content") or "")
        rest = packed_messages[1:]

    out: list[dict[str, Any]] = list(
        renderer.create_conversation_prefix_with_tools(tool_specs, sys_content)
    )
    for m in rest:
        role = m.get("role")
        if role == "user":
            out.append({"role": "user", "content": str(m.get("content") or "")})
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            converted_tcs: list[ToolCall] = []
            for tc in tcs:
                if isinstance(tc, ToolCall):
                    converted_tcs.append(tc)
                elif isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    name = str(fn.get("name") or "")
                    args = fn.get("arguments")
                    if not isinstance(args, str):
                        args = json.dumps(args or {}, ensure_ascii=False)
                    cid = str(tc.get("id") or f"{name}:0")
                    converted_tcs.append(
                        ToolCall(
                            type="function",
                            id=cid,
                            function=ToolCall.FunctionBody(name=name, arguments=args),
                        )
                    )
            msg_obj: dict[str, Any] = {"role": "assistant", "content": str(m.get("content") or "")}
            if converted_tcs:
                msg_obj["tool_calls"] = converted_tcs
            out.append(msg_obj)
        elif role == "tool":
            cid = str(m.get("tool_call_id") or "")
            tname = str(m.get("name") or cid_to_name.get(cid) or "tool")
            out.append(
                {
                    "role": "tool",
                    "name": tname,
                    "tool_call_id": cid,
                    "content": str(m.get("content") or ""),
                }
            )
    return out


async def run_muse_episode(
    index: hb.StructuredIndex,
    row: dict[str, Any],
    sampler: Any,
    renderer: Any,
    *,
    max_tool_calls: int = 30,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.95,
    submission_contract: bool = True,
    require_citations: bool = True,
    recent_turn_policy: str = "complete",
    harness_mode: str = "full",
    submit_reminders: int = 1,
    max_seq_len: int = 24576,
    improved_harness: bool = False,
    verbose: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one FinanceBench episode with Muse Glimmer via FiretitanSamplingClient.

    Returns (record_dict, segments_list) where each segment has:
      tokens, logprobs, loss_mask, finish_reason, routing_matrices, text
    """
    session_class = TeacherHarnessSession if harness_mode == "full" else MinimalHarnessSession
    session = session_class(
        index,
        max_turns=max_tool_calls,
        budget_cues=False,
        require_citations=require_citations,
    )
    memory = ResearchMemory(row["question"], recent_turn_policy=recent_turn_policy)
    sys_text = system_prompt(submission_contract, harness_mode)
    if improved_harness:
        sys_text = patch_system_prompt(sys_text)
    initial = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": row["question"]},
    ]
    tool_specs = copy.deepcopy(session.tool_specs())
    for spec in tool_specs:
        if spec.get("name") == "finish":
            spec["description"] = "Submit the final answer and source citations."
    if improved_harness and not any(s.get("name") == "run_python" for s in tool_specs):
        tool_specs.append({
            "name": "run_python",
            "description": "Execute pure Python code (math, statistics, Decimal pre-imported; 2s timeout) for accurate multi-step financial arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        })

    turns: list[list[dict[str, Any]]] = []
    trace: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    cid_to_name: dict[str, str] = {}
    seen_call_sigs: set[str] = set()

    reminders_used = 0
    termination = "tool_call_limit"
    answer = ""
    error: str | None = None
    started = time.time()
    stop_seqs = renderer.get_stop_sequences()

    try:
        while len(calls) < max_tool_calls:
            session.start_turn()
            packed = (
                memory.pack(initial, turns)
                if harness_mode == "full"
                else pack_conversation(initial, turns)
            )
            muse_msgs = _to_muse_messages(renderer, tool_specs, packed, cid_to_name)
            prompt = renderer.build_generation_prompt(muse_msgs)
            prompt_tokens = prompt.to_ints()

            remaining_ctx = max_seq_len - len(prompt_tokens)
            if remaining_ctx < 128:
                termination = "context_limit"
                break
            turn_max_tokens = min(max_tokens, remaining_ctx)

            sampling_params = types.SamplingParams(
                max_tokens=turn_max_tokens,
                temperature=temperature,
                top_p=top_p if temperature > 0 else 1.0,
                stop=stop_seqs,
            )

            # Support both async (sample_async) and sync (sample().result()) clients.
            if hasattr(sampler, "sample_async"):
                res = await sampler.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
            else:
                res = await asyncio.to_thread(
                    lambda p=prompt, sp=sampling_params: sampler.sample(
                        prompt=p, num_samples=1, sampling_params=sp
                    ).result()
                )

            seq = res.sequences[0]
            comp_tokens = list(seq.tokens)
            comp_logprobs = list(seq.logprobs) if seq.logprobs is not None else [0.0] * len(comp_tokens)
            finish_reason = str(getattr(seq, "stop_reason", None) or "stop")
            routing_matrices = getattr(seq, "routing_matrices", None)

            parsed_msg, parse_term = renderer.parse_response(comp_tokens)
            visible_text, thinking_text = _extract_text_and_thinking(parsed_msg)

            segments.append(
                {
                    "tokens": prompt_tokens + comp_tokens,
                    "logprobs": [0.0] * len(prompt_tokens) + comp_logprobs,
                    "loss_mask": [0] * len(prompt_tokens) + [1] * len(comp_tokens),
                    "finish_reason": finish_reason,
                    "routing_matrices": routing_matrices,
                    "text": visible_text,
                }
            )
            metadata.append(
                {
                    "finish_reason": finish_reason,
                    "parse_termination": str(parse_term),
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": len(comp_tokens),
                    "reasoning_content": thinking_text[:2000] if thinking_text else None,
                }
            )

            proposed = list(parsed_msg.get("tool_calls") or [])
            if not proposed:
                provider_text = "\n".join(p for p in (thinking_text, visible_text) if p)
                proposed = parse_provider_tool_calls(provider_text, len(calls))

            if not proposed:
                answer = visible_text
                trace.append({"role": "assistant", "content": answer})
                if (
                    reminders_used < submit_reminders
                    and finish_reason != "length"
                    and answer.strip()
                    and len(calls) < max_tool_calls
                ):
                    reminders_used += 1
                    trace.append({"role": "user", "content": SUBMIT_REMINDER, "harness_reminder": True})
                    turns.append(
                        [
                            {"role": "assistant", "content": answer},
                            {"role": "user", "content": SUBMIT_REMINDER},
                        ]
                    )
                    continue
                termination = (
                    "generation_token_limit"
                    if finish_reason == "length"
                    else "assistant_without_finish"
                )
                break

            executed, observations, api_calls = [], [], []
            terminal = False
            for tc in proposed:
                if len(calls) >= max_tool_calls or terminal:
                    break
                if hasattr(tc, "function"):
                    name = tc.function.name
                    raw = tc.function.arguments
                    cid = tc.id or f"{name}:{len(calls)}"
                else:
                    name = tc["function"]["name"]
                    raw = tc["function"]["arguments"]
                    cid = tc.get("id") or f"{name}:{len(calls)}"
                cid_to_name[cid] = name
                args, parse_error = parse_tool_arguments(raw)
                call = {"name": name, "arguments": args, "call_id": cid}
                if parse_error:
                    call["parse_error"] = parse_error
                    call["raw_arguments"] = raw
                calls.append(call)
                executed.append(call)
                api_calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": "{}"
                            if parse_error
                            else (raw if isinstance(raw, str) else json.dumps(raw)),
                        },
                    }
                )
                if parse_error:
                    content, stop = json.dumps({"error": parse_error}), False
                elif name == "run_python":
                    content = await asyncio.to_thread(run_python_sandbox, args.get("code", ""))
                    stop = False
                else:
                    result = await asyncio.to_thread(session.execute, name, args, cid)
                    content, stop = result.content, result.should_stop
                    if improved_harness:
                        content = patch_tool_observation(name, args, content, seen_call_sigs, len(session.state.receipts))
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "call_id": cid,
                        "name": name,
                        "content": content,
                    }
                )
                if stop:
                    terminal = True
                    termination = "finish" if session.accepted is not None else "terminal_rejected"

            trace.append(
                {
                    "role": "assistant",
                    "content": visible_text,
                    "tool_calls": executed,
                    "proposed_call_count": len(proposed),
                }
            )
            trace.extend(observations)
            turn = [{"role": "assistant", "content": visible_text or None, "tool_calls": api_calls}]
            turn.extend(
                {k: obs[k] for k in ("role", "tool_call_id", "name", "content")}
                for obs in observations
            )
            turns.append(turn)
            memory.observe(executed, observations, session.state.receipts)

            if verbose:
                print(
                    json.dumps(
                        {
                            "question": row["financebench_id"],
                            "turn": len(metadata),
                            "calls": len(calls),
                            "receipts": len(session.state.receipts),
                            "last_tool": executed[-1]["name"],
                        }
                    ),
                    flush=True,
                )
            if terminal:
                answer = submission_answer_text(session.accepted)
                break
    except Exception as exc:
        termination = "error"
        error = f"{type(exc).__name__}: {str(exc)[:180]}"

    record = {
        "financebench_id": row["financebench_id"],
        "question": row["question"],
        "gold": row["answer"][0] if isinstance(row.get("answer"), list) else row.get("answer", ""),
        "submission": session.accepted,
        "answer_text": answer,
        "termination_reason": termination,
        "error": error,
        "submit_reminders_used": reminders_used,
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "turn_count": len(metadata),
        "trace": trace,
        "generation_metadata": metadata,
        "source_receipts": copy.deepcopy(session.state.receipts),
        "wall_s": round(time.time() - started, 2),
        "score": {"F": int(session.accepted is not None)},
    }
    return record, segments


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    finishes = sum(1 for r in records if r.get("submission") is not None)
    scored = [r["score"] for r in records if isinstance(r.get("score"), dict) and r["score"].get("reward") is not None]
    term_counts: dict[str, int] = {}
    for r in records:
        t = r.get("termination_reason", "unknown")
        term_counts[t] = term_counts.get(t, 0) + 1

    def _mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n": n,
        "scored": len(scored),
        "unresolved": n - len(scored),
        "accepted_finishes": finishes,
        "finish_rate": round(finishes / n, 4),
        "mean_reward": _mean([s["reward"] for s in scored]),
        "mean_E": _mean([s["E"] for s in scored if s.get("E") is not None]),
        "mean_G": _mean([s["G"] for s in scored if s.get("G") is not None]),
        "mean_A": _mean([s["A"] for s in scored if s.get("A") is not None]),
        "mean_B": _mean([s["B"] for s in scored if s.get("B") is not None]),
        "accuracy_A1": round(sum(1 for s in scored if s.get("F") == 1 and s.get("A") == 1) / len(scored), 4) if scored else None,
        "grounded_accuracy": round(sum(1 for s in scored if s.get("F") == 1 and s.get("A") == 1 and s.get("G") == 1) / len(scored), 4) if scored else None,
        "mean_tool_calls": round(sum(r.get("tool_call_count", 0) for r in records) / n, 2),
        "mean_turns": round(sum(r.get("turn_count", 0) for r in records) / n, 2),
        "mean_wall_s": round(sum(r.get("wall_s", 0.0) for r in records) / n, 2),
        "terminations": term_counts,
    }


async def run_eval_cli(args: argparse.Namespace) -> None:
    api_key = Path(args.api_key_file).read_text().strip() if args.api_key_file else os.environ["FIREWORKS_API_KEY"]
    os.environ["FIREWORKS_API_KEY"] = api_key

    rows = selected_rows(args.split_file, args.split)
    if args.limit:
        rows = rows[: args.limit]

    all_docs = split_documents(args.split_file)
    print(f"Building StructuredIndex over {len(all_docs)} filings (eval rows={len(rows)}, split={args.split})...", flush=True)
    index = hb.build_index(doc_names=all_docs, use_cache=True)

    print(f"Loading tokenizer {args.tokenizer_model} and renderer {args.renderer_name}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)
    renderer = get_renderer(args.renderer_name, tok)

    service = FiretitanServiceClient(api_key=api_key, base_url=SERVERLESS_TRAINING_URL)
    if args.model_path:
        print(f"Creating FiretitanSamplingClient for checkpoint: {args.model_path}", flush=True)
        sampler = service.create_sampling_client(model_path=args.model_path, tokenizer=tok)
    else:
        print(f"Creating FiretitanSamplingClient for base model: {args.base_model}", flush=True)
        sampler = service.create_sampling_client(base_model=args.base_model, tokenizer=tok)

    judge = None
    if args.score:
        comp_judge = ComponentJudge(
            endpoint=args.judge_endpoint,
            model=args.judge_model,
            key=api_key,
            cache_dir=args.judge_cache,
        )
        judge = LiveAdditiveJudge(comp_judge)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists() and not args.overwrite:
        for line in results_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                completed[rec["financebench_id"]] = rec
        print(f"Resuming: {len(completed)}/{len(rows)} already completed.", flush=True)

    sem = asyncio.Semaphore(args.workers)
    write_lock = asyncio.Lock()

    async def _worker(row: dict[str, Any]) -> None:
        fid = row["financebench_id"]
        if fid in completed:
            return
        async with sem:
            record, segs = await run_muse_episode(
                index,
                row,
                sampler,
                renderer,
                max_tool_calls=args.max_tool_calls,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                submission_contract=True,
                require_citations=True,
                recent_turn_policy="complete",
                harness_mode="full",
                submit_reminders=args.submit_reminders,
                max_seq_len=args.max_seq_len,
                improved_harness=args.improved_harness,
                verbose=args.verbose,
            )
            record["num_segments"] = len(segs)
            record["trainable_tokens"] = sum(sum(s["loss_mask"]) for s in segs)
            if judge is not None:
                sc = await judge.score(
                    fid,
                    row["question"],
                    record["gold"],
                    record["submission"],
                    record["source_receipts"],
                )
                record["score"] = sc
            async with write_lock:
                completed[fid] = record
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                sc = record.get("score", {})
                print(
                    f"[DONE {len(completed)}/{len(rows)}] {fid} | term={record['termination_reason']} "
                    f"| calls={record['tool_call_count']} | R={sc.get('reward')} "
                    f"(E={sc.get('E')} G={sc.get('G')} A={sc.get('A')} B={sc.get('B')} F={sc.get('F')}) "
                    f"| {record['wall_s']}s",
                    flush=True,
                )

    try:
        await asyncio.gather(*[_worker(r) for r in rows])
    finally:
        sampler.close()

    ordered = [completed[r["financebench_id"]] for r in rows if r["financebench_id"] in completed]
    stats = summarize_records(ordered)
    stats["model"] = args.model_path or args.base_model
    stats["split"] = args.split
    stats["temperature"] = args.temperature
    stats["max_tool_calls"] = args.max_tool_calls
    stats["max_tokens"] = args.max_tokens
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default="split.json")
    ap.add_argument("--split", choices=("train", "dev", "eval", "all"), default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--model-path", default=None, help="Optional Fireworks serverless sampler checkpoint path")
    ap.add_argument("--tokenizer-model", default=TOKENIZER_DEFAULT)
    ap.add_argument("--renderer-name", default=RENDERER_DEFAULT)
    ap.add_argument("--api-key-file", default=str(Path.home() / ".config" / "fireworks" / "api_key"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-tool-calls", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-seq-len", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--submit-reminders", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--judge-endpoint", default=JUDGE_ENDPOINT_DEFAULT)
    ap.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    ap.add_argument("--judge-cache", default="run/judge_cache")
    ap.add_argument("--improved-harness", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    asyncio.run(run_eval_cli(args))


if __name__ == "__main__":
    main()
