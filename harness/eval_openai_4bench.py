#!/usr/bin/env python3
"""
Uniform 4-Benchmark OpenAI-Compatible Evaluator for Financeharness v3.5.
Evaluates any local/remote OpenAI-compatible model (e.g. Qwen3.5-4B on vLLM)
across FinanceBench, FinAgentBench, FinRetrieval, and FinSearchComp using the
exact same tools, UniformResearchMemory, forced final synthesis turn, and
DeepSeek-V4.1-Flash judges as Muse Glimmer 30B.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import financebench_harness as hb
import harness_v2_patches as v2p
from component_reward import ComponentJudge
from financeharness_v3 import UniformResearchMemory
from muse_glimmer_runner import (
    LiveAdditiveJudge,
    TeacherHarnessSession,
    parse_provider_tool_calls,
    parse_tool_arguments,
    selected_rows,
    split_documents,
    summarize_records as summarize_fb_records,
)
from benchmarks.multibench_env import (
    FinRetrievalDB,
    MultiBenchJudge,
    MultiBenchSession,
)
from eval_multibench import summarize_multibench

def load_multibench_rows(benchmarks: list[str], split: str = "eval", limit_per_benchmark: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bm in benchmarks:
        fp = ROOT / "data/multibench" / f"{bm}_{split}.jsonl"
        loaded = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
        rows.extend(loaded[:limit_per_benchmark])
    return rows

SUBMIT_NUDGE = "You have not called `finish` yet. Submit your verified final answer now using the `finish` tool."


def _openai_chat_sync(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 3072,
    timeout: int = 180,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = [{"type": "function", "function": t} for t in tools]
        payload["tool_choice"] = "auto"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_packed_messages(packed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in packed:
        role = m["role"]
        if role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            out.append(msg)
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", "call_0"),
                "name": m.get("name", "tool"),
                "content": str(m.get("content", "")),
            })
        else:
            out.append({"role": role, "content": str(m.get("content", ""))})
    return out



async def _exec_tool(session: Any, benchmark: str, name: str, args: dict[str, Any], cid: str) -> tuple[str, bool]:
    if benchmark == "financebench":
        res = await asyncio.to_thread(session.execute, name, args, cid)
        return str(res.content), bool(res.should_stop)
    return await asyncio.to_thread(session.execute, name, args)


async def run_openai_episode(
    *,
    benchmark: str,
    row: dict[str, Any],
    session: Any,
    base_url: str,
    api_key: str,
    model: str,
    max_tool_calls: int,
    max_tokens: int = 3072,
    temperature: float = 0.0,
    submit_reminders: int = 1,
    force_finish: bool = True,
) -> dict[str, Any]:
    question = row["question"]
    memory = UniformResearchMemory(question)
    sys_prompt = session.system_prompt() if callable(getattr(session, "system_prompt", None)) else getattr(session, "system_prompt", "")
    if benchmark == "financebench":
        sys_prompt = v2p.patch_system_prompt(sys_prompt)
    initial = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]
    tool_specs = session.tool_specs() if callable(getattr(session, "tool_specs", None)) else list(session.tools)
    # Normalize OpenAI-wrapped specs if TeacherHarnessSession returns {"type": "function", "function": ...}
    norm_specs = [t["function"] if isinstance(t, dict) and "function" in t else t for t in tool_specs]

    turns: list[list[dict[str, Any]]] = []
    calls: list[dict[str, Any]] = []
    reminders_used = 0
    termination = "tool_call_limit"
    error: str | None = None
    started = time.time()

    try:
        while len(calls) < max_tool_calls:
            if hasattr(session, "start_turn"):
                session.start_turn()
            packed = memory.pack(initial, turns, calls_used=len(calls), max_calls=max_tool_calls)
            msgs = _normalize_packed_messages(packed)
            resp = await asyncio.to_thread(
                _openai_chat_sync,
                base_url,
                api_key,
                model,
                msgs,
                norm_specs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp["choices"][0]
            msg = choice["message"]
            finish_reason = str(choice.get("finish_reason") or "stop")
            visible_text = (msg.get("content") or "").strip()
            reasoning_text = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            proposed = list(msg.get("tool_calls") or [])
            if not proposed:
                provider_text = "\n".join(p for p in (reasoning_text, visible_text) if p)
                proposed = parse_provider_tool_calls(provider_text, len(calls))

            if not proposed:
                if (
                    reminders_used < submit_reminders
                    and finish_reason != "length"
                    and visible_text
                    and len(calls) < max_tool_calls
                ):
                    reminders_used += 1
                    turns.append([
                        {"role": "assistant", "content": visible_text},
                        {"role": "user", "content": SUBMIT_NUDGE},
                    ])
                    continue
                termination = "generation_token_limit" if finish_reason == "length" else "assistant_without_finish"
                break

            observations, api_calls = [], []
            terminal = False
            for tc in proposed:
                if len(calls) >= max_tool_calls or terminal:
                    break
                fn = tc["function"]
                name = fn["name"]
                raw = fn.get("arguments", "{}")
                cid = tc.get("id") or f"{name}:{len(calls)}"
                args, parse_error = parse_tool_arguments(raw)
                calls.append({"name": name, "arguments": args, "call_id": cid})
                api_calls.append({
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": "{}" if parse_error else (raw if isinstance(raw, str) else json.dumps(raw)),
                    },
                })
                if parse_error:
                    content, stop = json.dumps({"error": parse_error}), False
                else:
                    content, stop = await _exec_tool(session, benchmark, name, args, cid)
                    rid = memory.record_tool_result(name, args, content)
                    if rid:
                        content = f"[receipt_id={rid}] " + content
                rem = max_tool_calls - len(calls)
                if not stop and 0 < rem <= 3:
                    content += f"\n[BUDGET: {rem} tool call(s) remaining. Synthesize your best verified answer (or abstain if unanswerable) and call `finish` now.]"
                observations.append({"role": "tool", "tool_call_id": cid, "name": name, "content": content})
                if stop:
                    terminal = True
                    termination = "finish"

            turn = [{"role": "assistant", "content": visible_text or None, "tool_calls": api_calls}]
            turn.extend(observations)
            turns.append(turn)
            if terminal:
                break

        if force_finish and termination == "tool_call_limit" and session.accepted is None:
            finish_only = [t for t in norm_specs if t.get("name") == "finish"]
            if finish_only:
                turns.append([{
                    "role": "user",
                    "content": (
                        "Tool-call budget reached. Do NOT search further. "
                        "Synthesize your best verified final answer from the evidence and receipts already collected "
                        "and call `finish` right now."
                    ),
                }])
                packed = memory.pack(initial, turns, calls_used=len(calls), max_calls=max_tool_calls)
                msgs = _normalize_packed_messages(packed)
                resp = await asyncio.to_thread(
                    _openai_chat_sync,
                    base_url,
                    api_key,
                    model,
                    msgs,
                    finish_only,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                msg = resp["choices"][0]["message"]
                visible_text = (msg.get("content") or "").strip()
                reasoning_text = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
                proposed = list(msg.get("tool_calls") or [])
                if not proposed:
                    proposed = parse_provider_tool_calls("\n".join(p for p in (reasoning_text, visible_text) if p), len(calls))
                for tc in proposed:
                    fn = tc["function"]
                    if fn["name"] == "finish":
                        fargs, ferr = parse_tool_arguments(fn.get("arguments", "{}"))
                        if not ferr:
                            _, fstop = await _exec_tool(session, benchmark, "finish", fargs, tc.get("id") or f"finish:{len(calls)}")
                            calls.append({"name": "finish", "arguments": fargs, "call_id": tc.get("id") or f"finish:{len(calls)}"})
                            if fstop:
                                termination = "finish"
                                break
                if session.accepted is None and visible_text:
                    fargs = {"answer": visible_text}
                    if benchmark == "financebench":
                        fargs["citations"] = []
                    _, fstop = await _exec_tool(session, benchmark, "finish", fargs, f"forced_finish:{len(calls)}")
                    calls.append({"name": "finish", "arguments": fargs, "call_id": f"forced_finish:{len(calls)}"})
                    if fstop:
                        termination = "finish"
    except Exception as exc:
        termination = "error"
        error = f"{type(exc).__name__}: {str(exc)[:220]}"

    if benchmark == "financebench":
        return {
            "financebench_id": row["financebench_id"],
            "question": question,
            "gold_answer": row.get("answer"),
            "gold_doc": row.get("doc_name"),
            "submission": session.accepted,
            "termination_reason": termination,
            "error": error,
            "tool_calls": calls,
            "tool_call_count": len(calls),
            "turn_count": len(turns),
            "wall_s": round(time.time() - started, 2),
        }
    return {
        "id": row["id"],
        "benchmark": benchmark,
        "question": question,
        "gold_answer": row.get("gold_answer"),
        "submission": session.accepted,
        "termination_reason": termination,
        "error": error,
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "wall_s": round(time.time() - started, 2),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--benchmarks", nargs="+", default=["financebench", "finagentbench", "finretrieval", "finsearchcomp"])
    ap.add_argument("--limit-fb", type=int, default=42)
    ap.add_argument("--limit-mb", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fireworks-key-file", default=str(Path.home() / ".config/fireworks/api_key"))
    ap.add_argument("--split-file", default=str(Path.home() / "Documents/Research/rl-for-ai-research/data/splits.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    fw_key = Path(args.fireworks_key_file).read_text().strip()
    os.environ["FIREWORKS_API_KEY"] = fw_key

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {"model": args.model, "base_url": args.base_url, "benchmarks": {}}

    # 1. FinanceBench
    if "financebench" in args.benchmarks:
        fb_dir = out_root / "financebench"
        fb_dir.mkdir(parents=True, exist_ok=True)
        fb_rows = selected_rows(args.split_file, "eval")[: args.limit_fb]
        all_docs = split_documents(args.split_file)
        print(f"[FinanceBench] Building index over {len(all_docs)} filings (n={len(fb_rows)})...", flush=True)
        index = hb.build_index(doc_names=all_docs, use_cache=True)
        comp_judge = ComponentJudge(
            endpoint="https://api.fireworks.ai/inference/v1/chat/completions",
            model="accounts/fireworks/models/deepseek-v4p1-flash",
            key=fw_key,
            cache_dir=str(ROOT / "artifacts/judge_cache"),
        )
        fb_judge = LiveAdditiveJudge(comp_judge)

        sem = asyncio.Semaphore(args.workers)
        fb_records: list[dict[str, Any]] = []
        done = 0

        async def _run_fb(r: dict[str, Any]) -> None:
            nonlocal done
            async with sem:
                sess = TeacherHarnessSession(index, max_turns=15, budget_cues=True, require_citations=True)
                rec = await run_openai_episode(
                    benchmark="financebench",
                    row=r,
                    session=sess,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    max_tool_calls=15,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                receipts = dict(getattr(getattr(sess, "state", None), "receipts", None) or {})
                rec["score"] = await fb_judge.score(r["financebench_id"], r["question"], r.get("answer", ""), rec["submission"], receipts)
                fb_records.append(rec)
                done += 1
                s = rec["score"]
                print(
                    f"[FB {done}/{len(fb_rows)}] {r['financebench_id']} | term={rec['termination_reason']} "
                    f"| calls={rec['tool_call_count']} | R={s.get('R')} A={s.get('A')} E={s.get('E')} | {rec['wall_s']}s",
                    flush=True,
                )

        await asyncio.gather(*[_run_fb(r) for r in fb_rows])
        (fb_dir / "results.jsonl").write_text("\n".join(json.dumps(x) for x in fb_records) + "\n")
        fb_summary = summarize_fb_records(fb_records)
        (fb_dir / "summary.json").write_text(json.dumps(fb_summary, indent=2))
        overall["benchmarks"]["financebench"] = fb_summary
        print("=== FINANCEBENCH SUMMARY ===", json.dumps(fb_summary, indent=2), flush=True)

    # 2. MultiBench (FinAgentBench, FinRetrieval, FinSearchComp)
    mb_list = [b for b in args.benchmarks if b in ("finagentbench", "finretrieval", "finsearchcomp")]
    if mb_list:
        fr_db = FinRetrievalDB()
        mb_judge = MultiBenchJudge(api_key=fw_key)
        max_calls_map = {"finagentbench": 15, "finretrieval": 15, "finsearchcomp": 24}

        for bm in mb_list:
            bm_dir = out_root / bm
            bm_dir.mkdir(parents=True, exist_ok=True)
            rows = load_multibench_rows([bm], split="eval", limit_per_benchmark=args.limit_mb)
            print(f"[{bm}] Starting eval (n={len(rows)}, max_calls={max_calls_map[bm]})...", flush=True)
            sem = asyncio.Semaphore(args.workers)
            records: list[dict[str, Any]] = []
            done = 0

            async def _run_mb(r: dict[str, Any]) -> None:
                nonlocal done
                async with sem:
                    sess = MultiBenchSession(bm, fr_db=fr_db)
                    rec = await run_openai_episode(
                        benchmark=bm,
                        row=r,
                        session=sess,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model,
                        max_tool_calls=max_calls_map[bm],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    rec["score"] = await mb_judge.score(r, rec["submission"], rec["tool_calls"])
                    records.append(rec)
                    done += 1
                    s = rec["score"]
                    print(
                        f"[{bm.upper()} {done}/{len(rows)}] {r['id']} | term={rec['termination_reason']} "
                        f"| calls={rec['tool_call_count']} | correct={s['correct']} R={s['reward']} | {rec['wall_s']}s",
                        flush=True,
                    )

            await asyncio.gather(*[_run_mb(r) for r in rows])
            (bm_dir / "results.jsonl").write_text("\n".join(json.dumps(x) for x in records) + "\n")
            bm_summary = summarize_multibench(records)
            (bm_dir / "summary.json").write_text(json.dumps(bm_summary, indent=2))
            overall["benchmarks"][bm] = bm_summary["per_benchmark"][bm]
            print(f"=== {bm.upper()} SUMMARY ===", json.dumps(bm_summary["per_benchmark"][bm], indent=2), flush=True)

    (out_root / "overall_summary.json").write_text(json.dumps(overall, indent=2))
    print("\n=== ALL 4 BENCHMARKS COMPLETE ===", flush=True)
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
