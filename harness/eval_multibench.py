#!/usr/bin/env python3
"""Unified multi-benchmark Muse Glimmer 30B evaluator for:
  - FinRetrieval (arXiv:2603.04403)
  - FinAgentBench
  - FinSearchComp (Global T2+T3)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(HARNESS_DIR / "benchmarks"))

from benchmarks.multibench_env import DATA_DIR, FinRetrievalDB, MultiBenchJudge, MultiBenchSession
from financeharness_v3 import UniformResearchMemory
from generate_teacher_traces import parse_tool_arguments
from muse_glimmer_runner import (
    BASE_MODEL_DEFAULT,
    JUDGE_ENDPOINT_DEFAULT,
    JUDGE_MODEL_DEFAULT,
    RENDERER_DEFAULT,
    SERVERLESS_TRAINING_URL,
    TOKENIZER_DEFAULT,
    _extract_text_and_thinking,
    _to_muse_messages,
)
from provider_tool_calls import parse_provider_tool_calls

from fireworks.training.sdk import FiretitanServiceClient, types
from transformers import AutoTokenizer
from training.renderer import get_renderer

SUBMIT_NUDGE = (
    "You have not submitted your final answer yet. Call the `finish` tool now with your "
    "verified answer (do not reply with plain text only)."
)


def pack_recent_turns(
    initial: list[dict[str, Any]], turns: list[list[dict[str, Any]]], max_chars: int = 24000
) -> list[dict[str, Any]]:
    """Keep initial system+user plus as many recent turns as fit within max_chars."""
    kept: list[list[dict[str, Any]]] = []
    used = sum(len(json.dumps(m, ensure_ascii=False)) for m in initial)
    for t in reversed(turns):
        sz = sum(len(json.dumps(m, ensure_ascii=False)) for m in t)
        if kept and used + sz > max_chars:
            break
        kept.append(t)
        used += sz
    out = list(initial)
    for t in reversed(kept):
        out.extend(t)
    return out


async def run_multibench_episode(
    row: dict[str, Any],
    sampler: Any,
    renderer: Any,
    fr_db: FinRetrievalDB,
    *,
    max_tool_calls: int = 15,
    max_tokens: int = 3072,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 24576,
    submit_reminders: int = 1,
) -> dict[str, Any]:
    bm = row["benchmark"]
    session = MultiBenchSession(bm, fr_db=fr_db)
    memory = UniformResearchMemory(row['question'])
    initial = [
        {"role": "system", "content": session.system_prompt()},
        {"role": "user", "content": row["question"]},
    ]
    tool_specs = session.tool_specs()
    turns: list[list[dict[str, Any]]] = []
    calls: list[dict[str, Any]] = []
    cid_to_name: dict[str, str] = {}
    reminders_used = 0
    termination = "tool_call_limit"
    error: str | None = None
    started = time.time()
    stop_seqs = renderer.get_stop_sequences()

    try:
        while len(calls) < max_tool_calls:
            packed = memory.pack(initial, turns, calls_used=len(calls), max_calls=max_tool_calls)
            muse_msgs = _to_muse_messages(renderer, tool_specs, packed, cid_to_name)
            prompt = renderer.build_generation_prompt(muse_msgs)
            prompt_tokens = prompt.to_ints()

            remaining_ctx = max_seq_len - len(prompt_tokens)
            if remaining_ctx < 128:
                termination = "context_limit"
                break
            sampling_params = types.SamplingParams(
                max_tokens=min(max_tokens, remaining_ctx),
                temperature=temperature,
                top_p=top_p if temperature > 0 else 1.0,
                stop=stop_seqs,
            )
            if hasattr(sampler, "sample_async"):
                res = await sampler.sample_async(prompt=prompt, num_samples=1, sampling_params=sampling_params)
            else:
                res = await asyncio.to_thread(
                    lambda p=prompt, sp=sampling_params: sampler.sample(
                        prompt=p, num_samples=1, sampling_params=sp
                    ).result()
                )
            seq = res.sequences[0]
            comp_tokens = list(seq.tokens)
            finish_reason = str(getattr(seq, "stop_reason", None) or "stop")
            parsed_msg, _ = renderer.parse_response(comp_tokens)
            visible_text, thinking_text = _extract_text_and_thinking(parsed_msg)

            proposed = list(parsed_msg.get("tool_calls") or [])
            if not proposed:
                provider_text = "\n".join(p for p in (thinking_text, visible_text) if p)
                proposed = parse_provider_tool_calls(provider_text, len(calls))

            if not proposed:
                if (
                    reminders_used < submit_reminders
                    and finish_reason != "length"
                    and visible_text.strip()
                    and len(calls) < max_tool_calls
                ):
                    reminders_used += 1
                    turns.append(
                        [
                            {"role": "assistant", "content": visible_text},
                            {"role": "user", "content": SUBMIT_NUDGE},
                        ]
                    )
                    continue
                termination = "generation_token_limit" if finish_reason == "length" else "assistant_without_finish"
                break

            observations, api_calls = [], []
            terminal = False
            for tc in proposed:
                if len(calls) >= max_tool_calls or terminal:
                    break
                if hasattr(tc, "function"):
                    name, raw, cid = tc.function.name, tc.function.arguments, tc.id or f"{tc.function.name}:{len(calls)}"
                else:
                    name, raw, cid = tc["function"]["name"], tc["function"]["arguments"], tc.get("id") or f"{tc['function']['name']}:{len(calls)}"
                cid_to_name[cid] = name
                args, parse_error = parse_tool_arguments(raw)
                call = {"name": name, "arguments": args, "call_id": cid}
                calls.append(call)
                api_calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": "{}" if parse_error else (raw if isinstance(raw, str) else json.dumps(raw)),
                        },
                    }
                )
                if parse_error:
                    content, stop = json.dumps({"error": parse_error}), False
                else:
                    content, stop = await asyncio.to_thread(session.execute, name, args)
                    rid = memory.record_tool_result(name, args, content)
                    if rid:
                        content = f'[receipt_id={rid}] ' + content
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

        # Forced final synthesis turn: if we hit tool_call_limit without calling finish,
        # give the model 1 final turn with ONLY `finish` enabled to synthesize from gathered receipts.
        if termination == "tool_call_limit" and session.accepted is None:
            finish_only = [t for t in tool_specs if t.get("name") == "finish"]
            if finish_only:
                turns.append([
                    {
                        "role": "user",
                        "content": (
                            "Tool-call budget reached. Do NOT search further. "
                            "Synthesize your best verified final answer from the evidence and receipts already collected "
                            "and call `finish` right now."
                        ),
                    }
                ])
                packed = memory.pack(initial, turns, calls_used=len(calls), max_calls=max_tool_calls)
                muse_msgs = _to_muse_messages(renderer, finish_only, packed, cid_to_name)
                prompt = renderer.build_generation_prompt(muse_msgs)
                prompt_tokens = prompt.to_ints()
                remaining_ctx = max_seq_len - len(prompt_tokens)
                if remaining_ctx >= 256:
                    sampling_params = types.SamplingParams(
                        max_tokens=min(max_tokens, remaining_ctx),
                        temperature=temperature,
                        top_p=top_p if temperature > 0 else 1.0,
                        stop=stop_seqs,
                    )
                    if hasattr(sampler, "sample_async"):
                        res = await sampler.sample_async(prompt=prompt, num_samples=1, sampling_params=sampling_params)
                    else:
                        res = await asyncio.to_thread(
                            lambda p=prompt, sp=sampling_params: sampler.sample(
                                prompt=p, num_samples=1, sampling_params=sp
                            ).result()
                        )
                    seq = res.sequences[0]
                    parsed_msg, _ = renderer.parse_response(list(seq.tokens))
                    visible_text, thinking_text = _extract_text_and_thinking(parsed_msg)
                    proposed = list(parsed_msg.get("tool_calls") or [])
                    if not proposed:
                        provider_text = "\n".join(p for p in (thinking_text, visible_text) if p)
                        proposed = parse_provider_tool_calls(provider_text, len(calls))
                    for tc in proposed:
                        if hasattr(tc, "function"):
                            fname, fraw, fcid = tc.function.name, tc.function.arguments, tc.id or f"finish:{len(calls)}"
                        else:
                            fname, fraw, fcid = tc["function"]["name"], tc["function"]["arguments"], tc.get("id") or f"finish:{len(calls)}"
                        if fname == "finish":
                            fargs, ferr = parse_tool_arguments(fraw)
                            if not ferr:
                                _, fstop = await asyncio.to_thread(session.execute, "finish", fargs)
                                calls.append({"name": "finish", "arguments": fargs, "call_id": fcid})
                                if fstop:
                                    termination = "finish"
                                    break
                    if session.accepted is None and visible_text and visible_text.strip():
                        fargs = {"answer": visible_text.strip()}
                        _, fstop = await asyncio.to_thread(session.execute, "finish", fargs)
                        calls.append({"name": "finish", "arguments": fargs, "call_id": f"forced_finish:{len(calls)}"})
                        if fstop:
                            termination = "finish"

    except Exception as exc:
        termination = "error"
        error = f"{type(exc).__name__}: {str(exc)[:180]}"

    return {
        "id": row["id"],
        "benchmark": bm,
        "question": row["question"],
        "gold_answer": row.get("gold_answer"),
        "submission": session.accepted,
        "termination_reason": termination,
        "error": error,
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "wall_s": round(time.time() - started, 2),
    }


def summarize_multibench(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_bm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_bm[r["benchmark"]].append(r)

    per_bm: dict[str, Any] = {}
    for bm, rows in sorted(by_bm.items()):
        n = len(rows)
        fin = sum(1 for r in rows if r["score"].get("F") == 1)
        cor = sum(1 for r in rows if r["score"].get("correct") == 1)
        rew = sum(float(r["score"].get("reward") or 0.0) for r in rows)
        calls = sum(int(r.get("tool_call_count", 0)) for r in rows)
        per_bm[bm] = {
            "n": n,
            "accuracy": round(cor / n, 4) if n else 0.0,
            "correct": cor,
            "finish_rate": round(fin / n, 4) if n else 0.0,
            "mean_reward": round(rew / n, 4) if n else 0.0,
            "mean_tool_calls": round(calls / n, 2) if n else 0.0,
        }

    macro_acc = round(sum(v["accuracy"] for v in per_bm.values()) / len(per_bm), 4) if per_bm else 0.0
    macro_rew = round(sum(v["mean_reward"] for v in per_bm.values()) / len(per_bm), 4) if per_bm else 0.0
    macro_fin = round(sum(v["finish_rate"] for v in per_bm.values()) / len(per_bm), 4) if per_bm else 0.0
    return {
        "total_n": len(records),
        "macro_accuracy": macro_acc,
        "macro_reward": macro_rew,
        "macro_finish_rate": macro_fin,
        "per_benchmark": per_bm,
    }


async def run_cli(args: argparse.Namespace) -> None:
    api_key = Path(args.api_key_file).read_text().strip()
    os.environ["FIREWORKS_API_KEY"] = api_key

    rows: list[dict[str, Any]] = []
    for bm in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        p = DATA_DIR / f"{bm}_{args.split}.jsonl"
        bm_rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if args.limit_per_benchmark > 0:
            bm_rows = bm_rows[: args.limit_per_benchmark]
        rows.extend(bm_rows)

    print(f"Loaded {len(rows)} rows across {args.benchmarks} (split={args.split})", flush=True)
    fr_db = FinRetrievalDB()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)
    renderer = get_renderer(args.renderer_name, tok)
    service = FiretitanServiceClient(api_key=api_key, base_url=SERVERLESS_TRAINING_URL)
    if args.model_path:
        print(f"Sampling checkpoint: {args.model_path}", flush=True)
        sampler = service.create_sampling_client(model_path=args.model_path, tokenizer=tok)
    else:
        print(f"Sampling base model: {args.base_model}", flush=True)
        sampler = service.create_sampling_client(base_model=args.base_model, tokenizer=tok)

    judge = MultiBenchJudge(api_key=api_key, endpoint=JUDGE_ENDPOINT_DEFAULT, model=JUDGE_MODEL_DEFAULT)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"

    completed: dict[str, dict[str, Any]] = {}
    if res_path.exists() and not args.overwrite:
        for line in res_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                completed[rec["id"]] = rec
        print(f"Resuming: {len(completed)}/{len(rows)} already done.", flush=True)

    sem = asyncio.Semaphore(args.workers)
    lock = asyncio.Lock()

    async def _worker(row: dict[str, Any]) -> None:
        rid = row["id"]
        if rid in completed:
            return
        async with sem:
            rec = await run_multibench_episode(
                row,
                sampler,
                renderer,
                fr_db,
                max_tool_calls=args.max_tool_calls,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            try:
                sc = await judge.score(row, rec["submission"], rec["tool_calls"])
            except Exception as exc:
                sc = {"F": int(rec["submission"] is not None), "correct": 0, "reward": 0.0, "error": str(exc)[:120]}
            rec["score"] = sc
            async with lock:
                completed[rid] = rec
                with res_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"[DONE {len(completed)}/{len(rows)}] {rid} | term={rec['termination_reason']} "
                    f"| calls={rec['tool_call_count']} | correct={sc.get('correct')} R={sc.get('reward')} | {rec['wall_s']}s",
                    flush=True,
                )

    try:
        await asyncio.gather(*[_worker(r) for r in rows])
    finally:
        sampler.close()

    ordered = [completed[r["id"]] for r in rows if r["id"] in completed]
    stats = summarize_multibench(ordered)
    stats["model"] = args.model_path or args.base_model
    stats["split"] = args.split
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2))
    print("\n=== MULTIBENCH SUMMARY ===", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="finretrieval,finagentbench,finsearchcomp")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--limit-per-benchmark", type=int, default=0)
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--tokenizer-model", default=TOKENIZER_DEFAULT)
    ap.add_argument("--renderer-name", default=RENDERER_DEFAULT)
    ap.add_argument("--api-key-file", default=str(Path.home() / ".config" / "fireworks" / "api_key"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-tool-calls", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(run_cli(args))


if __name__ == "__main__":
    main()
