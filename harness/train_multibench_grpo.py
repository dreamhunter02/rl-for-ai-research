#!/usr/bin/env python3
"""Multi-Benchmark Serverless Async GRPO for Muse Glimmer 30B across:
  - FinRetrieval (arXiv:2603.04403)
  - FinAgentBench
  - FinSearchComp (Global T2+T3)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(HARNESS_DIR / "benchmarks"))

from benchmarks.multibench_env import DATA_DIR, FinRetrievalDB, MultiBenchJudge, MultiBenchSession
from financeharness_v3 import UniformResearchMemory
from eval_multibench import SUBMIT_NUDGE, pack_recent_turns
from generate_teacher_traces import parse_tool_arguments
from muse_glimmer_runner import (
    BASE_MODEL_DEFAULT,
    JUDGE_ENDPOINT_DEFAULT,
    JUDGE_MODEL_DEFAULT,
    RENDERER_DEFAULT,
    TOKENIZER_DEFAULT,
    _extract_text_and_thinking,
    _to_muse_messages,
)
from provider_tool_calls import parse_provider_tool_calls

from training.recipes.experiment import async_rl_loop_serverless
from training.recipes.experiment.async_rl_loop_serverless import Config
from training.renderer import get_renderer
from training.utils import WandBConfig
from training.utils.rl.heldout import HeldoutRuns, make_heldout_evaluation
from training.utils.rl.losses import has_reward_variance
from training.utils.rl.rollout.renderer import (
    _build_text_only_rollout_sample,
    _completion_logprobs_from_sampled_completion,
    _completion_topk_from_sampled_completion,
    model_input_to_token_ids,
)
from training.utils.rl.rollout.setup import RolloutFn, RolloutSetup
from training.utils.rl.rollout.types import RolloutRun, RolloutSample


def load_interleaved_rows(
    benchmarks: list[str], split: str, per_benchmark: int, seed: int = 42
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: list[list[dict[str, Any]]] = []
    for bm in benchmarks:
        p = DATA_DIR / f"{bm}_{split}.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rng.shuffle(rows)
        if per_benchmark > 0:
            rows = rows[:per_benchmark]
        buckets.append(rows)
    interleaved: list[dict[str, Any]] = []
    max_len = max((len(b) for b in buckets), default=0)
    for i in range(max_len):
        for b in buckets:
            if i < len(b):
                interleaved.append(b[i])
    return interleaved


def compute_multibench_shaped_reward(
    score: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    efficiency_weight: float = 0.05,
    repeat_penalty: float = 0.03,
    target_max_calls: int = 6,
) -> float:
    r = float(score.get("reward") or 0.0)
    seen: set[str] = set()
    repeats = 0
    for c in calls:
        sig = f"{c.get('name')}:{json.dumps(c.get('arguments', {}), sort_keys=True, ensure_ascii=False)}"
        if sig in seen:
            repeats += 1
        else:
            seen.add(sig)
    if repeat_penalty > 0.0 and repeats > 0:
        r -= min(0.10, repeat_penalty * repeats)
    if efficiency_weight > 0.0 and score.get("correct") == 1:
        n_calls = len(calls)
        if n_calls <= target_max_calls:
            r += efficiency_weight
        elif n_calls <= target_max_calls * 2:
            frac = 1.0 - (n_calls - target_max_calls) / float(target_max_calls)
            r += efficiency_weight * max(0.0, frac)
    return max(0.0, r)


def multibench_eval_metrics(
    _rows: Sequence[Mapping[str, Any]], runs: HeldoutRuns
) -> dict[str, float]:
    expected = sum(len(row_runs) for row_runs in runs)
    valid = [run for row_runs in runs for run in row_runs if run is not None]
    if not valid:
        return {"eval/accuracy": 0.0, "eval/samples": 0.0, "eval/return_ratio": 0.0}
    by_bm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in valid:
        m = dict(run.metadata or {})
        by_bm[str(m.get("benchmark", "unknown"))].append(m)

    metrics: dict[str, float] = {}
    accs = []
    for bm, ms in sorted(by_bm.items()):
        acc = sum(int(m.get("correct", 0)) for m in ms) / len(ms)
        fin = sum(int(m.get("F", 0)) for m in ms) / len(ms)
        metrics[f"eval/{bm}_acc"] = acc
        metrics[f"eval/{bm}_finish"] = fin
        accs.append(acc)
    metrics["eval/macro_accuracy"] = sum(accs) / len(accs) if accs else 0.0
    metrics["eval/mean_tool_calls"] = sum(float((r.metadata or {}).get("tool_calls", 0)) for r in valid) / len(valid)
    metrics["eval/samples"] = float(len(valid))
    metrics["eval/return_ratio"] = len(valid) / expected if expected else 0.0
    return metrics


def make_multibench_rollout_fn(
    setup: RolloutSetup,
    *,
    fr_db: FinRetrievalDB,
    judge: MultiBenchJudge,
    renderer_name: str = RENDERER_DEFAULT,
    max_tool_calls: int = 15,
    max_turn_tokens: int = 3072,
    max_seq_len: int = 24576,
    submit_reminders: int = 1,
    efficiency_weight: float = 0.05,
    repeat_penalty: float = 0.03,
    target_max_calls: int = 6,
    trace_log_path: Path | None = None,
) -> RolloutFn:
    renderer = get_renderer(renderer_name, setup.tokenizer)
    sampler = setup.sampler
    stop = renderer.get_stop_sequences()
    if stop and all(isinstance(s, int) for s in stop):
        stop_strings = [setup.tokenizer.decode([s], skip_special_tokens=False) for s in stop]
    else:
        stop_strings = [str(s) for s in stop] if stop else None
    log_lock = asyncio.Lock()

    async def rollout_fn(row: dict[str, Any]) -> RolloutRun | None:
        rid = row["id"]
        bm = row["benchmark"]
        session = MultiBenchSession(bm, fr_db=fr_db)
        memory = UniformResearchMemory(row['question'])
        memory = UniformResearchMemory(row['question'])
        initial = [
            {"role": "system", "content": session.system_prompt()},
            {"role": "user", "content": row["question"]},
        ]
        tool_specs = session.tool_specs()
        turns: list[list[dict[str, Any]]] = []
        calls: list[dict[str, Any]] = []
        cid_to_name: dict[str, str] = {}
        raw_segments: list[dict[str, Any]] = []
        reminders_used = 0
        termination = "tool_call_limit"
        started = time.time()

        try:
            while len(calls) < max_tool_calls:
                packed = memory.pack(initial, turns, calls_used=len(calls), max_calls=max_tool_calls)
                muse_msgs = _to_muse_messages(renderer, tool_specs, packed, cid_to_name)
                model_input = renderer.build_generation_prompt(muse_msgs)
                prompt_token_ids = model_input_to_token_ids(model_input)

                remaining_ctx = max_seq_len - len(prompt_token_ids)
                if remaining_ctx < 128:
                    termination = "context_limit"
                    break

                sk = dict(setup.sample_kwargs or {})
                sk["n"] = 1
                sk["max_tokens"] = min(max_turn_tokens, remaining_ctx)
                sk["stop"] = stop_strings

                completions = await sampler.sample_with_prompt_tokens(prompt_token_ids, **sk)
                if not completions:
                    return None
                c = completions[0]
                prompt_len = int(c.prompt_len)
                out_tokens = list(c.full_tokens[prompt_len:])
                if not out_tokens:
                    return None
                out_logprobs = _completion_logprobs_from_sampled_completion(
                    c, prompt_len=prompt_len, completion_len=len(out_tokens),
                    attr="sampling_logprobs", source="sampling_logprobs", required=True,
                )
                if out_logprobs is None:
                    return None
                raw_out_logprobs = _completion_logprobs_from_sampled_completion(
                    c, prompt_len=prompt_len, completion_len=len(out_tokens),
                    attr="inference_logprobs", source="raw inference logprobs", required=False,
                )
                comp_topk = _completion_topk_from_sampled_completion(
                    c, prompt_len=prompt_len, completion_len=len(out_tokens)
                )
                finish_reason = str(getattr(c, "finish_reason", "stop") or "stop")
                parsed_msg, _ = renderer.parse_response(out_tokens)
                visible_text, thinking_text = _extract_text_and_thinking(parsed_msg)

                raw_segments.append(
                    {
                        "prompt_token_ids": prompt_token_ids,
                        "completion_tokens": out_tokens,
                        "completion_logprobs": out_logprobs,
                        "raw_completion_logprobs": raw_out_logprobs,
                        "routing_matrices": getattr(c, "routing_matrices", None),
                        "top_sampling_references": getattr(c, "top_sampling_references", None),
                        "sampling_mask_references": getattr(c, "sampling_mask_references", None),
                        "completion_topk": comp_topk,
                        "finish_reason": finish_reason,
                        "text": visible_text,
                    }
                )

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
                    calls.append({"name": name, "arguments": args, "call_id": cid})
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
                        content, stop_flag = json.dumps({"error": parse_error}), False
                    else:
                        content, stop_flag = await asyncio.to_thread(session.execute, name, args)
                        rid = memory.record_tool_result(name, args, content)
                        if rid:
                            content = f'[receipt_id={rid}] ' + content
                    rem = max_tool_calls - len(calls)
                    if not stop_flag and 0 < rem <= 3:
                        content += f"\n[BUDGET: {rem} tool call(s) remaining. Synthesize your best verified answer (or abstain if unanswerable) and call `finish` now.]"
                    observations.append({"role": "tool", "tool_call_id": cid, "name": name, "content": content})
                    if stop_flag:
                        terminal = True
                        termination = "finish"

                turn = [{"role": "assistant", "content": visible_text or None, "tool_calls": api_calls}]
                turn.extend(observations)
                turns.append(turn)
                if terminal:
                    break
        except Exception as exc:
            print(f"[ROLLOUT ERROR] {rid}: {type(exc).__name__}: {str(exc)[:180]}", flush=True)
            return None

        if not raw_segments:
            return None

        try:
            score = await judge.score(row, session.accepted, calls)
        except Exception as exc:
            print(f"[SCORE ERROR] {rid}: {type(exc).__name__}: {str(exc)[:140]}", flush=True)
            return None

        shaped_r = compute_multibench_shaped_reward(
            score,
            calls,
            efficiency_weight=efficiency_weight,
            repeat_penalty=repeat_penalty,
            target_max_calls=target_max_calls,
        )

        samples: list[RolloutSample] = []
        for seg in raw_segments:
            run_one = _build_text_only_rollout_sample(
                prompt_token_ids=seg["prompt_token_ids"],
                completion_tokens=seg["completion_tokens"],
                completion_logprobs=seg["completion_logprobs"],
                raw_completion_logprobs=seg["raw_completion_logprobs"],
                routing_matrices=seg["routing_matrices"],
                top_sampling_references=seg["top_sampling_references"],
                sampling_mask_references=seg["sampling_mask_references"],
                completion_topk=seg["completion_topk"],
                logprobs_echoed=False,
                reward=shaped_r,
                finish_reason=seg["finish_reason"],
                text=seg["text"],
            )
            samples.extend(run_one.segments)

        meta = {
            "id": rid,
            "benchmark": bm,
            "base_reward": score.get("reward"),
            "shaped_reward": shaped_r,
            "correct": score.get("correct"),
            "F": score.get("F"),
            "tool_calls": len(calls),
            "turns": len(raw_segments),
            "termination": termination,
            "wall_s": round(time.time() - started, 2),
        }
        if trace_log_path is not None:
            async with log_lock:
                with trace_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        return RolloutRun(segments=samples, metadata=meta)

    return rollout_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="finretrieval,finagentbench,finsearchcomp")
    ap.add_argument("--train-per-benchmark", type=int, default=30)
    ap.add_argument("--eval-per-benchmark", type=int, default=8)
    ap.add_argument("--api-key-file", default=str(Path.home() / ".config" / "fireworks" / "api_key"))
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--tokenizer-model", default=TOKENIZER_DEFAULT)
    ap.add_argument("--renderer-name", default=RENDERER_DEFAULT)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--prompt-groups-per-step", type=int, default=6)
    ap.add_argument("--completions-per-prompt", type=int, default=4)
    ap.add_argument("--min-group-size", type=int, default=2)
    ap.add_argument("--max-concurrency-rollout-sample", type=int, default=24)
    ap.add_argument("--max-head-offpolicy-versions", type=int, default=1)
    ap.add_argument("--max-tool-calls", type=int, default=15)
    ap.add_argument("--max-turn-tokens", type=int, default=3072)
    ap.add_argument("--max-seq-len", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=1.5e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eps-clip", type=float, default=0.2)
    ap.add_argument("--eps-clip-high", type=float, default=0.28)
    ap.add_argument("--efficiency-weight", type=float, default=0.05)
    ap.add_argument("--repeat-penalty", type=float, default=0.03)
    ap.add_argument("--target-max-calls", type=int, default=6)
    ap.add_argument("--evaluation-interval", type=int, default=4)
    ap.add_argument("--snapshot-prefix", default="mb-mg-v1")
    ap.add_argument("--init-from-checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = Path(args.api_key_file).read_text().strip()
    os.environ["FIREWORKS_API_KEY"] = api_key
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    bms = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    train_rows = load_interleaved_rows(bms, "train", args.train_per_benchmark, seed=args.seed)
    eval_rows = load_interleaved_rows(bms, "eval", args.eval_per_benchmark, seed=args.seed)
    print(f"MultiBench GRPO: train={len(train_rows)}, in-loop eval={len(eval_rows)} across {bms}", flush=True)

    fr_db = FinRetrievalDB()
    judge = MultiBenchJudge(api_key=api_key, endpoint=JUDGE_ENDPOINT_DEFAULT, model=JUDGE_MODEL_DEFAULT)

    cfg = Config(
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_completion_tokens=args.max_turn_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        epochs=args.epochs,
        shuffle=False,  # already balanced & interleaved!
        seed=args.seed,
        max_rows=len(train_rows),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        min_group_size=args.min_group_size,
        max_incomplete_group_retries=2,
        eps_clip=args.eps_clip,
        eps_clip_high=args.eps_clip_high,
        sample_timeout=2400.0,
        snapshot_prefix=args.snapshot_prefix,
        metrics_file=str(run_dir / "metrics.jsonl"),
        init_from_checkpoint=args.init_from_checkpoint,
        save_final_checkpoint=True,
        wandb=WandBConfig(entity=None),
    )

    trace_log = run_dir / "rollouts.jsonl"

    def factory(setup: RolloutSetup) -> RolloutFn:
        return make_multibench_rollout_fn(
            setup,
            fr_db=fr_db,
            judge=judge,
            renderer_name=args.renderer_name,
            max_tool_calls=args.max_tool_calls,
            max_turn_tokens=args.max_turn_tokens,
            max_seq_len=args.max_seq_len,
            efficiency_weight=args.efficiency_weight,
            repeat_penalty=args.repeat_penalty,
            target_max_calls=args.target_max_calls,
            trace_log_path=trace_log,
        )

    eval_fn = None
    if args.evaluation_interval > 0 and eval_rows:
        eval_fn = make_heldout_evaluation(
            eval_rows,
            samples_per_row=1,
            metrics_fn=multibench_eval_metrics,
            max_concurrency=min(24, len(eval_rows)),
        )

    result = async_rl_loop_serverless.main(
        cfg,
        rollout_fn_factory=factory,
        dynamic_filter_fn=has_reward_variance,
        evaluation_fn=eval_fn,
        evaluation_interval=args.evaluation_interval,
        rows=train_rows,
    )
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print("\n=== MULTIBENCH GRPO RESULT ===", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
