#!/usr/bin/env python3
"""Multi-turn Serverless Async GRPO for Muse Glimmer 30B on FinanceBench.

Uses Fireworks Serverless Training API (async_rl_loop_serverless) with:
  - Exact TITO multi-turn rollouts over the stateful FinanceBench harness
    (TeacherHarnessSession + ResearchMemory + StructuredIndex)
  - MoE Router Replay (R3) via routing_matrices captured from ServerlessSampler
  - DAPO Clip-Higher (eps_clip=0.2, eps_clip_high=0.28)
  - Dynamic zero-variance group filtering (has_reward_variance)
  - Live v6 additive rubric scoring via DeepSeek-V4.1-Flash (disk-cached)
  - Optional verifiable efficiency/anti-loop shaping
  - Periodic held-out dev (12) evaluation
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

_DEFAULT_CORPUS = Path.home() / "Documents" / "Research" / "rl-for-ai-research"
os.environ.setdefault("FINANCEBENCH_FILINGS", str(_DEFAULT_CORPUS / "filings"))
os.environ.setdefault("FINANCEBENCH_CACHE", str(_DEFAULT_CORPUS / "artifacts" / "page_cache"))

import financebench_harness as hb
from component_reward import ComponentJudge
from eval_stateful_harness import (
    SUBMIT_REMINDER,
    pack_conversation,
    system_prompt,
)
from generate_teacher_traces import (
    TeacherHarnessSession,
    parse_tool_arguments,
    split_documents,
    submission_answer_text,
)
from live_additive_reward import LiveAdditiveJudge
from muse_glimmer_runner import (
    BASE_MODEL_DEFAULT,
    JUDGE_ENDPOINT_DEFAULT,
    JUDGE_MODEL_DEFAULT,
    RENDERER_DEFAULT,
    TOKENIZER_DEFAULT,
    _extract_text_and_thinking,
    _to_muse_messages,
)
from persistent_state import ResearchMemory
from harness_v2_patches import patch_system_prompt, patch_tool_observation
from provider_tool_calls import parse_provider_tool_calls

from training.recipes.experiment import async_rl_loop_serverless
from training.recipes.experiment.async_rl_loop_serverless import Config
from training.utils import WandBConfig
from training.utils.rl.heldout import HeldoutRuns, make_heldout_evaluation
from training.utils.rl.losses import has_reward_variance
from training.renderer import get_renderer
from training.utils.rl.rollout.renderer import (
    _build_text_only_rollout_sample,
    _completion_logprobs_from_sampled_completion,
    _completion_topk_from_sampled_completion,
    model_input_to_token_ids,
)
from training.utils.rl.rollout.setup import RolloutFn, RolloutSetup
from training.utils.rl.rollout.types import RolloutRun, RolloutSample


def load_split_rows(split_file: str, split_name: str) -> list[dict[str, Any]]:
    data = json.loads(Path(split_file).read_text())
    if split_name == "all":
        return list(data["train"]) + list(data["dev"]) + list(data["eval"])
    return list(data[split_name])


def compute_shaped_reward(
    score: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    efficiency_weight: float = 0.0,
    repeat_penalty: float = 0.0,
    target_max_calls: int = 10,
) -> float | None:
    """Compute final scalar reward from LiveAdditiveJudge score + optional verifiable shaping."""
    base_r = score.get("reward")
    if base_r is None:
        return None
    r = float(base_r)

    # Count exact repeated identical (name, args) tool calls.
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

    # Small verifiable efficiency bonus ONLY when answer is fully correct & grounded.
    if (
        efficiency_weight > 0.0
        and score.get("F") == 1
        and score.get("A") == 1.0
        and score.get("G") == 1.0
    ):
        n_calls = len(calls)
        if n_calls <= target_max_calls:
            r += efficiency_weight
        elif n_calls <= target_max_calls * 2:
            frac = 1.0 - (n_calls - target_max_calls) / float(target_max_calls)
            r += efficiency_weight * max(0.0, frac)

    return max(0.0, r)


def financebench_eval_metrics(
    _rows: Sequence[Mapping[str, Any]], runs: HeldoutRuns
) -> dict[str, float]:
    expected = sum(len(row_runs) for row_runs in runs)
    valid_runs = [run for row_runs in runs for run in row_runs if run is not None]
    n = len(valid_runs)
    if n == 0:
        return {"eval/reward": 0.0, "eval/samples": 0.0, "eval/return_ratio": 0.0}
    metas = [run.metadata or {} for run in valid_runs]
    rewards = [float(m.get("base_reward", run.segments[0].reward)) for run, m in zip(valid_runs, metas)]
    finishes = [int(m.get("F", 0)) for m in metas]
    acc = [1 if (m.get("F") == 1 and m.get("A") == 1.0) else 0 for m in metas]
    g_acc = [1 if (m.get("F") == 1 and m.get("A") == 1.0 and m.get("G") == 1.0) else 0 for m in metas]
    calls = [float(m.get("tool_calls", 0)) for m in metas]
    return {
        "eval/reward": sum(rewards) / n,
        "eval/accuracy_A1": sum(acc) / n,
        "eval/grounded_accuracy": sum(g_acc) / n,
        "eval/finish_rate": sum(finishes) / n,
        "eval/mean_tool_calls": sum(calls) / n,
        "eval/samples": float(n),
        "eval/return_ratio": n / expected if expected else 0.0,
    }


def make_financebench_rollout_fn(
    setup: RolloutSetup,
    *,
    index: hb.StructuredIndex,
    judge: LiveAdditiveJudge,
    renderer_name: str = RENDERER_DEFAULT,
    max_tool_calls: int = 20,
    max_turn_tokens: int = 3072,
    max_seq_len: int = 24576,
    submit_reminders: int = 1,
    efficiency_weight: float = 0.05,
    repeat_penalty: float = 0.02,
    target_max_calls: int = 10,
    improved_harness: bool = False,
    trace_log_path: Path | None = None,
) -> RolloutFn:
    renderer = get_renderer(renderer_name, setup.tokenizer)
    sampler = setup.sampler  # borrowed from recipe; never close
    stop = renderer.get_stop_sequences()
    if stop and all(isinstance(s, int) for s in stop):
        stop_strings = [setup.tokenizer.decode([s], skip_special_tokens=False) for s in stop]
    else:
        stop_strings = [str(s) for s in stop] if stop else None

    log_lock = asyncio.Lock()

    async def rollout_fn(row: dict[str, Any]) -> RolloutRun | None:
        fid = row["financebench_id"]
        gold = row["answer"][0] if isinstance(row.get("answer"), list) else row.get("answer", "")
        session = TeacherHarnessSession(
            index, max_turns=max_tool_calls, budget_cues=False, require_citations=True
        )
        memory = ResearchMemory(row["question"], recent_turn_policy="complete")
        sys_text = system_prompt(submission_contract=True, harness_mode="full")
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

        turns: list[list[dict[str, Any]]] = []
        calls: list[dict[str, Any]] = []
        cid_to_name: dict[str, str] = {}
        seen_call_sigs: set[str] = set()
        raw_segments: list[dict[str, Any]] = []
        reminders_used = 0
        termination = "tool_call_limit"
        started = time.time()

        try:
            while len(calls) < max_tool_calls:
                session.start_turn()
                packed = memory.pack(initial, turns)
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
                    c,
                    prompt_len=prompt_len,
                    completion_len=len(out_tokens),
                    attr="sampling_logprobs",
                    source="sampling_logprobs",
                    required=True,
                )
                if out_logprobs is None:
                    return None
                raw_out_logprobs = _completion_logprobs_from_sampled_completion(
                    c,
                    prompt_len=prompt_len,
                    completion_len=len(out_tokens),
                    attr="inference_logprobs",
                    source="raw inference logprobs",
                    required=False,
                )
                comp_topk = _completion_topk_from_sampled_completion(
                    c, prompt_len=prompt_len, completion_len=len(out_tokens)
                )
                finish_reason = str(getattr(c, "finish_reason", "stop") or "stop")
                parsed_msg, _parse_term = renderer.parse_response(out_tokens)
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
                        content, stop_flag = json.dumps({"error": parse_error}), False
                    else:
                        result = await asyncio.to_thread(session.execute, name, args, cid)
                        content, stop_flag = result.content, result.should_stop
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
                    if stop_flag:
                        terminal = True
                        termination = "finish" if session.accepted is not None else "terminal_rejected"

                turn = [{"role": "assistant", "content": visible_text or None, "tool_calls": api_calls}]
                turn.extend(
                    {k: obs[k] for k in ("role", "tool_call_id", "name", "content")}
                    for obs in observations
                )
                turns.append(turn)
                memory.observe(executed, observations, session.state.receipts)
                if terminal:
                    break
        except Exception as exc:
            print(f"[ROLLOUT ERROR] {fid}: {type(exc).__name__}: {str(exc)[:180]}", flush=True)
            return None

        if not raw_segments:
            return None

        score = await judge.score(
            fid,
            row["question"],
            gold,
            session.accepted,
            session.state.receipts,
        )
        shaped_r = compute_shaped_reward(
            score,
            calls,
            efficiency_weight=efficiency_weight,
            repeat_penalty=repeat_penalty,
            target_max_calls=target_max_calls,
        )
        if shaped_r is None:
            print(f"[SCORE UNRESOLVED] {fid}: {score.get('reason')}", flush=True)
            return None

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
            "financebench_id": fid,
            "base_reward": score.get("reward"),
            "shaped_reward": shaped_r,
            "E": score.get("E"),
            "G": score.get("G"),
            "A": score.get("A"),
            "B": score.get("B"),
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
    ap = argparse.ArgumentParser(description="Serverless Async GRPO for Muse Glimmer 30B on FinanceBench")
    ap.add_argument("--split-file", default="split.json")
    ap.add_argument("--api-key-file", default=str(Path.home() / ".config" / "fireworks" / "api_key"))
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--tokenizer-model", default=TOKENIZER_DEFAULT)
    ap.add_argument("--renderer-name", default=RENDERER_DEFAULT)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=96)
    ap.add_argument("--prompt-groups-per-step", type=int, default=8)
    ap.add_argument("--completions-per-prompt", type=int, default=4)
    ap.add_argument("--min-group-size", type=int, default=3)
    ap.add_argument("--max-concurrency-rollout-sample", type=int, default=32)
    ap.add_argument("--max-head-offpolicy-versions", type=int, default=1)
    ap.add_argument("--max-tool-calls", type=int, default=20)
    ap.add_argument("--max-turn-tokens", type=int, default=3072)
    ap.add_argument("--max-seq-len", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eps-clip", type=float, default=0.2)
    ap.add_argument("--eps-clip-high", type=float, default=0.28)
    ap.add_argument("--efficiency-weight", type=float, default=0.05)
    ap.add_argument("--repeat-penalty", type=float, default=0.02)
    ap.add_argument("--target-max-calls", type=int, default=10)
    ap.add_argument("--evaluation-interval", type=int, default=4)
    ap.add_argument("--snapshot-prefix", default="fb-mg")
    ap.add_argument("--init-from-checkpoint", default=None)
    ap.add_argument("--judge-endpoint", default=JUDGE_ENDPOINT_DEFAULT)
    ap.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    ap.add_argument("--judge-cache", default="run/judge_cache")
    ap.add_argument("--improved-harness", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = Path(args.api_key_file).read_text().strip()
    os.environ["FIREWORKS_API_KEY"] = api_key

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split_rows(args.split_file, "train")[: args.max_rows]
    dev_rows = load_split_rows(args.split_file, "dev")
    all_docs = split_documents(args.split_file)
    print(f"Building StructuredIndex over {len(all_docs)} filings (train={len(train_rows)}, dev={len(dev_rows)})...", flush=True)
    index = hb.build_index(doc_names=all_docs, use_cache=True)

    comp_judge = ComponentJudge(
        endpoint=args.judge_endpoint,
        model=args.judge_model,
        key=api_key,
        cache_dir=args.judge_cache,
    )
    judge = LiveAdditiveJudge(comp_judge)

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
        shuffle=True,
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
        return make_financebench_rollout_fn(
            setup,
            index=index,
            judge=judge,
            renderer_name=args.renderer_name,
            max_tool_calls=args.max_tool_calls,
            max_turn_tokens=args.max_turn_tokens,
            max_seq_len=args.max_seq_len,
            efficiency_weight=args.efficiency_weight,
            repeat_penalty=args.repeat_penalty,
            target_max_calls=args.target_max_calls,
            improved_harness=args.improved_harness,
            trace_log_path=trace_log,
        )

    eval_fn = None
    if args.evaluation_interval > 0 and dev_rows:
        eval_fn = make_heldout_evaluation(
            dev_rows,
            samples_per_row=1,
            metrics_fn=financebench_eval_metrics,
            max_concurrency=len(dev_rows),
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
    print("\n=== GRPO RESULT ===", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
