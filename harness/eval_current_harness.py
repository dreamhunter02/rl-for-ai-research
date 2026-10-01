"""Evaluate an OpenAI-compatible search agent with the current typed harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import financebench_harness as hb
from generate_teacher_traces import _rows, resolve_api_key, run_teacher, split_documents
from run_identity import verify_run_config


def summarize(records: list[dict[str, Any]]) -> dict[str, int | float]:
    n = len(records)
    finished = sum(r.get("score", {}).get("F") == 1 for r in records)
    correct = sum(r.get("score", {}).get("A") == 1 for r in records)
    grounded = sum(
        r.get("score", {}).get("A") == 1 and r.get("score", {}).get("G") == 1
        for r in records
    )
    return {
        "n": n,
        "finished": finished,
        "correct": correct,
        "grounded": grounded,
        "finish_rate": finished / n if n else 0.0,
        "correct_rate": correct / n if n else 0.0,
        "grounded_rate": grounded / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--split-file", default=os.environ.get("FINANCEBENCH_SPLIT", "artifacts/workshop/split.json"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="EVAL_ENDPOINT_API_KEY")
    parser.add_argument("--targets", default=os.environ.get(
        "FINANCEBENCH_TARGETS",
        "results/paper_2026_rl4llm/targets_frozen_rubric-v3.json",
    ))
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", choices=("baseline", "post_sft"), required=True)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key, args.api_key_env) or "none"
    from reward_calculation import RewardConfig, build_judge
    reward_config = RewardConfig.from_env()
    judge = build_judge(reward_config)
    from run_identity import require_current_judge
    require_current_judge(judge)
    rows = _rows(args.split, args.split_file)
    targets = json.loads(Path(args.targets).read_text())
    index = hb.build_index(doc_names=split_documents(args.split_file))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from finance_env import Bm25Tool, FINANCE_TASK_INSTRUCTIONS
    from teacher_runtime import TOOL_NAMES
    from component_reward import SCORER_VERSION, PROMPT_VERSION
    config={k:v for k,v in vars(args).items() if k not in ('api_key','api_key_env','out')}
    config.update(scorer=SCORER_VERSION,judge_prompt=PROMPT_VERSION,
                  judge_model=getattr(judge,'model',None),task_prompt=FINANCE_TASK_INSTRUCTIONS,
                  tool_specs=[getattr(Bm25Tool(None),name).to_spec() for name in TOOL_NAMES],
                  targets_sha256=hashlib.sha256(Path(args.targets).read_bytes()).hexdigest(),
                  split_sha256=hashlib.sha256(Path(args.split_file).read_bytes()).hexdigest())
    verify_run_config(out,config)
    completed: dict[str, dict[str, Any]] = {}
    if out.exists():
        completed = {
            row["financebench_id"]: row
            for row in (json.loads(line) for line in out.read_text().splitlines() if line.strip())
        }

    for position, row in enumerate(rows, 1):
        fid = row["financebench_id"]
        if fid in completed:
            continue
        started = time.time()
        try:
            result = run_teacher(
                index,
                row,
                args.model,
                args.max_turns,
                base_url=args.base_url,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed,
                target=targets[fid],
                judge=judge,
                judge_confidence_threshold=reward_config.judge_confidence_threshold,
            )
            record = {
                "financebench_id": fid,
                "question": row["question"],
                "gold": row["answer"][0],
                "split": args.split,
                "phase": args.phase,
                "model": args.model,
                "submission": result.get("submission"),
                "answer_text": result.get("answer_text", ""),
                "score": result.get("score") or {},
                "termination_reason": result.get("termination_reason", "max_turns"),
                "tool_calls": result.get("tool_calls", []),
                "trace": result.get("trace", []),
                "generation_metadata": result.get('generation_metadata',[]),
                "wall_s": round(time.time() - started, 2),
            }
        except Exception as exc:
            record = {
                "financebench_id": fid,
                "question": row["question"],
                "gold": row["answer"][0],
                "split": args.split,
                "phase": args.phase,
                "model": args.model,
                "submission": None,
                "score": {"F": 0, "A": 0, "G": 0},
                "termination_reason": "error",
                "error": repr(exc),
                "wall_s": round(time.time() - started, 2),
            }
        with out.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed[fid] = record
        print(json.dumps({
            "position": position,
            "n": len(rows),
            "id": fid,
            "termination": record["termination_reason"],
            "F": record["score"].get("F", 0),
            "A": record["score"].get("A", 0),
            "G": record["score"].get("G", 0),
        }), flush=True)

    records = [completed[row["financebench_id"]] for row in rows]
    summary = {
        **summarize(records),
        "split": args.split,
        "phase": args.phase,
        "model": args.model,
        "out": str(out),
    }
    out.with_suffix(out.suffix + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
