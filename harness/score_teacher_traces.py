"""Score the 91 saved teacher traces with the reward-v2 answer/evidence stack.

Teacher traces predate the Tinker finish tool and terminate in an assistant
answer, so this audit reports finish-gate coverage separately rather than
silently zeroing all teacher examples.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

import financebench_harness as hb
from reward_calculation import RewardConfig, answer_quality_with_judge, build_judge, reward_formula


def has_finish(trace: list[dict[str, Any]]) -> bool:
    for item in trace:
        for call in item.get("tool_calls", []) if isinstance(item, dict) else []:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or call.get("function", {}).get("name")
            if name == "finish":
                return True
    return False


async def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    config = RewardConfig(
        judge_backend="deepinfra",
        judge_model="deepseek-ai/DeepSeek-V4.1-Flash",
        judge_cache_path=args.cache,
        require_finish=False,
    )
    judge = build_judge(config)
    results: list[dict[str, Any]] = []
    for row in rows:
        strong, weak = hb._trace_evidence_text(row.get("trace", []))
        evidence = "\n".join(x for x in (strong, weak) if x)
        q, meta = await answer_quality_with_judge(
            question=row.get("question", ""), gold=row.get("gold", ""),
            candidate=row.get("answer_text", ""), evidence=evidence,
            judge=judge, config=config,
        )
        e, evidence_parts = hb.score_evidence(row.get("gold", ""), row.get("answer_text", ""), row.get("trace", []))
        finish = has_finish(row.get("trace", []))
        results.append({
            "financebench_id": row.get("financebench_id"),
            "old_deterministic_reward": row.get("deterministic_reward"),
            "opus_correctness": row.get("judge", {}).get("correctness"),
            "opus_verdict": row.get("judge", {}).get("verdict"),
            "current_deterministic_quality": meta.get("deterministic_quality"),
            "new_answer_quality_without_finish_gate": q,
            "evidence_quality": e,
            "grounded_reward_without_finish_gate": reward_formula(q, e),
            "has_tinker_finish_call": finish,
            "reward_with_strict_finish_gate": reward_formula(q, e) if finish else 0.0,
            "judge_meta": meta,
            "evidence_parts": evidence_parts,
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    q=[r["new_answer_quality_without_finish_gate"] for r in results]
    e=[r["evidence_quality"] for r in results]
    g=[r["grounded_reward_without_finish_gate"] for r in results]
    strict=[r["reward_with_strict_finish_gate"] for r in results]
    used=[r for r in results if r["judge_meta"].get("judge_used")]
    opus_pass=[r for r in results if r["opus_correctness"] == 4]
    q_positive=[r for r in results if r["new_answer_quality_without_finish_gate"] > 0]
    confusion = {"true_positive": 0, "false_negative": 0, "false_positive": 0, "true_negative": 0}
    for r in results:
        reference = r["opus_correctness"] == 4
        predicted = r["new_answer_quality_without_finish_gate"] > 0
        if reference and predicted: confusion["true_positive"] += 1
        elif reference and not predicted: confusion["false_negative"] += 1
        elif not reference and predicted: confusion["false_positive"] += 1
        else: confusion["true_negative"] += 1
    summary={
        "n":len(results), "mean_answer_quality":sum(q)/len(q),
        "mean_evidence_quality":sum(e)/len(e), "mean_grounded_reward_without_finish_gate":sum(g)/len(g),
        "mean_strict_finish_gated_reward":sum(strict)/len(strict),
        "answer_positive":len(q_positive), "opus_correctness_4":len(opus_pass),
        "confusion_vs_opus_correctness4":confusion,
        "finish_calls":sum(r["has_tinker_finish_call"] for r in results),
        "judge_used":len(used), "judge_errors":sum(bool(r["judge_meta"].get("judge_error")) for r in results),
        "answer_quality_distribution":dict(Counter(round(x,3) for x in q)),
        "evidence_quality_distribution":dict(Counter(round(x,3) for x in e)),
        "grounded_distribution":dict(Counter(round(x,3) for x in g)),
        "out":args.out,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="results/teacher_traces/sft_judged_strict.jsonl")
    ap.add_argument("--cache",default="results/reward_judgments/deepseek_v41_flash_teacher91_cache.json")
    ap.add_argument("--out",default="results/reward_audits/teacher91_reward_v2_scores.json")
    ap.add_argument("--summary",default="results/reward_audits/teacher91_reward_v2_summary.json")
    asyncio.run(run(ap.parse_args()))

if __name__ == "__main__": main()
