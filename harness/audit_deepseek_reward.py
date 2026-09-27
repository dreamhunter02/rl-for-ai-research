"""Small DeepSeek semantic-reward audit over saved FinanceBench traces.

This reads the DeepInfra key from GNOME Keyring through reward_calculation at
request time. It never prints or writes the credential. The selected rows and
judge outputs are saved for calibration, not used as a training run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import financebench_harness as hb
from reward_calculation import RewardConfig, answer_quality_with_judge, build_judge, reward_formula



def known_cases() -> list[dict[str, Any]]:
    return [
        {
            "financebench_id": "known_01148_false_negative",
            "question": "What industry does AMCOR primarily operate in?",
            "gold": "Amcor is a global leader in packaging production for various use cases.",
            "answer_text": "Amcor primarily operates in the packaging industry, focusing on flexible and rigid packaging, specialty cartons, and closures for fast-moving consumer goods. This is identified in its 2019 10-K filing.",
            "_audit_evidence": "Amcor is a global leader in developing and producing responsible packaging for food, beverage, pharmaceutical, medical, home and personal-care products.",
            "deterministic_reward": 0.0,
            "judge": {"verdict": "reference_false_negative"},
            "trace": [],
        },
        {
            "financebench_id": "known_00566_hard_negative",
            "question": "Has Verizon increased its debt on balance sheet between 2022 and the 2021 fiscal period?",
            "gold": "No. Verizon's debt decreased by $229 million.",
            "answer_text": "Yes. Verizon's debt increased by about $407 million.",
            "_audit_evidence": "Verizon reported debt figures for 2021 and 2022.",
            "deterministic_reward": 0.0,
            "judge": {"verdict": "reference_hard_negative"},
            "trace": [],
        },
        {
            "financebench_id": "known_00684_positive",
            "question": "Does AMCOR have an improving gross margin profile as of FY2023?",
            "gold": "No. For AMCOR there has been a slight decline in gross margins by 0.8%.",
            "answer_text": "No, Amcor's gross margin did not improve; it fell from about 19.4% to 18.5%, a decline of about 0.8 percentage points.",
            "_audit_evidence": "Amcor's FY2023 filing reports gross margin of 18.5% versus 19.4% in FY2022.",
            "deterministic_reward": 1.0,
            "judge": {"verdict": "reference_positive"},
            "trace": [],
        },
    ]

def select_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    targets = {"financebench_id_01148", "financebench_id_00684", "financebench_id_00566"}
    for row in rows:
        if row.get("financebench_id") in targets:
            chosen.append(row)
            seen.add(row["financebench_id"])
    buckets = [
        lambda r: r.get("deterministic_reward") == 0 and r.get("judge", {}).get("correctness") == 4,
        lambda r: r.get("deterministic_reward") == 0 and r.get("judge", {}).get("correctness") == 3,
        lambda r: r.get("deterministic_reward", 0) > 0,
    ]
    for bucket in buckets:
        for row in rows:
            if len(chosen) >= limit:
                return chosen
            if row.get("financebench_id") in seen or not bucket(row):
                continue
            chosen.append(row)
            seen.add(row["financebench_id"])
    return chosen[:limit]


async def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    selected = known_cases() + select_rows(rows, max(0, args.limit - len(known_cases())))
    config = RewardConfig(
        judge_backend="deepinfra",
        judge_model="deepseek-ai/DeepSeek-V4.1-Flash",
        judge_cache_path=args.cache,
        require_finish=False,
    )
    judge = build_judge(config)
    out: list[dict[str, Any]] = []
    for row in selected:
        strong, weak = hb._trace_evidence_text(row.get("trace", []))
        evidence = row.get("_audit_evidence", "") or "\n".join(x for x in (strong, weak) if x)
        deterministic_quality, _ = hb.score_answer(row.get("gold", ""), row.get("answer_text", ""))
        quality, meta = await answer_quality_with_judge(
            question=row.get("question", ""),
            gold=row.get("gold", ""),
            candidate=row.get("answer_text", ""),
            evidence=evidence,
            judge=judge,
            config=config,
        )
        evidence_quality, evidence_parts = hb.score_evidence(row.get("gold", ""), row.get("answer_text", ""), row.get("trace", []))
        out.append({
            "financebench_id": row.get("financebench_id"),
            "question": row.get("question", ""),
            "gold": row.get("gold", ""),
            "answer_text": row.get("answer_text", ""),
            "old_deterministic_reward": row.get("deterministic_reward"),
            "opus_reference": row.get("judge", {}),
            "current_deterministic_quality": deterministic_quality,
            "new_answer_quality": quality,
            "new_evidence_quality": evidence_quality,
            "new_grounded_reward": reward_formula(quality, evidence_quality),
            "judge_meta": meta,
            "evidence_parts": evidence_parts,
        })
        print(json.dumps({"id": row.get("financebench_id"), "old": row.get("deterministic_reward"), "new": quality, "judge": meta.get("judge_verdict"), "used": meta.get("judge_used")}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    used = [x for x in out if x["judge_meta"].get("judge_used")]
    recovered = [x for x in out if x["current_deterministic_quality"] == 0 and x["new_answer_quality"] > 0]
    judge_recovered = [x for x in recovered if x["judge_meta"].get("judge_used")]
    print(json.dumps({"n": len(out), "judge_calls": len(used), "deterministic_zero_to_positive": len(recovered), "judge_recovered": len(judge_recovered), "out": args.out}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/teacher_traces/sft_judged_strict.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--cache", default="results/reward_judgments/deepseek_v41_flash_cache.json")
    ap.add_argument("--out", default="results/reward_audits/deepseek_v41_reward_audit_small.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
