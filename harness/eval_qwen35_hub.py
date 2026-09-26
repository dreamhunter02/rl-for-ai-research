"""FinanceBench baseline using the NVIDIA Inference Hub Qwen3.5-9B endpoint."""
from __future__ import annotations
import json
import time
from pathlib import Path

import financebench_harness as hb
from generate_targeted_traces import run_hub
from generate_teacher_traces import _rows

INDEX = hb.build_index()
ROWS = _rows("eval")
MODEL = "nvidia/qwen/qwen3.5-9b"
OUT = Path("results/local_eval/qwen35_9b_inference_hub_eval_10turn.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

for i, row in enumerate(ROWS, 1):
    started = time.time()
    try:
        rec = run_hub(INDEX, row, MODEL, max_turns=10)
        result = {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "gold": row["answer"][0],
            "model": MODEL,
            "answer_text": rec.get("answer_text", ""),
            "tool_calls": rec.get("tool_calls", []),
            "trace": rec.get("trace", []),
            "deterministic_reward": hb.reward(row["answer"][0], rec.get("answer_text", "")),
            "wall_s": round(time.time() - started, 2),
        }
    except Exception as exc:
        result = {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "gold": row["answer"][0],
            "model": MODEL,
            "error": repr(exc),
            "wall_s": round(time.time() - started, 2),
        }
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(
        f"[{i}/{len(ROWS)}] {row['financebench_id']} "
        f"reward={result.get('deterministic_reward')} "
        f"tools={len(result.get('tool_calls', []))} wall={result['wall_s']}s",
        flush=True,
    )

records = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()]
valid = [r for r in records if "deterministic_reward" in r]
print(json.dumps({
    "n": len(records),
    "valid": len(valid),
    "exact_or_numeric": sum(r["deterministic_reward"] >= 0.9 for r in valid),
    "mean_reward": sum(r["deterministic_reward"] for r in valid) / max(1, len(valid)),
    "out": str(OUT),
}))
