from __future__ import annotations
import argparse, json, time
from pathlib import Path
import financebench_harness as hb
from generate_teacher_traces import run_teacher
from generate_targeted_traces import run_hub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", choices=["lfm", "glm", "gpt56", "gpt56sol", "qwen38", "deepseek41", "gemini38"], required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    ids = [x.strip() for x in Path(args.ids_file).read_text().splitlines() if x.strip()]
    wanted = set(ids)
    rows = {r["financebench_id"]: r for r in __import__("finance_env").load_financebench("eval") if r["financebench_id"] in wanted}
    if set(ids) != set(rows):
        raise SystemExit(f"missing ids: {sorted(wanted - set(rows))}")
    index = hb.build_index()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = {"lfm": "LiquidAI/LFM2.5-8B-A1B", "glm": "nvidia/zai-org/glm-5.3", "gpt56": "azure/openai/gpt-5.6-terra", "gpt56sol": "azure/openai/gpt-5.6-sol", "qwen38": "nvidia/qwen/qwen3.8-flash-next", "deepseek41": "nvidia/deepseek-ai/deepseek-v4-flash", "gemini38": "gcp/google/gemini-3.8-flash"}[args.endpoint]
    records = []
    for fid in ids:
        row = rows[fid]
        started = time.time()
        if args.endpoint == "lfm":
            rec = run_teacher(index, row, model, args.max_turns, base_url="http://spark-a16b.local:18350/v1", api_key="none", temperature=0.0, max_tokens=args.max_tokens)
        elif args.endpoint == "gpt56":
            rec = run_hub(index, row, model, args.max_turns)
        else:
            rec = run_hub(index, row, model, args.max_turns)
        answer = rec.get("answer_text", "")
        records.append({
            "financebench_id": fid,
            "question": row["question"],
            "gold": row["answer"][0],
            "endpoint": args.endpoint,
            "teacher_model": model,
            "answer_text": answer,
            "reward": hb.reward(row["answer"][0], answer),
            "tool_calls": rec.get("tool_calls", []),
            "trace": rec.get("trace", []),
            "termination_reason": rec.get("termination_reason", "max_turns"),
            "wall_s": round(time.time() - started, 2),
        })
        print(json.dumps({"endpoint": args.endpoint, "id": fid, "reward": records[-1]["reward"], "termination": records[-1]["termination_reason"], "tools": len(records[-1]["tool_calls"])}, ensure_ascii=False), flush=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    rewards = [r["reward"] for r in records]
    print(json.dumps({"endpoint": args.endpoint, "n": len(records), "mean_reward": sum(rewards) / len(rewards), "empty": sum(not r["answer_text"].strip() for r in records), "finish": sum(r["termination_reason"] == "finish" for r in records), "strong": sum(x >= 0.9 for x in rewards), "out": str(out)}))

if __name__ == "__main__":
    main()
