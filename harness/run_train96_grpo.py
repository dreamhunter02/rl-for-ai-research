#!/usr/bin/env python3
"""Resume-safe train96 coordinator for the bounded Unsloth GRPO trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.batch_size < 1 or args.group_size < 2:
        raise ValueError("invalid batch or group size")
    split = json.loads(args.split.read_text())
    ids = [row["financebench_id"] for row in split["train"]]
    if len(ids) != 96 or len(set(ids)) != 96:
        raise ValueError("expected 96 unique frozen training IDs")
    if set(ids) & {row["financebench_id"] for part in ("dev", "eval") for row in split[part]}:
        raise ValueError("training split overlaps dev/eval")
    identity = {"split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
                "ids": ids, "model": args.model, "batch_size": args.batch_size,
                "group_size": args.group_size, "max_turns": args.max_turns,
                "max_new_tokens": args.max_new_tokens, "max_seq_length": args.max_seq_length,
                "learning_rate": args.learning_rate, "seed": args.seed,
                "corpus_scope": "train96_sources"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "identity.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("run identity mismatch; use another output directory")
    else:
        manifest.write_text(json.dumps(identity, indent=2))
    model = args.model
    summaries = []
    trainer = Path(__file__).with_name("unsloth_financebench_grpo_v6.py")
    for batch_no, start in enumerate(range(0, len(ids), args.batch_size), 1):
        batch_ids = ids[start:start + args.batch_size]
        batch_dir = args.output_dir / f"batch-{batch_no:03d}"
        summary_path = batch_dir / "summary.json"
        run_dir = batch_dir
        rollouts_from = None
        if batch_dir.exists() and not summary_path.exists():
            rollouts_from = batch_dir / "rollouts.jsonl"
            if not rollouts_from.is_file():
                raise RuntimeError(f"incomplete batch {batch_dir}; no saved rollouts to recover")
            run_dir = args.output_dir / f"batch-{batch_no:03d}-recovery-001"
            summary_path = run_dir / "summary.json"
            if run_dir.exists() and not summary_path.exists():
                raise RuntimeError(f"incomplete recovery {run_dir}; inspect before retrying")
        if not summary_path.exists():
            command = [sys.executable, str(trainer), "--model", model,
                       "--split", str(args.split), "--teacher-dataset", str(args.teacher_dataset),
                       "--ids", ",".join(batch_ids), "--output-dir", str(run_dir),
                       "--group-size", str(args.group_size), "--max-turns", str(args.max_turns),
                       "--max-new-tokens", str(args.max_new_tokens),
                       "--max-seq-length", str(args.max_seq_length),
                       "--learning-rate", str(args.learning_rate),
                       "--seed", str(args.seed + start * 1000), "--all-train-corpus"]
            if rollouts_from is not None:
                command += ["--rollouts-from", str(rollouts_from)]
            with (args.output_dir / f"{run_dir.name}.log").open("a") as log:
                print(json.dumps({"stage": "batch_start", "batch": batch_no,
                                  "ids": batch_ids, "model": model}), flush=True)
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads(summary_path.read_text())
        if summary.get("records") != len(batch_ids) * args.group_size:
            raise ValueError(f"incomplete records in batch {batch_no}")
        if summary.get("optimizer_steps") == 1:
            model = summary["adapter"]
            if not Path(model).is_dir():
                raise FileNotFoundError(model)
        elif summary.get("optimizer_steps") != 0:
            raise ValueError(f"invalid optimizer step count in batch {batch_no}")
        summaries.append({"batch": batch_no, "ids": batch_ids,
                          "optimizer_steps": summary["optimizer_steps"],
                          "retained_groups": summary["retained_groups"],
                          "adapter": model})
        (args.output_dir / "progress.json").write_text(json.dumps({
            "questions_processed": sum(len(item["ids"]) for item in summaries),
            "optimizer_steps": sum(item["optimizer_steps"] for item in summaries),
            "batches": summaries, "current_adapter": model}, indent=2))
        print(json.dumps({"stage": "batch_complete", **summaries[-1]}), flush=True)
    (args.output_dir / "complete.json").write_text(json.dumps({
        "questions_processed": 96, "optimizer_steps": sum(x["optimizer_steps"] for x in summaries),
        "final_adapter": model}, indent=2))


if __name__ == "__main__":
    main()
