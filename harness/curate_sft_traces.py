"""Balance fully verified traces to one trajectory per question and teacher model."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_sft_dataset import eligible_record


MODEL_ALIASES = {
    "deepseek_v4_flash": "nvidia/deepseek-ai/deepseek-v4-flash",
    "gemini38_flash": "gcp/google/gemini-3.8-flash",
    "glm53_flash": "nvidia/zai-org/glm-5.3-flash",
    "gpt56_terra": "openai/openai/gpt-5.6-terra",
    "nemotron3_ultra": "nvidia/nvidia/nemotron-3-ultra",
    "qwen38_flash_next": "nvidia/qwen/qwen3.8-flash-next",
}


def normalized_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def select_balanced(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen: set[tuple[str, str]] = set()
    for record in sorted(rows, key=lambda r: (len(r.get('tool_calls') or []), len(json.dumps(r.get('trace',[]))))):
        if not eligible_record(record):
            continue
        record = dict(record)
        record["teacher_model"] = normalized_model(str(record.get("teacher_model", "unknown")))
        key = (str(record["financebench_id"]), record["teacher_model"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    rows = []
    for name in args.input:
        rows.extend(json.loads(line) for line in Path(name).read_text().splitlines() if line.strip())
    selected = select_balanced(rows)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    per_question = Counter(row["financebench_id"] for row in selected)
    summary = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "unique_questions": len(per_question),
        "min_trajectories_per_question": min(per_question.values()) if per_question else 0,
        "max_trajectories_per_question": max(per_question.values()) if per_question else 0,
        "teacher_models": dict(Counter(row["teacher_model"] for row in selected)),
        "out": str(output),
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
