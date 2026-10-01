import json
import sys
from pathlib import Path

import pytest

import run_train96_grpo


def test_train96_coordinator_covers_split_and_resumes(tmp_path, monkeypatch):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "train": [{"financebench_id": f"train-{i:03d}"} for i in range(96)],
        "dev": [{"financebench_id": "dev-001"}],
        "eval": [{"financebench_id": "eval-001"}],
    }))
    teacher = tmp_path / "sft.jsonl"
    teacher.write_text("{}\n")
    adapter = tmp_path / "sft_adapter"
    adapter.mkdir()
    output = tmp_path / "run"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        batch = Path(command[command.index("--output-dir") + 1])
        batch.mkdir()
        next_adapter = batch / "adapter-step-1"
        next_adapter.mkdir()
        (batch / "summary.json").write_text(json.dumps({
            "records": 4, "optimizer_steps": 1, "retained_groups": 2,
            "adapter": str(next_adapter),
        }))

    monkeypatch.setattr(run_train96_grpo.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_train96_grpo.py", "--split", str(split),
        "--teacher-dataset", str(teacher), "--model", str(adapter),
        "--output-dir", str(output)])
    run_train96_grpo.main()
    assert len(calls) == 48
    assert all("--all-train-corpus" in call for call in calls)
    assert json.loads((output / "complete.json").read_text())["questions_processed"] == 96
    run_train96_grpo.main()
    assert len(calls) == 48  # Completed batches are not launched twice.


def test_train96_coordinator_rejects_overlap(tmp_path, monkeypatch):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "train": [{"financebench_id": f"train-{i:03d}"} for i in range(96)],
        "dev": [{"financebench_id": "train-000"}], "eval": [],
    }))
    monkeypatch.setattr(sys, "argv", ["run_train96_grpo.py", "--split", str(split),
        "--teacher-dataset", str(tmp_path / "sft.jsonl"), "--model", "adapter",
        "--output-dir", str(tmp_path / "run")])
    with pytest.raises(ValueError, match="overlaps"):
        run_train96_grpo.main()
