#!/usr/bin/env bash
set -euo pipefail
cd /home/dreamhunter/Documents/Research/rl-for-ai-research
for attempt in $(seq 1 60); do
  if curl -fsS --max-time 3 http://10.0.0.136:18361/health >/dev/null; then
    export JUDGE_BACKEND=deepinfra
    export DEEPINFRA_API_KEY_FILE=/home/dreamhunter/.config/financebench/deepinfra.key
    exec .venv-workshop/bin/python -u harness/eval_current_harness.py \
      --split dev --model Qwen3.5-4B:qwen-sft --base-url http://10.0.0.136:18361/v1 \
      --phase post_sft --max-turns 8 --max-tokens 1024 --temperature 0 --seed 0 \
      --out results/paper_2026_rl4llm/qwen35_4b_qlora_dev12_20260930.jsonl
  fi
  sleep 10
done
echo 'Server readiness timed out; evaluation was not started.' >&2
exit 1
