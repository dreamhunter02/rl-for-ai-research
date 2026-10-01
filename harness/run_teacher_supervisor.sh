#!/usr/bin/env bash
set -uo pipefail

slug=$1
model=$2
initial_delay=$3
cooldown=$4
root=${5:-results/teacher_traces/inference_hub_train96_raw12}
targets=${6:-results/teacher_traces/inference_hub_train96_shards/targets_generation_only_all96.json}

export PYTHONPATH=harness
export TEACHER_ENDPOINT_API_KEY="$(<"$HOME/.config/financebench/inference_hub.key")"
export JUDGE_BACKEND=deepinfra
export DEEPINFRA_API_KEY_FILE="$HOME/.config/financebench/deepinfra.key"
export JUDGE_CACHE_PATH="$root/$slug.judge_cache.json"

sleep "$initial_delay"
while true; do
  echo "SUPERVISOR_START $(date -Is)"
  .venv-workshop/bin/python harness/generate_teacher_traces.py \
    --split train \
    --split-file artifacts/workshop/split.json \
    --model "$model" \
    --base-url https://inference-api.nvidia.com/v1 \
    --api-key-env TEACHER_ENDPOINT_API_KEY \
    --targets "$targets" \
    --out "$root/$slug.jsonl" \
    --fail-out "$root/$slug.failed.jsonl" \
    --unresolved-out "$root/$slug.unresolved.jsonl" \
    --resume-from "$root/$slug.jsonl" \
    --resume-traces-from "$root/$slug.failed.jsonl" \
    --max-turns 12 \
    --max-tokens 4096 \
    --temperature 0.2 \
    --max-attempts 1 \
    --request-timeout 300 \
    --endpoint-max-retries 1 \
    --stop-on-rate-limit \
    --reward-mode grounded
  code=$?
  echo "SUPERVISOR_EXIT code=$code $(date -Is)"
  if [[ $code -eq 0 ]]; then
    exit 0
  fi
  if [[ $code -eq 75 ]]; then
    sleep "$cooldown"
  else
    sleep 60
  fi
done
