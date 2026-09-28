# FinanceBench × Nemotron 3.5 Lightning completion report

Protocol: `2026-09-27-repaired-v1`  
Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`  
Renderer: `nemotron3_ultra`  
Repository head at report generation: recorded by Git, with no credentials preserved.

## Scope decision

The execution brief separates harness repair from GRPO improvement. The repaired implementation, E0 preflight, E1 capability gates, repaired-base dev evaluation, and a minimal controlled E2 pilot were executed. A large R1 run was not launched because the controlled pilot did not establish a positive dev effect; this is an intentional gate, not a missing result. The previously run reward-v2 full experiment remains a negative diagnostic result and is not conflated with this protocol.

## Engineering gates

- E0: passed. The repository/model/renderer, corpus, active split documents, evaluator-only target artifacts, tool schemas, compact observations, pagination, and numeric sign/scale behavior were checked.
- E1: passed five deterministic gates in `e1_capability_tests.json`: validated terminal finish, supported oracle evidence, nonexistent-citation veto, evidence-free calculation/search not receiving full grounded credit, and malformed terminal fields failing safely without crashing rollouts.
- Unit suite: 32 tests passing; Python compilation and `git diff --check` passed during the final repair sequence.
- Corpus/page cache: 368 PDFs, 362 text fallbacks, and cached pages available; active split index construction filters to documents required by the split and preflight rejects missing/empty active documents.

## Controlled results

The minimal validated E2 run is `e2_minimal_validated_lr1e-5_20260927/`: 3 optimizer steps, 4 questions per batch, group size 4, 48 training rollouts, `train96`, `dev` evaluation after each update, learning rate `1e-5`, seed 0, per-turn generation cap 1024, trajectory cap 32768, and explicit zero-on-limit termination.

| checkpoint stage | train mean total reward | dev mean total reward | dev finish rate |
|---|---:|---:|---:|
| initial / iteration 0 | 0.01875 | 0.16458 | 0.4167 |
| after update 1 | 0.09688 | 0.11458 | 0.3636 |
| after update 2 | 0.07500 | 0.16458 | 0.4167 |

These 12-question dev measurements show no controlled improvement over the initial repaired-base sample. Training reward increased in one update while dev reward fell, so reward increase is not reported as learning. No final R1 checkpoint was selected and no eval42 GRPO claim is made.

## Baselines and evaluation boundary

The one-question repaired-base smoke is in `b1_base_dev1.json`; the 12-question repaired-base run used for the E2 initial comparison is preserved in the E2 iteration-000000 evaluation summaries. The fixed 42-question evaluation split was previously inspected and is therefore not described as untouched. The current repaired-base eval42 job is recorded separately when complete.

The old `results/local_eval/nemotron35_base_eval42_corrected_v1.json` is legacy diagnostic evidence from the earlier harness and is not used as a causal B0/B1 comparison. B0 under the repaired fixed-question protocol requires a separate execution from the original harness; no unsupported B0 number is substituted.

## Artifact index

- `protocol.md`, `split_manifest.json`, `e0_preflight.json`
- `e1_capability_tests.json`
- `E2_SUMMARY.json`, exact Tinker config, checkpoints, metrics, raw rollout summaries, trace events, and log trees under the E2 run directory
- `b1_base_dev1.json` and the repaired-base dev artifacts
- source implementation under `harness/`

All result files are intended to be reproducible artifacts; credentials, API keys, tokens, and connection strings are not included.
