# Archive and recovery

The archive-first cleanup was based on merged commit
`10093b32a651971d815dbe6fc81ee1939060146f`. It does not rewrite Git history.

## Verified external copy

Local archive on Vineeth's Mac:

`/Users/vikalluru/Documents/Code/Archives/rl-for-ai-research-20260930-cleanup/`

`files/` preserves original relative paths. `manifest.json` contains each path,
byte count and SHA256. All **580 files / 327,611,995 bytes** were copied and
checksum-verified before the 577 archival source files were removed from this
checkout. Three additional preimages were copied without removal: README,
the old 108/42 split and the frozen workshop split.

Manifest SHA256:
`5837f6d7ae5f91796a43c877030ee6f7c7b29b938d8c67b45f94f5e62e6b00d3`

The archive includes tracked and ignored result files, historical root reports,
logs, old plans and the old split generator. It does not include credentials,
virtual environments, the remote corpus or remote checkpoints. This is a local
archive, **not a cloud backup**. Previously tracked files also remain in Git
history. Previously ignored evidence depends on retaining this archive.

## Retained inputs

The following are exact-byte copies of archived inputs, not regenerated data:

| Current path | Archived source under `files/` |
|---|---|
| `split.json` | `artifacts/workshop/split.json` (96/12/42) |
| `data/targets.json` | `results/paper_2026_rl4llm/targets_frozen_rubric-v3.json` |
| `data/teacher_traces/sft.jsonl` | `results/teacher_traces/calculator_free_v4_20260930/sft_identified_teachers.jsonl` |

Dataset hashes and selected-example lineage are in `data/teacher_traces/`.
Default paths for current generation/evaluation commands now reference these
versioned inputs. Historical experiment configs/entry points may need explicit
restored inputs; they are not the supported latest recipe.

## Restore one artifact

Copy the required file from `files/<original-relative-path>` into a separate
local run directory. Verify its SHA256 against `manifest.json` before using it.
For previously tracked files, the pre-cleanup commit above is an alternative
recovery source. Do not restore an entire old split over the current frozen split.

Generated outputs now belong under ignored `results/` or outside this checkout.
The already running matched dev12 evaluation uses its independent remote code
snapshot; this local cleanup does not modify its files or serving process.
