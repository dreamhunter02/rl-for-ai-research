# Selected teacher traces

`sft.jsonl` is an exact-byte copy of
`calculator_free_v4_20260930/sft_identified_teachers.jsonl` from the external archive.
It contains 247 identified-teacher examples covering 73 of the frozen 96 training
questions. No dev/eval examples are included.

- Selected from 272 clean candidates; 25 synthetic-gold/unknown-teacher rows excluded.
- Calculator-free messages and six tool schemas; migrated traces, **not fresh
  native rollouts**. Editing and any synthetic-finish history remain in provenance.
- Selection used v4 rewards. Do not interpret the stored score as v6 validation.
- `provenance.jsonl` maps each selected question/teacher pair to its migration
  lineage and archived source record. `migration_manifest.json` preserves the
  original inventory's source hashes; its paths refer to historical locations.
- `selection_summary.json` is the original summary, including its historical
  output path. `dataset_manifest.json` records current hashes and coverage.

The quarantined, duplicate and rejected candidates were archived, not discarded.
See [recovery instructions](../../docs/ARCHIVE.md).
