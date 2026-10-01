"""Protect promoted frozen inputs and archived teacher provenance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_split_and_targets_are_self_contained():
    split = json.loads((ROOT / 'split.json').read_text())
    assert {k: len(v) for k, v in split.items()} == {'train': 96, 'dev': 12, 'eval': 42}
    ids = [r['financebench_id'] for rows in split.values() for r in rows]
    assert len(set(ids)) == 150
    targets = json.loads((ROOT / 'data/targets.json').read_text())
    assert set(targets) == set(ids)


def test_teacher_dataset_hashes_membership_and_lineage():
    folder = ROOT / 'data/teacher_traces'
    manifest = json.loads((folder / 'dataset_manifest.json').read_text())
    for filename, digest in manifest['files'].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((ROOT / 'split.json').read_bytes()).hexdigest() == manifest['split_sha256']
    rows = [json.loads(line) for line in (folder / 'sft.jsonl').read_text().splitlines()]
    provenance = [json.loads(line) for line in (folder / 'provenance.jsonl').read_text().splitlines()]
    keys = [(r['financebench_id'], r['teacher_model']) for r in rows]
    assert len(set(keys)) == len(rows) == manifest['rows'] == 247
    assert keys == [(r['financebench_id'], r['teacher_model']) for r in provenance]
    split = json.loads((ROOT / 'split.json').read_text())
    train = {r['financebench_id'] for r in split['train']}
    ids = {key[0] for key in keys}
    assert ids <= train
    assert len(ids) == manifest['unique_questions'] == 73
    assert sorted(train - ids) == manifest['missing_train_ids']
    for row in rows:
        names = {tool['function']['name'] for tool in row['tools']}
        assert names == {'bm25_search', 'grep_document', 'search_tables', 'read', 'read_table', 'finish'}
    assert manifest['selection_reward'] == 'v4'
