"""Seed fresh provider-recovery runs without selecting on answer quality.

The caller must first bind the new output configuration with verify_run_config.
This module does not create or modify configuration sidecars. Completed records
are retained regardless of reward, correctness, finish acceptance, or budgets.
Only named infrastructure exceptions are retried in normal recovery. An explicit
regenerate_all mode records a full protocol restart rather than selective reuse.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


PROVIDER_ERRORS = frozenset({
    'RateLimitError', 'APITimeoutError', 'APIConnectionError', 'BadRequestError',
    'AuthenticationError', 'PermissionDeniedError', 'NotFoundError',
    'ConflictError', 'UnprocessableEntityError', 'InternalServerError',
    'APIStatusError', 'ServiceUnavailableError', 'Timeout', 'TimeoutError',
    'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'PoolTimeout',
    'ConnectError', 'ReadError', 'WriteError', 'RemoteProtocolError',
})


def select_reusable(source_records: Iterable[dict], expected_ids: Iterable[str],
                    regenerate_all: bool = False) -> list[dict]:
    """Validate a complete source run and retain every non-provider-error record.

    Expected IDs are supplied by the frozen benchmark manifest, never inferred
    from whichever rows completed successfully. Returned dictionaries are the
    original objects and no episode fields are added or changed.
    """
    records = list(source_records)
    expected = list(expected_ids)
    if len(set(expected)) != len(expected):
        raise ValueError('Expected question IDs must be unique')
    ids = [record.get('financebench_id') for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Source question IDs must be unique')
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError('Source must contain exactly the frozen expected question IDs')
    models = {record.get('model') for record in records}
    if len(models) > 1:
        raise ValueError('Source records must use one model identity')
    if regenerate_all:
        return []
    reusable = []
    for record in records:
        reason = record.get('termination_reason')
        if not isinstance(reason, str) or not reason:
            raise ValueError('Source record lacks a termination reason')
        if reason == 'error':
            if record.get('error') not in PROVIDER_ERRORS:
                raise ValueError('Error is not a recognized provider infrastructure exception')
        else:
            reusable.append(record)
    return reusable


def seed_recovery(source_jsonl: str | Path, source_protocol: str | Path,
                  new_basepath: str | Path, expected_ids: Iterable[str],
                  regenerate_all: bool = False) -> dict:
    """Exclusively create a fresh JSONL and separate recovery_lineage.json.

    Source JSONL and protocol bytes are hashed and never modified. Retained
    record lines are copied byte-for-byte, except a missing final newline is
    added to allow safe appending. Record JSON is unchanged. Lineage stays
    outside episodes.
    Existing outputs or lineage are refused, including empty output files.
    The caller owns configuration verification and actor scheduling.
    """
    source = Path(source_jsonl)
    protocol = Path(source_protocol)
    target = Path(new_basepath)
    lineage_path = target.parent / 'recovery_lineage.json'
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Recovery output already exists: {target}')
    if lineage_path.exists() or lineage_path.is_symlink():
        raise FileExistsError(f'Recovery lineage already exists: {lineage_path}')
    raw_source = source.read_bytes()
    raw_protocol = protocol.read_bytes()
    lines = [line for line in raw_source.splitlines(keepends=True) if line.strip()]
    records = [json.loads(line) for line in lines]
    expected = list(expected_ids)
    retained = select_reusable(records, expected, regenerate_all=regenerate_all)
    retained_ids = [record['financebench_id'] for record in retained]
    retained_set = set(retained_ids)
    retry_ids = [qid for qid in expected if qid not in retained_set]
    lineage = {
        'schema_version': 'provider-recovery-v1',
        'selection_rule': ('explicit_full_regeneration' if regenerate_all else
                           'retain_all_nonerror_records_retry_named_provider_infrastructure_errors'),
        'selection_explanation': (
            'No selection uses rewards, correctness, accepted finish, or tool/token budgets. '
            'Normal recovery retries only named provider infrastructure exceptions; '
            'explicit full regeneration retains no source episodes.'),
        'regenerate_all': bool(regenerate_all),
        'source_jsonl': str(source.resolve()),
        'source_protocol': str(protocol.resolve()),
        'source_jsonl_sha256': hashlib.sha256(raw_source).hexdigest(),
        'source_protocol_sha256': hashlib.sha256(raw_protocol).hexdigest(),
        'output_jsonl': str(target.resolve()),
        'source_count': len(records),
        'retained_count': len(retained_ids),
        'retry_count': len(retry_ids),
        'retained_ids': retained_ids,
        'retry_ids': retry_ids,
        'provider_error_types': sorted({record.get('error') for record in records
                                        if record.get('termination_reason') == 'error'
                                        and isinstance(record.get('error'), str)}),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as destination:
        for line, record in zip(lines, records):
            if record['financebench_id'] in retained_set:
                destination.write(line)
                if not line.endswith(b'\n'):
                    destination.write(b'\n')
    with lineage_path.open('x', encoding='utf-8') as destination:
        json.dump(lineage, destination, indent=2, sort_keys=True)
        destination.write('\n')
    return lineage
