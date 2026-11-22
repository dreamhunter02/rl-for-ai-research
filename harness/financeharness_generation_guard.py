"""Generation-only cancellation boundary. Never scores, launches or reruns actors."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time


def sentinel_bytes():
    return b'"""Scoring-only user cancellation; original module preserved separately."""\nraise SystemExit("JUDGING_CANCELLED_BY_USER: generation preserved; no judge constructed")\n'


def inspect_process(pid):
    root = Path('/proc')/str(int(pid))
    try:
        fields = (root/'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        argv = [part.decode() for part in (root/'cmdline').read_bytes().split(b'\0') if part]
        return {'pid': int(pid), 'argv': argv, 'cwd': os.readlink(root/'cwd'), 'start_ticks': fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def terminate_verified(pin):
    """A pidfd binds the signal to this process, never a later reused PID."""
    descriptor = os.pidfd_open(pin['pid'])
    try:
        if inspect_process(pin['pid']) != pin:
            raise ProcessLookupError('process identity changed')
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    finally:
        os.close(descriptor)


def _raw_ids(job, state, expected):
    path = Path(job['source_directory'])/'base.jsonl'
    stat = path.stat()
    stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    if state.get('raw_cache', {}).get('stat') == stamp:
        return set(state['raw_cache']['ids'])
    ids = set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            fid = row.get('financebench_id')
            if fid in ids or fid not in expected or row.get('model') != job['model']:
                raise ValueError('invalid raw membership or model')
            ids.add(fid)
    after = path.stat()
    if stamp != [after.st_size, after.st_mtime_ns, after.st_ino]:
        raise json.JSONDecodeError('raw changed during read', '', 0)
    state['raw_cache'] = {'stat': stamp, 'ids': sorted(ids)}
    return ids


def _protocol_valid(job):
    blob = (Path(job['source_directory'])/'protocol.json').read_bytes()
    if hashlib.sha256(blob).hexdigest() != job['protocol_sha256']:
        return False
    protocol = json.loads(blob)
    expected = dict(model=job['model'], max_tool_calls=50, max_tokens=1024, temperature=0,
                    seed=0, forced_finish=False, budget_cues=False)
    return all(protocol.get(key) == value for key, value in expected.items())


def _archive(job):
    source = Path(job['source_directory'])
    target = Path(job['archive_directory'])
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copy2(path, target/path.name)
            hashes[path.name] = hashlib.sha256((target/path.name).read_bytes()).hexdigest()
    lineage = dict(source_directory=str(source), selection='all_150_actor_outcomes_no_filter',
        scoring='cancelled_by_user_no_judge_launch', originals_preserved=True, sha256=hashes)
    (target/'generation_only_lineage.json').write_text(json.dumps(lineage, indent=2)+'\n')
    return lineage


def advance_actor(job, state, expected_ids, *, inspect=inspect_process, terminate=terminate_verified):
    state = dict(state)
    if state.get('state') in ('blocked', 'generation_complete_judging_cancelled'):
        return state
    def blocked(reason):
        return dict(state, state='blocked', blocker=reason)
    current = inspect(job['process']['pid'])
    if current is not None and current != job['process']:
        return blocked('process_identity_mismatch')
    try:
        if not _protocol_valid(job):
            return blocked('financial_protocol_mismatch')
        ids = _raw_ids(job, state, expected_ids)
    except json.JSONDecodeError:
        return dict(state, state='waiting_generation', blocker='raw_write_in_progress')
    except ValueError:
        return blocked('invalid_raw_membership_or_model')
    except OSError:
        ids = set()
    state['raw_count'] = len(ids)
    if ids != set(expected_ids):
        return blocked('actor_missing_before_full_generation') if current is None else dict(state, state='waiting_generation')
    if current is None:
        try:
            lineage = _archive(job)
        except OSError as exc:
            return blocked('archive_' + type(exc).__name__)
        return dict(state, state='generation_complete_judging_cancelled', lineage=lineage,
                    exit_observation='already_terminal_at_full_boundary')
    if state.get('state') == 'stopping_at_boundary':
        return state
    if inspect(job['process']['pid']) != job['process']:
        return blocked('process_changed_at_boundary')
    try:
        terminate(job['process'])
    except OSError as exc:
        return blocked('boundary_signal_' + type(exc).__name__)
    return dict(state, state='stopping_at_boundary', signaled_pid=job['process']['pid'])


def run(config):
    poll = config.get('poll_seconds', 1)
    if not 0 < poll <= 1 or config.get('judging') is not False:
        raise ValueError('generation-only guard requires judging=false and poll <=1s')
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('safe Linux pidfd signaling required')
    split_blob = Path(config['split_file']).read_bytes()
    if hashlib.sha256(split_blob).hexdigest() != config['split_sha256']:
        raise ValueError('frozen split identity mismatch')
    split = json.loads(split_blob)
    ids = [r['financebench_id'] for key in ('train', 'dev', 'eval') for r in split[key]]
    if len(ids) != 150 or len(set(ids)) != 150:
        raise ValueError('exact 150 frozen unique IDs required')
    jobs = config['actors']
    if len(jobs) != 4 or len({j['process']['pid'] for j in jobs}) != 4:
        raise ValueError('four uniquely pinned actors required')
    state_path = Path(config['state_file'])
    lock = state_path.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    states = {}
    while True:
        for job in jobs:
            states[job['name']] = advance_actor(job, states.get(job['name'], {}), set(ids))
        status = {'judging': False, 'actors': states, 'poll_seconds': poll}
        with tempfile.NamedTemporaryFile(mode='w', dir=state_path.parent, delete=False) as stream:
            json.dump(status, stream, indent=2)
            temporary = stream.name
        os.replace(temporary, state_path)
        if all(s['state'] in ('blocked', 'generation_complete_judging_cancelled') for s in states.values()):
            return status
        time.sleep(poll)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text()))), flush=True)
