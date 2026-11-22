"""Boundary-only handoff of existing local actors to governed score-only jobs.

Never stops an unfinished actor, reruns actor records, edits the old snapshot,
or builds a report. Deployment and starting this guard require parent approval.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from eval_stateful_harness import load_record_map, selected_rows, validate_recovery_protocol
from run_financeharness_comparison import build_command, build_environment


def inspect_pid(pid):
    result = subprocess.run(['ps', '-p', str(int(pid)), '-o', 'stat=', '-o', 'args='], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    status, _, command = result.stdout.strip().partition(' ')
    if status.startswith('Z'):
        return None
    return shlex.split(command.strip())


def _blocked(state, reason):
    return dict(state, state='blocked', blocker=reason)


def _validated_records(job, expected_ids):
    records = load_record_map(job['source_jsonl'])
    if not set(records) <= set(expected_ids):
        raise ValueError('foreign_raw_ids')
    if any(r.get('model') != job['expected_protocol']['model'] for r in records.values()):
        raise ValueError('raw_model_mismatch')
    return records


def _raw_progress(job, state, expected_ids):
    path = Path(job['source_jsonl'])
    def signature():
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    stamp = signature()
    cached = state.get('raw_cache', {})
    if cached.get('stat') == stamp:
        return set(cached['ids'])
    records = _validated_records(job, expected_ids)
    if signature() != stamp:
        raise json.JSONDecodeError('raw changed during read', '', 0)
    state['raw_cache'] = dict(stat=stamp, ids=list(records))
    return set(records)


def stage_score_only(job, expected_ids):
    """Copy every completed outcome into an exclusive fresh score directory."""
    validate_recovery_protocol(job['source_protocol'], job['expected_protocol'])
    records = _validated_records(job, expected_ids)
    if set(records) != set(expected_ids):
        raise ValueError('full_frozen_raw_required')
    source = Path(job['source_jsonl'])
    protocol = Path(job['source_protocol'])
    raw = source.read_bytes()
    protocol_bytes = protocol.read_bytes()
    target = Path(job['target_directory'])
    target.mkdir(parents=True, exist_ok=False)
    lineage = dict(schema_version='phase-score-handoff-v1',
        source_jsonl=str(source), source_protocol=str(protocol),
        source_jsonl_sha256=hashlib.sha256(raw).hexdigest(),
        source_protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        retained_ids=list(records), retained_count=len(records),
        selection_rule='all_actor_records_retained_no_outcome_filter',
        original_scoring_artifacts='preserved_in_original_directory',
        output_jsonl=str(target / 'base.jsonl'))
    with (target / 'base.jsonl').open('xb') as fh:
        fh.write(raw)
    with (target / 'protocol.json').open('xb') as fh:
        fh.write(protocol_bytes)
    with (target / 'phase_handoff_lineage.json').open('x') as fh:
        json.dump(lineage, fh, indent=2, sort_keys=True)
    return lineage


def advance_actor(job, state, expected_ids, *, inspect_pid=inspect_pid, send_signal=os.kill):
    """One observation step; a blocked transition never retries automatically."""
    state = dict(state)
    phase = state.get('state', 'waiting_generation')
    if phase in ('blocked', 'ready_scoring', 'scoring', 'complete', 'failed'):
        return state
    current = inspect_pid(job['pid'])
    if phase == 'stopping_at_boundary':
        if current is None:
            try:
                lineage = stage_score_only(job, expected_ids)
            except Exception as exc:
                return _blocked(state, 'score_staging_' + type(exc).__name__)
            return dict(state, state='ready_scoring', generation_terminal=True,
                        lineage=lineage)
        if current != job['expected_command']:
            return _blocked(state, 'pid_reused_before_exit_confirmation')
        return state
    if current is None:
        return _blocked(state, 'actor_pid_missing')
    if current != job['expected_command']:
        return _blocked(state, 'actor_pid_command_mismatch')
    try:
        records = _raw_progress(job, state, expected_ids)
    except json.JSONDecodeError:
        return dict(state, state='waiting_generation', blocker='raw_write_in_progress')
    except ValueError:
        return _blocked(state, 'invalid_raw_membership_or_model')
    except OSError:
        return dict(state, state='waiting_generation', raw_count=0)
    state.update(state='waiting_generation', raw_count=len(records))
    if set(records) != set(expected_ids):
        status_path = Path(job['source_jsonl']).parent / 'status.json'
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
            except (OSError, ValueError):
                status = {}
            if status.get('stage') in ('complete', 'failed', 'error'):
                return _blocked(state, 'actor_terminal_without_full_raw')
        return state
    try:
        validate_recovery_protocol(job['source_protocol'], job['expected_protocol'])
    except (ValueError, OSError):
        return _blocked(state, 'financial_protocol_mismatch')
    if Path(job['target_directory']).exists():
        return _blocked(state, 'score_target_already_exists')
    if inspect_pid(job['pid']) != job['expected_command']:
        return _blocked(state, 'actor_changed_at_boundary')
    try:
        send_signal(job['pid'], signal.SIGTERM)
    except ProcessLookupError:
        return _blocked(state, 'actor_disappeared_at_boundary')
    except OSError:
        return _blocked(state, 'boundary_signal_failed')
    return dict(state, state='stopping_at_boundary', signaled_pid=job['pid'],
                signal='SIGTERM', raw_count=len(records))


def baseline_ready(full_statuses, runtime_proof):
    if len(full_statuses) != 5 or not all(
            s.get('generation_terminal') and s.get('scoring_terminal') for s in full_statuses):
        return False, 'full_phases_not_terminal'
    if not all(s.get('data_complete') for s in full_statuses):
        return False, 'full_dataset_incomplete'
    keys = ('checkpoint_expected', 'checkpoint_observed', 'runtime_expected', 'runtime_observed')
    if not runtime_proof.get('base_url') or not all(runtime_proof.get(key) for key in keys):
        return False, 'minimal_runtime_unverified'
    if (runtime_proof['checkpoint_expected'] != runtime_proof['checkpoint_observed']
            or runtime_proof['runtime_expected'] != runtime_proof['runtime_observed']):
        return False, 'minimal_runtime_mismatch'
    return True, None


def score_command(config, job):
    protocol = job['expected_protocol']
    command = [str(config['python']), str(Path(config['root']) / 'harness/eval_stateful_harness.py'),
        '--score-only', '--split', 'all', '--split-file', str(config['split_file']),
        '--model', protocol['model'], '--base-url', protocol['base_url'],
        '--output-dir', str(job['target_directory']), '--max-tool-calls', str(protocol['max_tool_calls']),
        '--max-tokens', str(protocol['max_tokens']), '--workers', '1', '--submission-contract',
        '--require-citations', '--extraction', protocol['extraction_mode'],
        '--recent-turn-policy', protocol['recent_turn_policy'], '--harness-mode', protocol['harness_mode']]
    return command


def _completed_data(directory, expected_ids):
    raw = load_record_map(Path(directory) / 'base.jsonl')
    scored = load_record_map(Path(directory) / 'scored.jsonl')
    return set(raw) == set(scored) == set(expected_ids) and all(
        isinstance(r.get(kind), dict) and type(r[kind].get('unresolved')) is bool
        for r in scored.values() for kind in ('score', 'correctness_score'))


def external_phase(job, state, expected_ids, *, inspect_pid=inspect_pid):
    state = dict(state)
    current = inspect_pid(job['pid'])
    if current is not None and current != job['expected_command']:
        return dict(state, state='blocked', blocker='external_pid_command_mismatch', data_complete=False)
    try:
        child = json.loads(Path(job['status_path']).read_text())
    except (OSError, ValueError):
        child = {}
    if current is not None:
        return dict(state, state='waiting_exit' if child.get('stage') == 'complete' else 'waiting',
                    generation_terminal=False, scoring_terminal=False, data_complete=False)
    if child.get('stage') != 'complete' or child.get('n') != len(expected_ids):
        return dict(state, state='blocked', blocker='external_missing_or_terminal_incomplete',
                    generation_terminal=True, scoring_terminal=True, data_complete=False)
    directory = Path(job['status_path']).parent
    try:
        stamp = [[p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino]
                 for p in (directory / 'base.jsonl', directory / 'scored.jsonl')]
        complete = state.get('data_complete') if state.get('validated_stat') == stamp else _completed_data(directory, expected_ids)
    except (OSError, ValueError):
        complete, stamp = False, None
    return dict(state, state='complete' if complete else 'blocked',
                blocker=None if complete else 'external_raw_or_scores_incomplete', validated_stat=stamp,
                generation_terminal=True, scoring_terminal=True, data_complete=complete)


def _save(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(temporary, path)


def validate_guard_config(config):
    if not (Path(config['root']) / 'harness/eval_stateful_harness.py').is_file():
        raise ValueError('governed evaluator must be deployed before starting the guard')
    if not Path(config['python']).is_file():
        raise ValueError('configured Python must be deployed')
    interval = float(config.get('request_interval', 4))
    poll = float(config.get('poll_seconds', 1))
    if not config.get('rate_state') or not math.isfinite(interval) or interval < 4:
        raise ValueError('four-second shared rate state required')
    if not math.isfinite(poll) or not 0 < poll <= 10:
        raise ValueError('poll interval must be positive and at most ten seconds')
    pids = [config['original_supervisor_pid']]
    for job in config['actors'] + config.get('external_full_jobs', []):
        pids.append(job['pid'])
        if not isinstance(job.get('expected_command'), list) or not job['expected_command']:
            raise ValueError('exact nonempty process argv required')
    if any(type(pid) is not int or pid <= 1 for pid in pids) or len(set(pids)) != len(pids):
        raise ValueError('positive unique process PIDs required')


def run(config, *, inspect_pid=inspect_pid, send_signal=os.kill, popen=subprocess.Popen, sleep=time.sleep):
    """Run only after original supervisor has stopped; no scoring job restarts."""
    validate_guard_config(config)
    expected = {r['financebench_id'] for r in selected_rows(config['split_file'], 'all')}
    root = Path(config['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'phase_guard_status.json'
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    with (root / 'phase_guard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get('config_sha256') != identity:
                raise ValueError('phase guard configuration differs')
            return saved  # Never restart scoring or actor control from stale state.
        state = dict(state='waiting', config_sha256=identity, pid=os.getpid(),
                     jobs={j['name']: {} for j in config['actors']}, external_jobs={}, baseline={})
        if inspect_pid(config['original_supervisor_pid']) is not None:
            state.update(state='blocked', blocker='original_supervisor_must_stop_first')
            _save(path, state)
            return state
        env = build_environment(config)
        if not config.get('rate_state') or float(config.get('request_interval', 4)) < 4:
            raise ValueError('four-second shared rate state required')
        active = {}
        while True:
            for job in config['actors']:
                name = job['name']
                entry = advance_actor(job, state['jobs'][name], expected,
                                      inspect_pid=inspect_pid, send_signal=send_signal)
                state['jobs'][name] = entry
                if entry.get('state') == 'ready_scoring':
                    command = score_command(config, job)
                    handle = (root / (name + '.score.log')).open('a')
                    try:
                        process = popen(command, cwd=config['root'], env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                    except Exception as exc:
                        handle.close()
                        entry.update(state='failed', scoring_terminal=True, blocker='score_launch_' + type(exc).__name__)
                    else:
                        active[name] = (process, handle, job)
                        entry.update(state='scoring', scoring_pid=process.pid, command=command)
            for name, (process, handle, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                entry = state['jobs'][name]
                try:
                    child = json.loads((Path(job['target_directory']) / 'status.json').read_text())
                except (OSError, ValueError):
                    child = {}
                try:
                    complete = code == 0 and child.get('stage') == 'complete' and child.get('n') == 150 and _completed_data(job['target_directory'], expected)
                except (OSError, ValueError):
                    complete = False
                entry.update(state='complete' if complete else 'failed', scoring_terminal=True,
                             returncode=code, evaluator_status=child, data_complete=complete,
                             blocker=None if complete else 'score_process_failed_or_incomplete')
                active.pop(name)
            full = list(state['jobs'].values())
            for external in config.get('external_full_jobs', []):
                entry = external_phase(external, state['external_jobs'].get(external['name'], {}), expected,
                                       inspect_pid=inspect_pid)
                state['external_jobs'][external['name']] = entry
                full.append(entry)
            proof = config.get('minimal_runtime_proof', {})
            ready, blocker = baseline_ready(full, proof)
            state['baseline'].update(state='ready' if ready else 'blocked', blocker=blocker)
            if ready:
                # Parent can review the exact disclosed endpoint and launch command.
                minimal = config.get('minimal_job')
                if minimal:
                    command = build_command(config, dict(minimal, base_url=proof['base_url'], harness_mode='minimal'))
                    state['baseline'].update(state='ready_for_parent_launch', command=command,
                                             runtime_proof=proof, base_url=proof['base_url'])
                state['state'] = 'ready_for_baseline'
            elif any(e.get('state') == 'blocked' for e in full):
                state['state'] = 'blocked'
            elif blocker == 'full_dataset_incomplete':
                state.update(state='blocked', blocker=blocker)
            elif all(e.get('scoring_terminal') for e in state['jobs'].values()) and not active:
                state['state'] = 'waiting_external_or_runtime'
            _save(path, state)
            all_locals_terminal = all(e.get('state') == 'blocked' or e.get('scoring_terminal') for e in state['jobs'].values())
            if state['state'] == 'ready_for_baseline' or (state['state'] == 'blocked' and all_locals_terminal and not active):
                return state
            sleep(min(float(config.get('poll_seconds', 1)), 10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    state = run(json.loads(args.config.read_text()))
    print(json.dumps({'state': state['state'], 'blocker': state.get('blocker')}))


if __name__ == '__main__':
    main()
