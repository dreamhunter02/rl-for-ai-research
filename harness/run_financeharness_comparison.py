"""Launch the matched comparison once; delay the minimal baseline until full jobs end."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone


def validate_config(config):
    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {'api_key', 'key', 'token', 'secret', 'password',
                                   'judge_api_key', 'component_judge_api_key'}:
                    raise ValueError('credentials must use file paths, never raw values')
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(config)
    for key in ('root', 'python', 'split_file', 'run_root', 'keyfile', 'jobs'):
        if key not in config:
            raise ValueError('missing config field: '+key)
    names = set()
    for job in config['jobs']:
        for key in ('name', 'label', 'model', 'base_url', 'workers'):
            if key not in job:
                raise ValueError('missing job field: '+key)
        name = job['name']
        if not name or Path(name).name != name or name in ('.', '..') or name in names:
            raise ValueError('job names must be unique directory names')
        names.add(name)
        if job.get('harness_mode', 'full') not in ('full', 'mini', 'minimal'):
            raise ValueError('unknown harness mode')
        if int(job['workers']) < 1:
            raise ValueError('workers must be positive')
        if job.get('reasoning_effort') not in (None, 'low', 'medium', 'high'):
            raise ValueError('unsupported reasoning effort')


def build_command(config, job):
    mode = job.get('harness_mode', 'full')
    command = [str(config['python']), str(Path(config['root'])/'harness/eval_stateful_harness.py'),
               '--split', 'all', '--split-file', str(config['split_file']),
               '--model', job['model'], '--base-url', job['base_url'],
               '--output-dir', str(Path(config['run_root'])/job['name']),
               '--max-tool-calls', '50', '--max-tokens', '1024',
               '--workers', str(job['workers']), '--score', '--submission-contract',
               '--require-citations', '--timeout', '120', '--max-retries', '1',
               '--extraction', 'plain', '--recent-turn-policy', 'complete',
               '--harness-mode', 'minimal' if mode in ('mini', 'minimal') else 'full']
    for key, flag in [('api_key_file', '--api-key-file'), ('reasoning_effort', '--reasoning-effort'),
                      ('resume_source', '--resume-source'),
                      ('resume_source_protocol', '--resume-source-protocol')]:
        if job.get(key):
            command.extend([flag, str(job[key])])
    if job.get('regenerate_all'):
        command.append('--regenerate-all')
    return command


def build_environment(config, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for key in ('OPENAI_API_KEY', 'NVIDIA_API_KEY', 'INFERENCE_API_KEY', 'COMPONENT_JUDGE_API_KEY'):
        env.pop(key, None)
    original = Path(config.get('original_root', config['root']))
    env.update(FINANCEBENCH_ROOT=str(config['root']),
               FINANCEBENCH_FILINGS=str(original/'filings'), FINANCEBENCH_TEXT=str(original/'text'),
               FINANCEBENCH_CACHE=str(original/'artifacts/page_cache'),
               COMPONENT_JUDGE_ENDPOINT=config.get('judge_endpoint', 'https://inference-api.nvidia.com/v1/chat/completions'),
               COMPONENT_JUDGE_MODEL=config.get('judge_model', 'openai/openai/gpt-5.6-terra'),
               COMPONENT_JUDGE_KEY_FILE=str(config['keyfile']),
               COMPONENT_JUDGE_CACHE=str(config.get('judge_cache', Path(config['run_root'])/'shared_judge_cache')),
               PYTHONUNBUFFERED='1')
    if config.get('rate_state'):
        env['FINANCEHARNESS_RATE_STATE'] = str(config['rate_state'])
        env['FINANCEHARNESS_REQUEST_INTERVAL'] = str(config.get('request_interval', 4))
    return env


def _now():
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _save(path, status):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as handle:
        json.dump(status, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(config, *, popen=None, sleep=None, pid_alive=None, report_builder=None):
    validate_config(config)
    popen = popen or subprocess.Popen
    sleep = sleep or time.sleep
    pid_alive = pid_alive or _pid_alive
    root = Path(config['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    path = root/'supervisor_status.json'
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    with (root/'supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('live supervisor already owns this run')
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get('state') == 'running' and pid_alive(previous['pid']):
                raise RuntimeError('live supervisor must not be restarted')
            if any(j.get('state') == 'running' and j.get('pid') and pid_alive(j['pid'])
                   for j in previous.get('jobs', {}).values()):
                raise RuntimeError('live evaluator must not be restarted')
            if previous.get('config_sha256') != fingerprint:
                raise ValueError('saved supervisor protocol differs from configuration')
            if previous.get('state') == 'running':
                previous.update(state='failed', ended_at=_now(), error='InterruptedSupervisor')
                for job in previous.get('jobs', {}).values():
                    if job.get('state') in ('running', 'pending'):
                        job.update(state='failed', ended_at=_now(), error='InterruptedSupervisor')
                _save(path, previous)
            return previous
        if report_builder is None:
            from financeharness_comparison import build_report
            report_builder = build_report
        status = {'pid': os.getpid(), 'state': 'running', 'started_at': _now(),
                  'config_sha256': fingerprint, 'root': str(config['root']),
                  'split_file': str(config['split_file']), 'jobs': {}}
        for job in config['jobs']:
            status['jobs'][job['name']] = {'state': 'pending', 'label': job['label'],
                                         'model': job['model'], 'base_url': job['base_url'],
                                         'workers': job['workers'],
                                         'harness_mode': job.get('harness_mode', 'full')}
        _save(path, status)
        env = build_environment(config)

        def group(jobs):
            active = {}
            for job in jobs:
                entry = status['jobs'][job['name']]
                command = build_command(config, job)
                log_path = root/(job['name']+'.log')
                entry.update(started_at=_now(), command=command, log=str(log_path))
                handle = None
                try:
                    handle = log_path.open('a')
                    process = popen(command, cwd=str(config['root']), env=env,
                                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    entry.update(state='running', pid=process.pid)
                    active[job['name']] = (process, handle)
                except Exception as exc:
                    if handle is not None:
                        handle.close()
                    entry.update(state='failed', ended_at=_now(), returncode=None, error=type(exc).__name__)
                _save(path, status)
            while active:
                for name, (process, handle) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    entry = status['jobs'][name]
                    child = {}
                    try:
                        child = json.loads((root/name/'status.json').read_text())
                    except (OSError, ValueError):
                        pass
                    complete = code == 0 and child.get('stage') == 'complete' and child.get('n') == config.get('expected_count', 150)
                    entry.update(state='complete' if complete else 'failed', returncode=code,
                                 ended_at=_now(), evaluator_status=child)
                    active.pop(name)
                    _save(path, status)
                if active:
                    sleep(1)

        try:
            group([j for j in config['jobs'] if j.get('harness_mode', 'full') == 'full'])
            group([j for j in config['jobs'] if j.get('harness_mode', 'full') in ('mini', 'minimal')])
            status['state'] = 'complete' if all(j['state'] == 'complete' for j in status['jobs'].values()) else 'failed'
            report_builder(config, root)
        except Exception as exc:
            status.update(state='failed', error=type(exc).__name__)
        status['ended_at'] = _now()
        _save(path, status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    status = run(config)
    print(json.dumps({'state': status['state'], 'run_root': str(config['run_root'])}), flush=True)
    return 0 if status['state'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
