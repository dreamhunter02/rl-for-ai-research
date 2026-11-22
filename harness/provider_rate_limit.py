"""Opt-in, process-shared Hub request pacing and HTTP 429 cooldowns.

The state file contains timestamps and counters only. Waiting happens outside
the flock so every actor and judge can publish a provider cooldown immediately.
"""
from __future__ import annotations

from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import fcntl
import json
import math
import os
from pathlib import Path
import time
import urllib.error
from urllib.parse import urlsplit


def is_http_429(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429
    response = getattr(exc, 'response', None)
    return (getattr(exc, 'status_code', None) == 429 or getattr(exc, 'code', None) == 429
            or getattr(response, 'status_code', None) == 429)


class RateGovernor:
    def __init__(self, state_path, *, interval=2.0, clock=time.time, sleep=time.sleep):
        if not math.isfinite(interval) or interval < 2:
            raise ValueError('request interval must be at least two seconds')
        self.path = Path(state_path)
        self.interval = interval
        self.clock = clock
        self.sleep = sleep

    @contextmanager
    def _state(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a+') as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                text = fh.read()
                state = json.loads(text) if text.strip() else {}
                if not isinstance(state, dict):
                    raise ValueError('invalid provider rate state')
                for key in ('next_request_at', 'cooldown_until', 'consecutive_429'):
                    value = state.setdefault(key, 0)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError('invalid provider rate state')
                yield state
                fh.seek(0)
                fh.truncate()
                json.dump(state, fh, sort_keys=True)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def acquire(self):
        while True:
            with self._state() as state:
                now = self.clock()
                wait = max(state['next_request_at'], state['cooldown_until']) - now
                if wait <= 0:
                    state['next_request_at'] = now + self.interval
                    return
            self.sleep(min(wait, 10.0))

    def record_429(self, exc):
        if isinstance(exc, urllib.error.HTTPError):
            headers = exc.headers or {}
        else:
            headers = getattr(exc, 'headers', None) or getattr(getattr(exc, 'response', None), 'headers', {}) or {}
        retry_after = next((value for key, value in headers.items() if key.lower() == 'retry-after'), None)
        now = self.clock()
        delay = 0
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except (ValueError, TypeError):
                try:
                    delay = parsedate_to_datetime(str(retry_after)).timestamp() - now
                except (ValueError, TypeError, OverflowError):
                    delay = 0
            if not math.isfinite(delay) or delay < 0:
                delay = 0
        with self._state() as state:
            strikes = int(state['consecutive_429']) + 1
            state['consecutive_429'] = strikes
            cooldown = max(60 * (2 ** min(strikes - 1, 2)), delay)
            state['cooldown_until'] = max(state['cooldown_until'], now + cooldown)

    def call(self, operation, *, max_retries=6):
        for attempt in range(max_retries + 1):
            self.acquire()
            acquired_at = self.clock()
            try:
                result = operation()
            except Exception as exc:
                if not is_http_429(exc):
                    raise
                self.record_429(exc)
                if attempt == max_retries:
                    raise
            else:
                with self._state() as state:
                    if acquired_at >= state['cooldown_until']:
                        state['consecutive_429'] = 0
                return result


def get_rate_governor(endpoint):
    path = os.environ.get('FINANCEHARNESS_RATE_STATE')
    url = urlsplit(str(endpoint or ''))
    if not path or url.scheme != 'https' or url.hostname != 'inference-api.nvidia.com':
        return None
    return RateGovernor(path, interval=float(os.environ.get('FINANCEHARNESS_REQUEST_INTERVAL', '2')))
