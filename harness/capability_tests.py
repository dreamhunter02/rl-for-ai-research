"""Run offline and SDK regression gates; absent SDK tests are a failing E1 gate."""
import importlib.util
import sys
import unittest
from pathlib import Path

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sdk_missing = importlib.util.find_spec('tinker_cookbook') is None
    if sdk_missing or result.skipped:
        print('E1 BLOCKED: all SDK contract tests must execute on the training machine.', file=sys.stderr)
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped and not sdk_missing else 1)
