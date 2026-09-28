"""Compatibility entrypoint: fail closed on unreviewed labels or missing corpus."""
import json
import os
from pathlib import Path
import financebench_harness as hb
from workshop_prepare import preflight

if __name__ == '__main__':
    targets = os.environ.get('FINANCEBENCH_TARGETS')
    if not targets:
        raise SystemExit('Set FINANCEBENCH_TARGETS to reviewed JSON targets; see docs/REVIEW_FIXES.md')
    result = preflight(os.environ.get('FINANCEBENCH_SPLIT', str(hb.BASE / 'split.json')), targets, corpus=True)
    print(json.dumps(result, indent=2))
