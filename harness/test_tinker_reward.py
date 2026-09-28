"""SDK contract suite for the validated environment (replaces legacy history grading tests)."""
from pathlib import Path

def load_tests(loader, tests, pattern):
    directory=str(Path(__file__).resolve().parents[1] / 'tests')
    return loader.discover(directory, pattern='test_finance_env.py', top_level_dir=directory)
