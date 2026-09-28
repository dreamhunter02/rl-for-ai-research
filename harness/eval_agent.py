"""Compatibility entrypoint for the matched workshop evaluator.

Legacy --backend/--project/--limit invocations are intentionally rejected: they
used a separate parser and incomplete terminal contract. Historical behavior is
available at f29df1a; new B1/R1 comparisons must use reviewed targets.
"""
from workshop_eval import main

if __name__ == '__main__':
    main()
