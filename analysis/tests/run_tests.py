"""Run every test in analysis/tests and keep the log in analysis/output/tests.txt.

Run from the repository's root: python analysis/tests/run_tests.py
"""
import io
import sys
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
OUTPUT = TESTS.parent / "output"


def main():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(TESTS), top_level_dir=str(TESTS))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(stream.getvalue().strip().splitlines()[-1])
    print(f"{result.testsRun} tests run; log written to {OUTPUT / 'tests.txt'}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
