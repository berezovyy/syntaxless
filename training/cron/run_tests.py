"""Discover the training harness's argument-free test functions with stdlib unittest."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import unittest


def suite() -> unittest.TestSuite:
    tests = unittest.TestSuite()
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        module = importlib.import_module(path.stem)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and function.__module__ == module.__name__:
                tests.addTest(unittest.FunctionTestCase(function))
    if tests.countTestCases() == 0:
        raise RuntimeError("No Python training tests found")
    return tests


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(0 if result.wasSuccessful() else 1)
