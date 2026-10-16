"""Tests for src/code_metrics.py."""

from pathlib import Path
import pytest
from src.code_metrics import (
    analyze_python_file,
    analyze_directory,
    CodeMetricsError,
)


def test_analyze_python_file(tmp_path: Path):
    test_code = '''"""Sample module."""

def calculate(x: int, y: int) -> int:
    if x > 0:
        return x + y
    elif x < 0:
        return x - y
    return y

class Worker:
    def do_work(self):
        pass
'''
    py_file = tmp_path / "sample.py"
    py_file.write_text(test_code, encoding="utf-8")

    res = analyze_python_file(py_file)
    assert res["file"] == "sample.py"
    assert res["loc"] > 5
    assert res["sloc"] > 0
    assert res["total_complexity"] >= 3
    assert res["function_count"] >= 1
    assert res["class_count"] == 1
    assert res["maintainability_index"] > 50


def test_analyze_directory(tmp_path: Path):
    # Create sample structure
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    (tmp_path / "sub" / "b.py").write_text("def b(): pass\n", encoding="utf-8")

    summary = analyze_directory(tmp_path)
    assert summary["files_analyzed"] == 2
    assert summary["total_loc"] >= 2
    assert summary["status"] == "SUCCESS"


def test_empty_directory(tmp_path: Path):
    summary = analyze_directory(tmp_path)
    assert summary["files_analyzed"] == 0
    assert summary["status"] == "NO_PYTHON_FILES"
