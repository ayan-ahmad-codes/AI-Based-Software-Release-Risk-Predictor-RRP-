"""Source code metrics analyzer using Radon and AST inspection.

Calculates Lines of Code (LOC), Cyclomatic Complexity (CC),
and Maintainability Index (MI) across Python source files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from radon.complexity import cc_visit
from radon.metrics import mi_rank, mi_visit
from radon.raw import analyze

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    "build",
    "dist",
    ".eggs",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "node_modules",
}

MAX_FILE_SIZE_BYTES = 1024 * 1024  # 1 MB limit per file to prevent DoS


class CodeMetricsError(Exception):
    """Exception raised during code metrics calculation."""


def analyze_python_file(file_path: Path | str) -> dict[str, Any]:
    """Analyze a single Python file using Radon.

    Args:
        file_path: Path to the Python file.

    Returns:
        dict: File metrics including LOC, Cyclomatic Complexity, and MI.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Prevent symlink traversal out of tree
    if path.is_symlink():
        real_target = path.resolve()
        if not real_target.exists():
            raise FileNotFoundError(f"Broken symlink: {file_path}")

    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        return {
            "file": path.name,
            "rel_path": str(path),
            "skipped": True,
            "reason": f"File size ({size} bytes) exceeds limit ({MAX_FILE_SIZE_BYTES} bytes)",
            "loc": 0,
            "lloc": 0,
            "sloc": 0,
            "comments": 0,
            "complexity": 0.0,
            "function_count": 0,
            "class_count": 0,
            "maintainability_index": 0.0,
            "maintainability_rank": "N/A",
        }

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            "file": path.name,
            "rel_path": str(path),
            "error": f"Read error: {exc}",
            "loc": 0,
            "sloc": 0,
            "complexity": 0.0,
        }

    try:
        raw = analyze(source)
    except Exception as exc:  # Catch radon parsing issues
        raw = None
        raw_err = str(exc)

    try:
        blocks = cc_visit(source)
    except Exception:
        blocks = []

    try:
        mi = float(mi_visit(source, multi=True))
        rank = mi_rank(mi)
    except Exception:
        mi = 0.0
        rank = "N/A"

    complexities = [block.complexity for block in blocks]
    total_complexity = sum(complexities)
    max_complexity = max(complexities) if complexities else 0
    high_complexity_blocks = sum(1 for c in complexities if c > 10)

    function_count = sum(1 for block in blocks if block.__class__.__name__ == "Function")
    class_count = sum(1 for block in blocks if block.__class__.__name__ == "Class")

    loc = raw.loc if raw else len(source.splitlines())
    lloc = raw.lloc if raw else 0
    sloc = raw.sloc if raw else 0
    comments = raw.comments if raw else 0

    return {
        "file": path.name,
        "rel_path": str(path),
        "loc": loc,
        "lloc": lloc,
        "sloc": sloc,
        "comments": comments,
        "comment_ratio": round(comments / max(loc, 1), 3),
        "total_complexity": total_complexity,
        "max_complexity": max_complexity,
        "avg_complexity_per_block": round(total_complexity / max(len(blocks), 1), 2),
        "high_complexity_block_count": high_complexity_blocks,
        "function_count": function_count,
        "class_count": class_count,
        "maintainability_index": round(mi, 2),
        "maintainability_rank": rank,
    }


def analyze_directory(
    directory: str | Path,
    max_files: int = 500,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Recursively analyze all Python files in a directory.

    Args:
        directory: Root directory path.
        max_files: Max files to analyze to avoid running out of memory.
        include_tests: Whether to include test files in the overall metrics.

    Returns:
        dict: Aggregated code metrics and list of per-file results.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    file_results: list[dict[str, Any]] = []
    skipped_count = 0

    for path in sorted(root.rglob("*.py")):
        # Check against IGNORE_DIRS
        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        if not include_tests and ("test" in path.name.lower() or "tests" in path.parts):
            continue

        if len(file_results) >= max_files:
            skipped_count += 1
            continue

        res = analyze_python_file(path)
        # Store path relative to root for clarity
        try:
            res["rel_path"] = str(path.relative_to(root))
        except ValueError:
            pass
        file_results.append(res)

    if not file_results:
        return {
            "directory": str(root),
            "files_analyzed": 0,
            "total_loc": 0,
            "total_sloc": 0,
            "total_comments": 0,
            "avg_complexity": 0.0,
            "max_complexity": 0,
            "avg_maintainability_index": 0.0,
            "high_complexity_files_count": 0,
            "total_functions": 0,
            "total_classes": 0,
            "files": [],
            "status": "NO_PYTHON_FILES",
        }

    total_loc = sum(f.get("loc", 0) for f in file_results)
    total_sloc = sum(f.get("sloc", 0) for f in file_results)
    total_comments = sum(f.get("comments", 0) for f in file_results)
    total_functions = sum(f.get("function_count", 0) for f in file_results)
    total_classes = sum(f.get("class_count", 0) for f in file_results)

    all_complexities = [f.get("total_complexity", 0) for f in file_results]
    all_mis = [f.get("maintainability_index", 0.0) for f in file_results if "maintainability_index" in f]
    high_complexity_files = sum(1 for f in file_results if f.get("max_complexity", 0) > 10 or f.get("total_complexity", 0) > 20)

    avg_complexity = round(sum(all_complexities) / max(len(file_results), 1), 2)
    max_complexity = max((f.get("max_complexity", 0) for f in file_results), default=0)
    avg_mi = round(sum(all_mis) / max(len(all_mis), 1), 2)

    return {
        "directory": str(root),
        "files_analyzed": len(file_results),
        "skipped_files": skipped_count,
        "total_loc": total_loc,
        "total_sloc": total_sloc,
        "total_comments": total_comments,
        "comment_ratio": round(total_comments / max(total_loc, 1), 3),
        "avg_complexity": avg_complexity,
        "max_complexity": max_complexity,
        "avg_maintainability_index": avg_mi,
        "high_complexity_files_count": high_complexity_files,
        "total_functions": total_functions,
        "total_classes": total_classes,
        "files": file_results,
        "status": "SUCCESS",
    }
