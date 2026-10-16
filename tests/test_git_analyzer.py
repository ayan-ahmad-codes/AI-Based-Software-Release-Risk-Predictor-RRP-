"""Tests for src/git_analyzer.py."""

import subprocess
from pathlib import Path
import pytest
from src.git_analyzer import (
    get_latest_commit,
    get_commit_stats,
    get_recent_churn,
    run_git_cmd,
    GitAnalyzerError,
)


def test_git_cmd_on_current_repo():
    # software-rrp is a git repo
    root = Path(__file__).parent.parent
    commit = get_latest_commit(root)
    assert "hash" in commit
    assert len(commit["hash"]) == 40
    assert "commit_date" in commit


def test_git_commit_stats():
    root = Path(__file__).parent.parent
    stats = get_commit_stats(root, "HEAD")
    assert "files_changed" in stats
    assert "lines_added" in stats
    assert "lines_deleted" in stats
    assert isinstance(stats["total_churn"], int)


def test_recent_churn():
    root = Path(__file__).parent.parent
    churn = get_recent_churn(root, window_days=7)
    assert churn["window_days"] == 7
    assert churn["recent_commit_count"] >= 1


def test_invalid_repo_dir(tmp_path: Path):
    with pytest.raises(GitAnalyzerError):
        get_latest_commit(tmp_path)
