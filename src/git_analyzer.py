"""Git repository history and commit churn analyzer.

Executes controlled git operations via subprocess to safely extract commit metadata,
diff statistics, and churn frequency.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


class GitAnalyzerError(Exception):
    """Exception raised for git operations failures."""


def run_git_cmd(args: list[str], cwd: Path | str, timeout: int = 30) -> str:
    """Run a git command safely using an argument list (no shell=True)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise GitAnalyzerError(f"Git command failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitAnalyzerError(f"Git command timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise GitAnalyzerError("Git executable not found in PATH.") from exc


def get_latest_commit(repo_path: Path | str) -> dict[str, str]:
    """Retrieve hash, author, date, and message of the HEAD commit."""
    repo = Path(repo_path)
    if not (repo / ".git").exists():
        raise GitAnalyzerError(f"Directory {repo_path} is not a valid git repository.")

    # %H: full hash, %an: author name, %ae: author email, %cI: commit date ISO, %s: subject
    log_output = run_git_cmd(["log", "-1", "--format=%H%x00%an%x00%ae%x00%cI%x00%s"], repo)
    parts = log_output.split("\x00")
    if len(parts) < 5:
        raise GitAnalyzerError("Failed to parse git log output.")

    return {
        "hash": parts[0],
        "author_name": parts[1],
        "author_email": parts[2],
        "commit_date": parts[3],
        "message": parts[4],
    }


def get_commit_stats(repo_path: Path | str, commit_ref: str = "HEAD") -> dict[str, Any]:
    """Get line addition, deletion, and file change statistics for a commit."""
    repo = Path(repo_path)
    # Get numstat: added \t deleted \t filename
    raw_numstat = run_git_cmd(["show", "--numstat", "--format=", commit_ref], repo)

    files_changed = 0
    lines_added = 0
    lines_deleted = 0
    file_details = []

    for line in raw_numstat.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            add_str, del_str, filename = parts[0], parts[1], parts[2]
            # Ignore virtualenv, git, and cache paths
            if any(p in filename for p in [".venv/", "venv/", "__pycache__/", "site-packages/"]):
                continue
            # Binary files show '-' for additions/deletions
            adds = int(add_str) if add_str.isdigit() else 0
            dels = int(del_str) if del_str.isdigit() else 0
            lines_added += adds
            lines_deleted += dels
            files_changed += 1
            file_details.append({
                "filename": filename,
                "additions": adds,
                "deletions": dels,
            })

    return {
        "commit_ref": commit_ref,
        "files_changed": files_changed,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "total_churn": lines_added + lines_deleted,
        "file_details": file_details,
    }


def get_recent_churn(repo_path: Path | str, window_days: int = 30) -> dict[str, Any]:
    """Calculate commit frequency and churn over the given time window."""
    repo = Path(repo_path)
    since_arg = f"--since={window_days}.days"

    # List commit hashes in window
    commit_list_out = run_git_cmd(["log", since_arg, "--format=%H"], repo)
    commits = [c for c in commit_list_out.splitlines() if c.strip()]

    # Shortlog for author counts
    author_out = run_git_cmd(["shortlog", "-s", "-n", since_arg], repo)
    authors = [a.strip() for a in author_out.splitlines() if a.strip()]

    # Churn stats in window
    shortstat_out = run_git_cmd(["log", since_arg, "--shortstat", "--oneline"], repo)
    total_added = 0
    total_deleted = 0
    for line in shortstat_out.splitlines():
        if "insertion" in line or "deletion" in line:
            parts = line.split(",")
            for part in parts:
                part = part.strip()
                if "insertion" in part:
                    total_added += int(part.split()[0])
                elif "deletion" in part:
                    total_deleted += int(part.split()[0])

    return {
        "window_days": window_days,
        "recent_commit_count": len(commits),
        "active_contributors_count": len(authors),
        "recent_lines_added": total_added,
        "recent_lines_deleted": total_deleted,
        "recent_total_churn": total_added + total_deleted,
    }


def clone_repository(
    repo_url: str,
    target_dir: Path | str,
    depth: int = 50,
    timeout: int = 120,
) -> Path:
    """Safely shallow clone a repository for analysis."""
    dest = Path(target_dir)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "git",
        "clone",
        "--depth",
        str(depth),
        "--no-single-branch",
        "--filter=blob:none",  # Partial clone to save bandwidth when possible
        repo_url,
        str(dest),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError:
        # Fallback to standard shallow clone if blob filter is not supported by remote
        cmd = ["git", "clone", "--depth", str(depth), repo_url, str(dest)]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            raise GitAnalyzerError(f"Failed to clone repository: {exc.stderr.strip()}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitAnalyzerError(f"Clone timed out after {timeout} seconds.") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitAnalyzerError(f"Clone timed out after {timeout} seconds.") from exc

    return dest
