"""GitHub Connector for Software Release Risk Predictor.

Validates GitHub repository URLs and fetches metadata, commits, and issue evidence.
"""

from __future__ import annotations

import os
import re
from typing import Any
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_API_BASE = "https://api.github.com"


class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""


class GitHubNotFoundError(GitHubAPIError):
    """Raised when repository or resource is not found."""


class GitHubRateLimitError(GitHubAPIError):
    """Raised when GitHub API rate limit is exceeded."""


def parse_github_url(url: str) -> tuple[str, str]:
    """Extract owner and repository name from a canonical GitHub URL.

    Args:
        url: URL string, e.g. "https://github.com/psf/requests"

    Returns:
        tuple[str, str]: (owner, repo)

    Raises:
        ValueError: If URL format is invalid, empty, or points to unsupported hosts.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")

    cleaned_url = url.strip()
    if not cleaned_url:
        raise ValueError("Repository URL cannot be empty.")

    pattern = (
        r"^https://github\.com/"
        r"([A-Za-z0-9_.-]+)/"
        r"([A-Za-z0-9_.-]+)/?$"
    )
    match = re.fullmatch(pattern, cleaned_url)
    if not match:
        raise ValueError(
            "Enter a canonical GitHub repository URL (e.g. https://github.com/owner/repo)."
        )

    owner, repo = match.groups()
    if repo.endswith(".git"):
        repo = repo[:-4]

    if not owner or not repo:
        raise ValueError("Invalid repository URL: owner and repo must be non-empty.")

    return owner, repo


def get_headers(token: str | None = None) -> dict[str, str]:
    """Prepare HTTP headers with optional authentication."""
    auth_token = token or os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Software-Release-Risk-Predictor",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


def get_repository_info(
    owner: str,
    repo: str,
    token: str | None = None,
    timeout: int = 15,
) -> dict[str, Any]:
    """Fetch repository metadata from GitHub REST API.

    Args:
        owner: GitHub account / organization
        repo: Repository name
        token: Optional GitHub personal access token
        timeout: Request timeout in seconds

    Returns:
        dict: Repository metadata including default branch, open issues, language, etc.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
    headers = get_headers(token)

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise GitHubAPIError(f"Network error while connecting to GitHub: {exc}") from exc

    if response.status_code == 404:
        raise GitHubNotFoundError(
            f"Repository '{owner}/{repo}' not found or is private."
        )
    if response.status_code == 403:
        if "rate limit" in response.text.lower():
            raise GitHubRateLimitError(
                "GitHub API rate limit exceeded. Set GITHUB_TOKEN in .env for 5000 requests/hr."
            )
        raise GitHubAPIError(f"Access forbidden (403): {response.text}")
    if response.status_code != 200:
        raise GitHubAPIError(
            f"GitHub API error {response.status_code}: {response.text}"
        )

    data = response.json()
    return {
        "name": data.get("name"),
        "full_name": data.get("full_name"),
        "owner": owner,
        "repo": repo,
        "description": data.get("description") or "No description provided",
        "default_branch": data.get("default_branch", "main"),
        "language": data.get("language") or "Unknown",
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "open_issues_count": data.get("open_issues_count", 0),
        "pushed_at": data.get("pushed_at"),
        "created_at": data.get("created_at"),
        "html_url": data.get("html_url"),
        "size_kb": data.get("size", 0),
        "is_fork": data.get("fork", False),
    }


def get_commit_history(
    owner: str,
    repo: str,
    count: int = 30,
    branch: str | None = None,
    token: str | None = None,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    """Fetch recent commit history for the repository."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
    params: dict[str, Any] = {"per_page": min(count, 100)}
    if branch:
        params["sha"] = branch

    headers = get_headers(token)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise GitHubAPIError(f"Network error fetching commits: {exc}") from exc

    if response.status_code == 404:
        raise GitHubNotFoundError(f"Repository or branch not found: '{owner}/{repo}'.")
    if response.status_code == 403 and "rate limit" in response.text.lower():
        raise GitHubRateLimitError("GitHub API rate limit exceeded fetching commits.")
    if response.status_code != 200:
        raise GitHubAPIError(f"Failed to fetch commits ({response.status_code}): {response.text}")

    commits = response.json()
    parsed_commits = []
    for c in commits:
        commit_data = c.get("commit", {})
        parsed_commits.append({
            "sha": c.get("sha", ""),
            "author": commit_data.get("author", {}).get("name", "Unknown"),
            "date": commit_data.get("author", {}).get("date", ""),
            "message": commit_data.get("message", ""),
            "url": c.get("html_url", ""),
        })
    return parsed_commits


def get_commit_details(
    owner: str,
    repo: str,
    commit_sha: str,
    token: str | None = None,
    timeout: int = 15,
) -> dict[str, Any]:
    """Fetch detailed stats for a single commit (lines added, deleted, files changed)."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{commit_sha}"
    headers = get_headers(token)

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise GitHubAPIError(f"Network error fetching commit details: {exc}") from exc

    if response.status_code != 200:
        raise GitHubAPIError(f"Failed to fetch commit {commit_sha}: {response.status_code}")

    data = response.json()
    stats = data.get("stats", {})
    files = [
        {
            "filename": f.get("filename"),
            "status": f.get("status"),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
        }
        for f in data.get("files", [])
    ]
    return {
        "sha": data.get("sha"),
        "author": data.get("commit", {}).get("author", {}).get("name"),
        "date": data.get("commit", {}).get("author", {}).get("date"),
        "message": data.get("commit", {}).get("message"),
        "total_changes": stats.get("total", 0),
        "additions": stats.get("additions", 0),
        "deletions": stats.get("deletions", 0),
        "files_changed_count": len(files),
        "files": files,
    }
