"""Tests for src/github_connector.py."""

import pytest
from src.github_connector import parse_github_url


def test_valid_repository_url():
    assert parse_github_url("https://github.com/psf/requests") == ("psf", "requests")


def test_trailing_slash():
    assert parse_github_url("https://github.com/psf/requests/") == ("psf", "requests")


def test_dot_git_suffix():
    assert parse_github_url("https://github.com/psf/requests.git") == ("psf", "requests")


def test_rejects_other_hosts():
    with pytest.raises(ValueError, match="canonical GitHub repository URL"):
        parse_github_url("https://gitlab.com/user/repo")


def test_rejects_extra_path():
    with pytest.raises(ValueError, match="canonical GitHub repository URL"):
        parse_github_url("https://github.com/user/repo/tree/main")


def test_rejects_empty_repository():
    with pytest.raises(ValueError, match="canonical GitHub repository URL"):
        parse_github_url("https://github.com/user/")


def test_rejects_empty_string():
    with pytest.raises(ValueError, match="Repository URL cannot be empty"):
        parse_github_url("")


def test_rejects_non_string():
    with pytest.raises(ValueError, match="URL must be a string"):
        parse_github_url(123)  # type: ignore
