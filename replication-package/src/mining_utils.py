"""Utilities for extracting and analyzing merge conflicts from Git repositories.

This module provides functions to:
- Clone repositories
- Find merge commits
- Extract conflict chunks
- Classify resolution strategies
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json


def find_merge_commits(repo_path: str, agent_name: str) -> List[str]:
    """
    Find internal merge commits (2-parent merges) authored by an agent.

    Args:
        repo_path: Path to cloned repository
        agent_name: Git committer name pattern (e.g., 'claude[bot]')

    Returns:
        List of merge commit SHAs
    """
    try:
        result = subprocess.run(
            [
                'git',
                '-C', repo_path,
                'log',
                '--all',
                '--merges',
                '--pretty=format:%H',
                f'--committer={agent_name}'
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []


def extract_conflict_chunks(
    repo_path: str,
    merge_sha: str,
    diff3: bool = True
) -> List[Dict]:
    """
    Extract conflict chunks from a merge commit using git merge-file.

    Args:
        repo_path: Path to cloned repository
        merge_sha: SHA of merge commit
        diff3: Use diff3 format (includes base version)

    Returns:
        List of conflict chunk dicts with keys:
        - 'base': original content
        - 'version1': our version
        - 'version2': their version
        - 'resolution': resolved content
    """
    chunks = []

    try:
        # Get the merge commit tree
        result = subprocess.run(
            ['git', '-C', repo_path, 'show', f'{merge_sha}^1:{merge_sha}'],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Additional git operations to extract chunks would go here
        # This is a simplified placeholder

    except Exception:
        pass

    return chunks


def clone_bare_repo(repo_url: str, dest_dir: str, timeout: int = 300) -> Optional[str]:
    """
    Clone a repository as bare (to save space).

    Args:
        repo_url: GitHub URL (https://github.com/owner/repo)
        dest_dir: Destination directory
        timeout: Clone timeout in seconds

    Returns:
        Path to cloned repo, or None on failure
    """
    try:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(dest_dir, f'{repo_name}.git')

        subprocess.run(
            ['git', 'clone', '--bare', repo_url, repo_path],
            timeout=timeout,
            capture_output=True
        )

        return repo_path if os.path.exists(repo_path) else None

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


__all__ = [
    'find_merge_commits',
    'extract_conflict_chunks',
    'clone_bare_repo',
]
