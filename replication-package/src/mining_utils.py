"""Mining utilities for merge conflict extraction.

Core Git operations and conflict analysis functions from the main project,
refactored for the replication-package pipeline.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Any


class FunctionTimeoutError(Exception):
    """Raised when a function exceeds its time limit."""
    pass


def timeout_handler(signum, frame):
    raise FunctionTimeoutError("Function took too long to execute!")


@contextmanager
def time_limit(seconds):
    """Context manager for signal-based timeouts on Unix systems."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# --- Constants ---

LOC_BOF = "<BOF>\n"
LOC_EOF = "\n<EOF>"

DIFF3_CHUNK_REGEX = re.compile(
    r"<<<<<<< .*?\n(.*?)\n?\|\|\|\|\|\|\| .*?\n(.*?)\n?=======\n(.*?)\n?>>>>>>> .*?\n",
    re.DOTALL,
)

MERGE_TREE_CONFLICT_REGEX = re.compile(
    r"changed in both\n"
    r"\s*base\s+\d+\s+([a-f0-9]+)\s+(.+)\n"
    r"\s*our\s+\d+\s+([a-f0-9]+)\s+\2\n"
    r"\s*their\s+\d+\s+([a-f0-9]+)\s+\2",
    re.MULTILINE,
)

_PR_INTEGRATION_MERGE_RE = re.compile(r"merge pull request\s+#\d+", re.IGNORECASE)

LOCALIZE_TIMEOUT_S = 60


# --- Git Command Utilities ---

def run_git_command(
    cwd: Path, *args: str, check: bool = True, timeout: int = 30
) -> bytes:
    """Execute a git command in the specified directory.

    Includes safety measures: GIT_CEILING_DIRECTORIES prevents git from
    walking up the tree and accidentally operating on a parent repository.
    """
    command = ["git", "-C", str(cwd), *args]
    my_env = os.environ.copy()
    my_env["GIT_TERMINAL_PROMPT"] = "0"
    my_env["GIT_CEILING_DIRECTORIES"] = str(Path(cwd).resolve().parent)

    try:
        result = subprocess.run(
            command,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=my_env
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout: Git command {' '.join(command)} took longer than {timeout}s")
        raise
    except subprocess.CalledProcessError as e:
        if check:
            logging.error(f"Git command failed in {cwd}: {' '.join(command)}")
            logging.error(f"STDERR: {e.stderr.decode('utf-8', 'ignore')}")
            raise
        return b""


def _is_valid_bare_repo(repo_path: Path, env: dict) -> bool:
    """Check if repo_path contains a valid bare git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--is-bare-repository"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=env,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return result.stdout.strip() == b"true"


def clone_repo_bare(repo_url: str, repos_dir: Path) -> Optional[Path]:
    """Clone or update a bare repository."""
    repo_name = (
        repo_url.split("://")[-1].replace("/", "_").replace(".git", "") + ".git"
    )
    repo_path = repos_dir / repo_name

    my_env = os.environ.copy()
    my_env["GIT_TERMINAL_PROMPT"] = "0"
    my_env["GIT_CEILING_DIRECTORIES"] = str(repos_dir.resolve())

    try:
        if repo_path.exists() and not _is_valid_bare_repo(repo_path, my_env):
            logging.warning(
                f"Scratch directory {repo_path} exists but is not a valid "
                f"bare repository. Removing it before re-cloning."
            )
            shutil.rmtree(repo_path, ignore_errors=True)

        if repo_path.exists():
            logging.info(f"Fetching updates for {repo_url}...")
            run_git_command(
                repo_path, "fetch", "origin", "+refs/heads/*:refs/heads/*", "--prune",
                timeout=1800
            )
        else:
            logging.info(f"Cloning {repo_url} into {repo_path}...")
            subprocess.run(
                ["git", "clone", "--bare", "--filter=blob:none", repo_url, str(repo_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1800,
                env=my_env
            )
        return repo_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone/fetch {repo_url}: {e.stderr.decode('utf-8', 'ignore')}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred with {repo_url}: {e}")
        return None


def _parse_commit_object(blob: bytes) -> Dict[str, Any]:
    """Parse the output of 'git cat-file -p <sha>' for a commit object."""
    text = blob.decode("utf-8", "ignore")
    parts = text.split("\n\n", 1)
    metadata = parts[0]
    message = parts[1] if len(parts) > 1 else ""

    parents = []
    author = "Unknown"
    committer = "Unknown"
    for line in metadata.split("\n"):
        if line.startswith("parent "):
            parents.append(line.split(" ", 1)[1].strip())
        elif line.startswith("author "):
            author = line.split("author ", 1)[1].split("<")[0].strip()
        elif line.startswith("committer "):
            committer = line.split("committer ", 1)[1].split("<")[0].strip()

    return {
        "parents": parents,
        "author": author,
        "committer": committer,
        "message": message,
    }


def get_new_merge_commits(
    repo_path: Path,
    processed_merges: set,
    repo_url: str,
) -> Generator[Tuple[str, str, str], None, None]:
    """List all 2-parent merge commits in the repo that haven't been processed."""
    try:
        log_output = run_git_command(
            repo_path, "log", "--all", "--merges", "--format=%H %P"
        ).decode("utf-8", "ignore")

        for line in log_output.splitlines():
            parts = line.split()
            if len(parts) == 3:
                merge_hash, p1, p2 = parts
                if (repo_url, merge_hash) not in processed_merges:
                    yield merge_hash, p1, p2

    except Exception as e:
        logging.error(f"Failed to get merge commits for {repo_path}: {e}")


# --- Resolution Localization (LOCALIZERESREGION) ---

def _find_resolution_start(pre_context: str, resolved_content: str) -> int:
    """Finds the start index of the resolution (end of pre-context)."""
    for i in range(1, len(pre_context) + 1):
        suffix = pre_context[-i:]
        if resolved_content.count(suffix) == 1:
            return resolved_content.find(suffix) + len(suffix)
    return -1


def _find_resolution_end(post_context: str, resolved_content: str) -> int:
    """Finds the end index of the resolution (start of post-context)."""
    for i in range(1, len(post_context) + 1):
        prefix = post_context[:i]
        if resolved_content.count(prefix) == 1:
            return resolved_content.find(prefix)
    return -1


def find_resolution(
    pre_context: str,
    post_context: str,
    resolved_content: str,
) -> Tuple[Optional[str], str]:
    """Implements the LOCALIZERESREGION algorithm.

    Returns:
        A tuple of (resolution_text, status_string).
    """
    search_pre_context = LOC_BOF + pre_context
    search_post_context = post_context + LOC_EOF
    search_resolved_content = LOC_BOF + resolved_content + LOC_EOF

    start_index_in_search = _find_resolution_start(
        search_pre_context, search_resolved_content
    )
    end_index_in_search = _find_resolution_end(
        search_post_context, search_resolved_content
    )

    if (
        start_index_in_search != -1
        and end_index_in_search != -1
        and start_index_in_search <= end_index_in_search
    ):
        real_start = start_index_in_search - len(LOC_BOF)
        real_end = end_index_in_search - len(LOC_BOF)

        real_start = max(0, real_start)
        real_end = min(len(resolved_content), real_end)

        if real_start <= real_end:
            resolution = resolved_content[real_start:real_end]
            return resolution, "found"
        else:
            return None, "not_localizable (indices_inverted)"
    else:
        return None, "not_localizable (context_not_unique)"


# --- Conflict Analysis Core ---

def run_merge_file(
    p1_content: bytes,
    base_content: bytes,
    p2_content: bytes,
) -> Tuple[str, bool]:
    """Run 'git merge-file' to simulate a 3-way merge.

    Returns:
        A tuple of (merged_output_string, has_conflict_bool).
    """
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            p1_path = Path(tmpdir) / "p1.tmp"
            base_path = Path(tmpdir) / "base.tmp"
            p2_path = Path(tmpdir) / "p2.tmp"

            p1_path.write_bytes(p1_content)
            base_path.write_bytes(base_content)
            p2_path.write_bytes(p2_content)

            result = subprocess.run(
                [
                    "git",
                    "merge-file",
                    "-p",
                    "--diff3",
                    str(p1_path),
                    str(base_path),
                    str(p2_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
            merged_content = result.stdout.decode("utf-8", "ignore")
            has_conflict = result.returncode != 0
            return merged_content, has_conflict

    except Exception as e:
        logging.error(f"git merge-file failed: {e}")
        return "", False


def parse_diff3_chunks(conflict_content: str) -> List[Dict[str, str]]:
    """Parse a file containing diff3 markers into a list of chunks."""
    chunks = []
    parts = DIFF3_CHUNK_REGEX.split(conflict_content)

    if not parts:
        return []

    pre_context = parts[0]

    for i in range(1, len(parts), 4):
        try:
            v1 = parts[i]
            base = parts[i+1]
            v2 = parts[i+2]
            post_context = parts[i+3]

            chunks.append(
                {
                    "v1": v1.rstrip("\n"),
                    "base": base.rstrip("\n"),
                    "v2": v2.rstrip("\n"),
                    "pre_context": pre_context,
                    "post_context": post_context,
                }
            )
            pre_context = post_context
        except IndexError:
            continue

    return chunks


__all__ = [
    "FunctionTimeoutError",
    "time_limit",
    "LOC_BOF",
    "LOC_EOF",
    "DIFF3_CHUNK_REGEX",
    "MERGE_TREE_CONFLICT_REGEX",
    "_PR_INTEGRATION_MERGE_RE",
    "LOCALIZE_TIMEOUT_S",
    "run_git_command",
    "clone_repo_bare",
    "_parse_commit_object",
    "get_new_merge_commits",
    "find_resolution",
    "run_merge_file",
    "parse_diff3_chunks",
]
