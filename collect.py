#!/usr/bin/env python3

"""
Conflict Miner Script

This script processes a list of Git repositories to find new merge commits
that are not present in an existing dataset. For each new merge, it
re-simulates the merge, identifies 3-way conflicts, and extracts the
conflicting chunks (v1, v2, base). It then attempts to locate the
corresponding resolved chunk from the final merge commit using the
"minimal unique prefix/suffix" (LOCALIZERESREGION) strategy[1].

[1] Dinella et al. (2023). DeepMerge: Learning to Merge Programs, IEEE Transactions on Software Engineering

All data is saved incrementally to a JSONL output file.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Generator,
    Any,
)

import signal

class FunctionTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise FunctionTimeoutError("Function took too long to answer!")


from contextlib import contextmanager

@contextmanager
def time_limit(seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# --- Configuration ---

# Path to the input JSON dataset (configurable)
INPUT_FILE = Path("../sbcr_genetic/data/dataset2_Java_metadata.json")

# Directory to store bare clones of repositories
REPOS_DIR = Path("bare_repos")

# Output file for new conflicts (JSON Lines format)
OUTPUT_FILE = Path("new_conflicts.jsonl")

# --- Constants ---

# Special tokens for resolution localization, as per the paper
LOC_BOF = "<BOF>\n"
LOC_EOF = "\n<EOF>"

# Regex to parse diff3 conflict blocks
# Groups: (1) v1, (2) base, (3) v2
DIFF3_CHUNK_REGEX = re.compile(
    r"<<<<<<< .*?\n(.*?)\n?\|\|\|\|\|\|\| .*?\n(.*?)\n?=======\n(.*?)\n?>>>>>>> .*?\n",
    re.DOTALL,
)

# Regex to parse 'git merge-tree' output for conflicts
# Groups: (1) base_blob, (2) path, (3) p1_blob, (4) p2_blob
MERGE_TREE_CONFLICT_REGEX = re.compile(
    r"changed in both\n"
    r"\s*base\s+\d+\s+([a-f0-9]+)\s+(.+)\n"
    r"\s*our\s+\d+\s+([a-f0-9]+)\s+\2\n"
    r"\s*their\s+\d+\s+([a-f0-9]+)\s+\2",
    re.MULTILINE,
)

# --- Logging Setup ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("conflict_miner.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


# --- Git Command Utility ---

def run_git_command(cwd: Path, *args: str, check: bool = True, timeout: int = 30) -> bytes:
    command = ["git", "-C", str(cwd), *args]
    my_env = os.environ.copy()
    my_env["GIT_TERMINAL_PROMPT"] = "0"
    # Prevent git's directory discovery from walking up past `cwd` if the
    # scratch directory is empty or corrupt. Without this, `git -C <scratch>`
    # would find the caller's own project .git and operate on it, which has
    # actually happened in the wild (fetch refusing to update the currently
    # checked-out branch of /home/.../agentic-conflicts). The ceiling is set
    # to cwd.parent: git will stay inside cwd, but any attempt to chdir upward
    # is blocked.
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
        raise  # Re-raise to let the caller handle it or skip the commit
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed in {cwd}: {' '.join(command)}")
        logging.error(f"STDERR: {e.stderr.decode('utf-8', 'ignore')}")
        raise


# --- State Management ---

def get_unique_repos(input_file: Path) -> Set[str]:
    """
    Reads the column-oriented input JSON and extracts a unique set of 'repo' URLs
    from the 'repo' column.
    """
    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)

    repos = set()
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
        
        # Check if 'repo' key exists and is a list
        if "repo" in data and isinstance(data["repo"], list):
            # Add all items from the 'repo' list to the set
            repos = set(data["repo"])
        else:
            logging.error(f"Input file {input_file} does not contain a 'repo' key with a list of URLs.")
            sys.exit(1)

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse input JSON {input_file}: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading {input_file}: {e}")
        sys.exit(1)

    logging.info(f"Found {len(repos)} unique repositories in {input_file}")
    return repos


def load_processed_merges(
    input_file: Path,
    output_file: Path,
) -> Set[Tuple[str, str]]:
    """
    Loads all (repo_url, commit_hash) tuples from the column-oriented input dataset
    and the row-oriented (JSONL) output file to avoid reprocessing.
    """
    processed = set()

    # Load from the old dataset (column-oriented)
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
        
        if "repo" in data and "commitHash" in data:
            repo_list = data["repo"]
            hash_list = data["commitHash"]

            if len(repo_list) != len(hash_list):
                logging.warning(f"Mismatched lengths for 'repo' ({len(repo_list)}) and 'commitHash' ({len(hash_list)}) columns in {input_file}.")

            # Zip the two columns together to create (repo, hash) pairs
            for repo_url, commit_hash in zip(repo_list, hash_list):
                processed.add((repo_url, commit_hash))
        else:
            logging.warning(f"Could not load old dataset: {input_file} is missing 'repo' or 'commitHash' columns.")
            
    except Exception as e:
        logging.warning(f"Could not load or parse old dataset {input_file}: {e}")

    # Load from the incremental output file (row-oriented JSONL)
    # This part remains the same as it correctly reads the JSONL file we create
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        data_line = json.loads(line)
                        if "repo" in data_line and "commit_hash" in data_line:
                            processed.add((data_line["repo"], data_line["commit_hash"]))
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping malformed line in {output_file}")
        except Exception as e:
            logging.warning(f"Could not load output file {output_file}: {e}")

    logging.info(f"Loaded {len(processed)} already processed merges.")
    return processed


# --- Repository Management ---

def _is_valid_bare_repo(repo_path: Path, env: dict) -> bool:
    """Return True iff `repo_path` actually contains a bare git repository.

    This guards against the case where a previous clone attempt was interrupted
    and left an empty or partial directory behind. Without this check, the
    subsequent fetch would trigger git's upward directory discovery and
    silently operate on an unintended repository up the tree.
    """
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
    """
    Clones or updates a bare repository.
    """
    repo_name = (
        repo_url.split("://")[-1].replace("/", "_").replace(".git", "") + ".git"
    )
    repo_path = repos_dir / repo_name

    # Get current environment and disable interactive prompts.
    my_env = os.environ.copy()
    my_env["GIT_TERMINAL_PROMPT"] = "0"
    # Lock git's repo discovery to the scratch directory. If the scratch slot
    # is empty or corrupt, git must NOT walk up the tree and silently attach
    # to the caller's own project repo. See run_git_command for the same
    # guard on fetch-time commands.
    my_env["GIT_CEILING_DIRECTORIES"] = str(repos_dir.resolve())

    try:
        if repo_path.exists() and not _is_valid_bare_repo(repo_path, my_env):
            logging.warning(
                f"Scratch directory {repo_path} exists but is not a valid "
                f"bare repository (likely a broken prior clone). Removing "
                f"it before re-cloning."
            )
            shutil.rmtree(repo_path, ignore_errors=True)

        if repo_path.exists():
            logging.info(f"Fetching updates for {repo_url}...")
            run_git_command(repo_path, "fetch", "origin", "+refs/heads/*:refs/heads/*", "--prune", timeout=1800)
        else:
            logging.info(f"Cloning {repo_url} into {repo_path}...")
            # We use git clone directly (not -C) as -C requires the dir to exist
            subprocess.run(
                ["git", "clone", "--bare", repo_url, str(repo_path)],
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


def get_new_merge_commits(
    repo_path: Path,
    processed_merges: Set[Tuple[str, str]],
    repo_url: str,
) -> Generator[Tuple[str, str, str], None, None]:
    """
    Lists all 2-parent merge commits in the repo that haven't been processed.
    """
    try:
        # Get all commits with their hash and parent hashes
        log_output = run_git_command(
            repo_path, "log", "--all", "--merges", "--format=%H %P"
        ).decode("utf-8", "ignore")

        for line in log_output.splitlines():
            parts = line.split()
            if len(parts) == 3:  # Exactly 1 hash + 2 parent hashes
                merge_hash, p1, p2 = parts
                if (repo_url, merge_hash) not in processed_merges:
                    yield merge_hash, p1, p2

    except Exception as e:
        logging.error(f"Failed to get merge commits for {repo_path}: {e}")


# --- Resolution Localization (from Paper) ---

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
    """
    Implements the LOCALIZERESREGION algorithm.

    Args:
        pre_context: All text in the conflicting file before the chunk.
        post_context: All text in the conflicting file after the chunk.
        resolved_content: The full content of the resolved file.

    Returns:
        A tuple of (resolution_text, status_string).
    """
    logging.info("Looking for resolution...")
    # Add BOF/EOF tokens as per the paper's strategy
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
        # Adjust indices to account for the added <BOF> token
        real_start = start_index_in_search - len(LOC_BOF)
        real_end = end_index_in_search - len(LOC_BOF)

        # Clamp indices to the bounds of the *original* content
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
    """
    Runs `git merge-file` on the content of three blobs.

    Args:
        p1_content: Bytes content of parent 1's file.
        base_content: Bytes content of base's file.
        p2_content: Bytes content of parent 2's file.

    Returns:
        A tuple of (merged_output_string, has_conflict_bool).
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            p1_path = Path(tmpdir) / "p1.tmp"
            base_path = Path(tmpdir) / "base.tmp"
            p2_path = Path(tmpdir) / "p2.tmp"

            p1_path.write_bytes(p1_content)
            base_path.write_bytes(base_content)
            p2_path.write_bytes(p2_content)
            logging.info(f"Merging: {p1_path}, {p2_path}, base:{base_path}")
            # -p: Send merged result to stdout
            # --diff3: Use diff3 style for conflicts
            # Exit code is 0 on success, >0 on conflict
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
            logging.info(f"Finished merging")
            merged_content = result.stdout.decode("utf-8", "ignore")
            has_conflict = result.returncode != 0
            return merged_content, has_conflict

    except Exception as e:
        logging.error(f"git merge-file failed: {e}")
        return "", False


def parse_diff3_chunks(
    conflict_content: str,
) -> List[Dict[str, str]]:
    """
    Parses a file containing diff3 markers into a list of chunks,
    including the context *between* chunks.
    """
    logging.info("Parsing conflicting chunks")
    chunks = []
    # Split the file by the start of a conflict marker
    # The first item is the pre-context for the first chunk
    parts = DIFF3_CHUNK_REGEX.split(conflict_content)
    
    if not parts:
        return []

    pre_context = parts[0]
    
    # The regex split returns a list like:
    # [pre_context_0, v1_1, base_1, v2_1, post_context_1_and_pre_context_2, ...]
    # We iterate in groups of 4, starting from index 1.
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
            # The post-context of this chunk is the pre-context of the next
            pre_context = post_context
        except IndexError:
            # Malformed split, ignore this chunk
            continue
            
    return chunks


def analyze_merge_conflicts(
    repo_path: Path,
    repo_url: str,
    merge_hash: str,
    p1: str,
    p2: str,
) -> List[Dict[str, Any]]:
    """
    The core analysis function for a single merge commit.
    """
    results = []
    try:
        # 1. Find merge base
        base = (
            run_git_command(repo_path, "merge-base", p1, p2)
            .decode("utf-8", "ignore")
            .strip()
        )
        if not base:
            logging.warning(f"No merge base found for {merge_hash}. Skipping.")
            return []

        # 2. Run merge-tree to find conflicting file blobs
        tree_output = run_git_command(
            repo_path, "merge-tree", base, p1, p2
        ).decode("utf-8", "ignore")
        
        conflicting_files = MERGE_TREE_CONFLICT_REGEX.finditer(tree_output)
        
        for match in conflicting_files:
            base_blob, path, p1_blob, p2_blob = match.groups()
            
            logging.info(f"  Found conflict in: {path}")
            if '.java' in path:
                try:
                    # 3. Get content of all 3 file versions and the resolved file
                    base_content = run_git_command(repo_path, "show", base_blob)
                    p1_content = run_git_command(repo_path, "show", p1_blob)
                    p2_content = run_git_command(repo_path, "show", p2_blob)
                    
                    # Use --textconv if available, but simple 'show' is safer
                    resolved_content_bytes = run_git_command(
                        repo_path, "show", f"{merge_hash}:{path}"
                    )
                    logging.info(f"Merge resolution file size: {len(resolved_content_bytes)/1000} KB")
                    resolved_content = resolved_content_bytes.decode('utf-8', 'ignore')
    
                    # 4. Re-run the merge with `git merge-file` to get diff3 output
                    conflict_content, has_conflict = run_merge_file(
                        p1_content, base_content, p2_content
                    )
    
                    if not has_conflict:
                        continue

                    try:
                        with time_limit(60):  
                            chunks = parse_diff3_chunks(conflict_content)
                            if not chunks:
                                logging.warning(f"  Failed to parse diff3 chunks for {path}")
                                continue
            
                            # 6. For each chunk, find its resolution
                            for chunk in chunks:
                                resolution, status = find_resolution(
                                    chunk["pre_context"],
                                    chunk["post_context"],
                                    resolved_content,
                                )
                                
                                result_data = {
                                    "repo": repo_url,
                                    "commit_hash": merge_hash,
                                    "file_path": path,
                                    "v1": chunk["v1"],
                                    "v2": chunk["v2"],
                                    "base": chunk["base"],
                                    "resolution": resolution,
                                    "resolution_status": status,
                                }
                                results.append(result_data)
                    except FunctionTimeoutError:
                        logging.error(f"Timeout processing internal file: {path}")
                        continue
                except Exception as e:
                    logging.error(
                        f"  Failed to analyze file {path} in {merge_hash}: {e}"
                    )
                    continue

    except Exception as e:
        logging.error(f"Failed to analyze merge {merge_hash}: {e}")
    
    return results


# --- Main Execution ---

def save_results(output_file: Path, results: List[Dict[str, Any]]):
    """
    Appends a list of results to the JSONL output file.
    """
    try:
        with open(output_file, "a", encoding="utf-8") as f:
            for item in results:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
    except Exception as e:
        logging.error(f"Failed to write results to {output_file}: {e}")


def main():
    """
    Main script execution.
    """
    logging.info("--- Conflict Miner Started ---")
    REPOS_DIR.mkdir(exist_ok=True)

    # 1. Load state
    processed_merges = load_processed_merges(INPUT_FILE, OUTPUT_FILE)
    repos_to_process = get_unique_repos(INPUT_FILE)

    if not repos_to_process:
        logging.info("No repositories found in input file. Exiting.")
        return

    # 2. Process each repository
    try:
        for repo_url in repos_to_process:
            logging.info(f"--- Processing repository: {repo_url} ---")
            repo_path = clone_repo_bare(repo_url, REPOS_DIR)
            if not repo_path:
                continue

            # 3. Find new merge commits
            new_merges = get_new_merge_commits(
                repo_path, processed_merges, repo_url
            )
            
            merges_to_process = list(new_merges)
            logging.info(f"Found {len(merges_to_process)} new 2-parent merges.")

            # 4. Analyze each new merge
            for i, (merge_hash, p1, p2) in enumerate(merges_to_process):
                logging.info(
                    f"Analyzing merge {i+1}/{len(merges_to_process)}: {merge_hash}"
                )
                
                # Wrap analysis in try/except to be robust to single-merge failures
                try:
                    conflict_results = analyze_merge_conflicts(
                        repo_path, repo_url, merge_hash, p1, p2
                    )

                    if conflict_results:
                        logging.info(
                            f"  Found {len(conflict_results)} conflict chunks."
                        )
                        save_results(OUTPUT_FILE, conflict_results)
                    
                    # Mark as processed even if no conflicts were found
                    # to avoid re-checking it next time.
                    processed_merges.add((repo_url, merge_hash))
                    
                except Exception as e:
                    logging.error(
                        f"Critical error analyzing {merge_hash}. Skipping. Error: {e}"
                    )

    except KeyboardInterrupt:
        logging.info("\n--- Script interrupted by user. Shutting down. ---")
        logging.info("Results found so far have been saved.")
    except Exception as e:
        logging.error(f"--- An unhandled exception occurred: {e} ---")
    finally:
        logging.info("--- Conflict Miner Finished ---")


if __name__ == "__main__":
    main()