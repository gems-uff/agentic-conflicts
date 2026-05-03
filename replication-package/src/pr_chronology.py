"""Extract PR commit chronology using Git refs without GitHub API rate limits.

For each PR, find all commits between the fork point (merge-base with main)
and the PR tip, in chronological order. Output: (pr_id, sha, author_date, commit_index).
"""

import json
import logging
import os
import shutil
import stat
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd

from .mining_utils import clone_repo_bare, run_git_command


def _normalize_repo_url(repo_url: str) -> str:
    """Normalize repo URL to https://github.com/<owner>/<repo>.git format."""
    if not isinstance(repo_url, str):
        return repo_url
    url = repo_url.strip()
    if url.startswith("https://api.github.com/repos/"):
        url = url.replace("https://api.github.com/repos/", "https://github.com/")
    if not url.endswith(".git"):
        url += ".git"
    return url


def extract_pr_commits(
    repo_info: Tuple[str, pd.DataFrame],
    scratch_dir: Path,
) -> Tuple[str, List[Dict], List[Dict]]:
    """Extract all commits for each PR in a repository.

    For each PR in the repo:
    1. Fetch the PR ref from GitHub (refs/pull/*/head)
    2. Find fork point (merge-base with default branch)
    3. Extract all commits between fork and PR tip

    Args:
        repo_info: Tuple of (repo_full_name, repo_dataframe)
        scratch_dir: Directory for temporary bare clones

    Returns:
        Tuple of (repo_full_name, chronology_records, errors)
    """
    repo_full_name, repo_df = repo_info
    repo_url = _normalize_repo_url(repo_df.iloc[0]['repo_url'])

    repo_path = clone_repo_bare(repo_url, scratch_dir)
    if not repo_path:
        logging.error(f"Failed to clone/update repository {repo_full_name}")
        return repo_full_name, [], [{
            "repo_full_name": repo_full_name,
            "error": "clone_failed"
        }]

    chronology_data = []
    errors = []

    try:
        # Fetch all PR heads efficiently using wildcards
        fetch_bytes = run_git_command(
            repo_path,
            "fetch",
            "origin",
            "+refs/pull/*/head:refs/pull/*/head",
            check=False
        )
        if fetch_bytes is None:
            errors.append({
                "repo_full_name": repo_full_name,
                "error": "fetch_failed"
            })
            return repo_full_name, chronology_data, errors

        # Process each PR once. When the input universe already contains
        # commit-level rows (for example from an existing pr_commits.parquet),
        # repo_df has multiple rows per PR; re-fetching the same PR chronology
        # for every commit would be correct after downstream deduplication but
        # painfully slow on a full collection.
        pr_rows = repo_df.drop_duplicates(subset=["pr_id"])
        for _, row in pr_rows.iterrows():
            pr_id = row['pr_id']
            pr_number = row['number']

            if pd.isna(pr_number):
                continue

            pr_number = int(pr_number)
            head_ref = f"refs/pull/{pr_number}/head"

            # Verify PR ref exists locally
            check_ref = run_git_command(
                repo_path,
                "show-ref",
                "--verify",
                head_ref,
                check=False
            )
            if not check_ref:
                errors.append({
                    "repo_full_name": repo_full_name,
                    "pr_id": pr_id,
                    "pr_number": pr_number,
                    "error": "pr_ref_not_found"
                })
                continue

            # Find fork point (merge-base with default branch HEAD)
            base_bytes = run_git_command(
                repo_path,
                "merge-base",
                head_ref,
                "HEAD",
                check=False
            )
            if not base_bytes:
                errors.append({
                    "repo_full_name": repo_full_name,
                    "pr_id": pr_id,
                    "pr_number": pr_number,
                    "error": "no_merge_base"
                })
                continue

            base_sha = base_bytes.decode("utf-8", "ignore").strip()

            # Extract all commits between fork point and PR tip (chronological order)
            log_bytes = run_git_command(
                repo_path,
                "log",
                f"{base_sha}..{head_ref}",
                "--reverse",
                "--format=%H|%aI",
                check=False
            )
            if not log_bytes:
                # PR has 0 own commits (identical to base) - nothing to emit
                continue

            log_lines = [line for line in log_bytes.decode("utf-8", "ignore").splitlines() if line.strip()]

            for commit_index, line in enumerate(log_lines):
                parts = line.split("|")
                sha = parts[0]
                author_date = parts[1] if len(parts) > 1 else None

                chronology_data.append({
                    "pr_id": pr_id,
                    "repo_full_name": repo_full_name,
                    "pr_number": pr_number,
                    "sha": sha,
                    "author_date": author_date,
                    "commit_index": commit_index,
                    "commit_count": len(log_lines),
                })

    except Exception as e:
        errors.append({
            "repo_full_name": repo_full_name,
            "error": str(e)
        })
    finally:
        # Cleanup with read-only file handling
        def _remove_readonly(func, path, excinfo):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

        try:
            shutil.rmtree(repo_path, onerror=_remove_readonly)
        except Exception:
            pass

    return repo_full_name, chronology_data, errors


def append_jsonl(filename: str, records: List[Dict], data_dir: Path) -> None:
    """Append records to a JSONL file."""
    if not records:
        return
    with open(data_dir / filename, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def aggregate_pr_commits_to_parquet(
    data_dir: Path,
    jsonl_file: str = "pr_commits.jsonl",
    parquet_file: str = "pr_commits.parquet"
) -> pd.DataFrame:
    """Convert JSONL PR commits to Parquet with deduplication.

    Args:
        data_dir: Data directory containing JSONL file
        jsonl_file: Input JSONL filename (default: pr_commits.jsonl)
        parquet_file: Output Parquet filename (default: pr_commits.parquet)

    Returns:
        DataFrame with aggregated PR commits
    """
    jsonl_path = data_dir / jsonl_file
    if not jsonl_path.exists():
        logging.warning(f"{jsonl_file} not found, returning empty DataFrame")
        return pd.DataFrame()

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed line in {jsonl_file}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Deduplicate on (pr_id, sha)
    dedup_keys = ["pr_id", "sha"]
    if all(k in df.columns for k in dedup_keys):
        before = len(df)
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
        after = len(df)
        if before != after:
            logging.info(f"  Dedup {jsonl_file}: {before:,} -> {after:,} rows")

    df.to_parquet(data_dir / parquet_file)
    logging.info(f"Aggregated {jsonl_file} -> {parquet_file} ({len(df)} rows)")

    return df


__all__ = [
    "extract_pr_commits",
    "append_jsonl",
    "aggregate_pr_commits_to_parquet",
]
