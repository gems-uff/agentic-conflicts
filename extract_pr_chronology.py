#!/usr/bin/env python3
"""Pipeline for extracting the full commit list of Pull Requests.
Utilizes Git's refs/pull/*/head refs to extract every commit of each PR
in a repository without hitting GitHub API rate limits.

Output: one row per (pr_id, sha) with commit_index (0-based chronological
order) and author_date.  This can later replace/extend pr_commits.parquet.
"""

import argparse
import json
import logging
import multiprocessing
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from collect import clone_repo_bare, run_git_command

AIDEV_DIR = Path("AIDev")
DATA_DIR = Path("data/pr_chronology")
SCRATCH_DIR = DATA_DIR / "scratch"
LOGS_DIR = DATA_DIR / "logs"

for d in [DATA_DIR, SCRATCH_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("extract_pr_chronology")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"pipeline_{timestamp}.log"
fh = logging.FileHandler(LOGS_DIR / log_filename, mode='w', encoding='utf-8')
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)
logger.propagate = False

def _normalize_repo_url(repo_url: str) -> str:
    if not isinstance(repo_url, str):
        return repo_url
    url = repo_url.strip()
    if url.startswith("https://api.github.com/repos/"):
        url = url.replace("https://api.github.com/repos/", "https://github.com/")
    if not url.endswith(".git"):
        url += ".git"
    return url

def build_universe(is_pilot: bool, pilot_count: int, use_full_aidev: bool = False) -> pd.DataFrame:
    logger.info("Stage 0: Building PR universe...")
    
    if use_full_aidev:
        logger.info("Using full AIDev dataset...")
        pr_df = pd.read_parquet(AIDEV_DIR / "all_pull_request.parquet")
        repo_df = pd.read_parquet(AIDEV_DIR / "all_repository.parquet")
    else:
        logger.info("Using AIDev-pop dataset...")
        pr_df = pd.read_parquet(AIDEV_DIR / "pull_request.parquet")
        repo_df = pd.read_parquet(AIDEV_DIR / "repository.parquet")

    # We need pr id, number, and repo url
    df = pr_df[['id', 'number', 'repo_id']].merge(
        repo_df[['id', 'full_name', 'url']], 
        left_on='repo_id', right_on='id'
    )
    
    # Rename columns for clarity, dropping the redundant right_on key 'id_y'
    df = df.drop(columns=['id_y'] if 'id_y' in df.columns else ['id'])
    df = df.rename(columns={'id_x': 'pr_id', 'url': 'repo_url', 'id': 'pr_id'})
    
    if is_pilot:
        unique_repos = df['full_name'].dropna().unique()[:pilot_count]
        df = df[df['full_name'].isin(unique_repos)]
        logger.info(f"Pilot mode: restricted to {len(unique_repos)} repos.")

    out_path = DATA_DIR / "chronology_universe.parquet"
    df.to_parquet(out_path)
    logger.info(f"Stage 0 complete. Universe has {len(df)} PRs across {df['full_name'].nunique()} repositories.")
    return df

def process_repository(repo_info: tuple):
    repo_full_name, repo_df = repo_info
    
    repo_url = _normalize_repo_url(repo_df.iloc[0]['repo_url'])
    
    repo_path = clone_repo_bare(repo_url, SCRATCH_DIR)
    if not repo_path:
        logger.error(f"Failed to clone/update repository {repo_full_name}.")
        return repo_full_name, [], [{"repo_full_name": repo_full_name, "error": "clone_failed"}]

    chronology_data = []
    errors = []
    
    try:
        # Fetch all PR heads efficiently directly from GitHub using the wildcards.
        fetch_bytes = run_git_command(repo_path, "fetch", "origin", "+refs/pull/*/head:refs/pull/*/head", check=False)
        if fetch_bytes is None:
             errors.append({"repo_full_name": repo_full_name, "error": "fetch_failed"})
             return repo_full_name, chronology_data, errors
             
        for _, row in repo_df.iterrows():
            pr_id = row['pr_id']
            pr_number = row['number']
            
            if pd.isna(pr_number):
                continue
            
            pr_number = int(pr_number)
            
            # 1. Ensure the PR ref exists locally
            head_ref = f"refs/pull/{pr_number}/head"
            check_ref = run_git_command(repo_path, "show-ref", "--verify", head_ref, check=False)
            if not check_ref:
                errors.append({
                    "repo_full_name": repo_full_name, 
                    "pr_id": pr_id, 
                    "pr_number": pr_number, 
                    "error": "pr_ref_not_found"
                })
                continue
                
            # 2. Find the fork point from the main branch (HEAD in the bare clone tracks the default branch)
            base_bytes = run_git_command(repo_path, "merge-base", head_ref, "HEAD", check=False)
            if not base_bytes:
                errors.append({
                    "repo_full_name": repo_full_name, 
                    "pr_id": pr_id, 
                    "pr_number": pr_number, 
                    "error": "no_merge_base"
                })
                continue
            base_sha = base_bytes.decode("utf-8", "ignore").strip()
            
            # 3. Log all commits between the fork point and the PR tip in chronological order
            log_bytes = run_git_command(repo_path, "log", f"{base_sha}..{head_ref}", "--reverse", "--format=%H|%aI", check=False)
            if not log_bytes:
                # PR has 0 own commits (identical to base) -- nothing to emit
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
        errors.append({"repo_full_name": repo_full_name, "error": str(e)})
    finally:
        def _remove_readonly(func, path, excinfo):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        try:
            shutil.rmtree(repo_path, onerror=_remove_readonly)
        except Exception as e:
            pass

    return repo_full_name, chronology_data, errors

def append_jsonl(filename: str, records: list):
    if not records:
        return
    with open(DATA_DIR / filename, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

def check_processed() -> set:
    tracker = DATA_DIR / "processed_repos.txt"
    if not tracker.exists():
        return set()
    with open(tracker, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def mark_processed(repo_name: str):
    with open(DATA_DIR / "processed_repos.txt", "a", encoding="utf-8") as f:
        f.write(f"{repo_name}\n")

def aggregate_to_parquet():
    logger.info("Aggregating incremental JSONL files to Parquet...")
    files_map = {
        "pr_commits.jsonl": ("pr_commits.parquet", ["pr_id", "sha"]),
        "errors.jsonl": ("errors.parquet", None)
    }

    for jsonl_name, (parquet_name, keys) in files_map.items():
        jsonl_path = DATA_DIR / jsonl_name
        if not jsonl_path.exists():
            continue
            
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        df = pd.DataFrame(records)
        
        if keys and not df.empty and all(k in df.columns for k in keys):
            df = df.drop_duplicates(subset=keys, keep="first")
            
        df.to_parquet(DATA_DIR / parquet_name)
        logger.info(f"Converted {jsonl_name} -> {parquet_name} ({len(df)} rows)")

def main():
    parser = argparse.ArgumentParser(description="Extract PR chronological events directly via Git refs without GitHub API.")
    parser.add_argument("--pilot", type=int, default=0, help="Run in pilot mode for N repositories.")
    parser.add_argument("--full-aidev", action="store_true", help="Process the full AIDev dataset instead of AIDev-pop.")
    args = parser.parse_args()

    universe_df = build_universe(args.pilot, args.pilot, use_full_aidev=args.full_aidev)
    processed_repos = check_processed()
    logger.info(f"Found {len(processed_repos)} already processed repositories.")

    repo_groups = []
    for name, group in universe_df.groupby('full_name'):
        if name not in processed_repos:
            repo_groups.append((name, group))

    pool_size = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Processing {len(repo_groups)} repositories with {pool_size} workers...")

    total_repo = len(repo_groups)
    if pool_size == 1:
        for i, rg in enumerate(repo_groups):
            name, chronology, errs = process_repository(rg)
            logger.info(f"[Progresso: {i+1}/{total_repo} ({(i+1)/total_repo:.1%})] Processado {name}")
            append_jsonl("pr_commits.jsonl", chronology)
            append_jsonl("errors.jsonl", errs)
            mark_processed(name)
    else:
        with multiprocessing.Pool(pool_size) as pool:
            for i, (name, chronology, errs) in enumerate(pool.imap_unordered(process_repository, repo_groups)):
                logger.info(f"[Progresso: {i+1}/{total_repo} ({(i+1)/total_repo:.1%})] Processado {name}")
                append_jsonl("pr_commits.jsonl", chronology)
                append_jsonl("errors.jsonl", errs)
                mark_processed(name)

    aggregate_to_parquet()
    logger.info("Chronology extraction completed successfully.")

if __name__ == "__main__":
    main()
