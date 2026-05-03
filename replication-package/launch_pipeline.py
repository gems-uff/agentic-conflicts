#!/usr/bin/env python3
"""Replication Pipeline Orchestrator

Single entry point for the merge conflict resolution study pipeline.

Usage:
    # Full pipeline (5-7 days on server)
    python launch_pipeline.py --aidev-dir /path/to/AIDev --data-dir ./data --workers 32

    # Pilot run (30-60 minutes)
    python launch_pipeline.py --aidev-dir /path/to/AIDev --data-dir ./data --pilot 10 --workers 4

    # Analysis only (5-10 minutes, requires pre-downloaded data)
    python launch_pipeline.py --analyze-only --data-dir ./data
"""

import argparse
import functools
import logging
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from analysis.common import load_tables
from analysis.dataset_characterization import analyze_dataset
from analysis.rq1_resolver import analyze_rq1
from analysis.rq1_detailed import analyze_rq1_detailed
from analysis.rq2_strategies import analyze_rq2
from analysis.rq2_detailed import analyze_rq2_detailed
from analysis.plotting import generate_all_figures
from src.analysis_utils import (
    process_single_repository,
    aggregate_jsonl_to_parquet,
    append_jsonl,
    cleanup_repo_scratch,
    create_final_merge_audit,
)
from src.pr_chronology import (
    extract_pr_commits,
    append_jsonl as append_jsonl_pr,
    aggregate_pr_commits_to_parquet,
)


def setup_logging(log_file: Path):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_dir_size_mb(path: Path) -> float:
    """Calculate total size of directory in MB."""
    if not path.exists():
        return 0
    try:
        return sum(f.stat().st_size for f in path.rglob('*')) / (1024 * 1024)
    except Exception:
        return 0


def log_disk_usage(data_dir: Path):
    """Log current disk usage of data directory."""
    scratch_dir = data_dir / 'scratch'
    jsonl_size_mb = sum(get_dir_size_mb(data_dir / f) for f in ['*.jsonl'] if (data_dir / f).exists())
    scratch_size_mb = get_dir_size_mb(scratch_dir)

    logging.info(f"[DISK USAGE] Scratch: {scratch_size_mb:.1f} MB, JSONL: {jsonl_size_mb:.1f} MB")


def build_universe(
    aidev_dir: Path,
    use_pilot: bool = False,
    pilot_count: int = 0,
) -> pd.DataFrame:
    """Build the analysis universe from AIDev dataset."""
    logging.info("Stage 0: Building universe DataFrame from AIDev...")

    try:
        pr_df = pd.read_parquet(aidev_dir / "all_pull_request.parquet")
        repo_df = pd.read_parquet(aidev_dir / "all_repository.parquet")
    except FileNotFoundError as e:
        logging.error(f"Required AIDev file not found: {e}")
        raise

    try:
        task_df = pd.read_parquet(aidev_dir / "pr_task_type.parquet")
    except FileNotFoundError:
        task_df = pd.DataFrame(columns=['id', 'type'])

    pr_repo_df = pr_df.merge(
        repo_df[['id', 'full_name', 'language']],
        left_on='repo_id', right_on='id', suffixes=('', '_repo'),
    )

    if not task_df.empty:
        pr_repo_task_df = pr_repo_df.merge(task_df[['id', 'type']], on='id', how='left')
        pr_repo_task_df.rename(columns={'type': 'pr_task_type'}, inplace=True)
    else:
        pr_repo_task_df = pr_repo_df.copy()
        pr_repo_task_df['pr_task_type'] = None

    cols_to_keep = ['id', 'number', 'repo_url', 'full_name', 'language',
                    'agent', 'pr_task_type', 'state']
    if 'merged_at' in pr_repo_task_df.columns:
        cols_to_keep.append('merged_at')
    cols_to_keep = [c for c in cols_to_keep if c in pr_repo_task_df.columns]

    try:
        commits_df = pd.read_parquet(aidev_dir / "pr_commits.parquet")
    except FileNotFoundError:
        logging.warning("pr_commits.parquet not found; generating from universe...")
        commits_df = pd.DataFrame({'pr_id': [], 'sha': []})

    if not commits_df.empty:
        universe_df = commits_df.merge(
            pr_repo_task_df[cols_to_keep],
            left_on='pr_id', right_on='id',
        )
    else:
        universe_df = pr_repo_task_df[cols_to_keep].rename(columns={'id': 'pr_id'})
        universe_df['sha'] = None

    if use_pilot and pilot_count > 0:
        unique_repos = universe_df['full_name'].dropna().unique()[:pilot_count]
        universe_df = universe_df[universe_df['full_name'].isin(unique_repos)]
        logging.info(f"Pilot mode: restricted to {len(unique_repos)} repositories")

    universe_df.to_parquet(Path('data') / 'universe.parquet')
    logging.info(
        f"Stage 0 complete. Universe: {len(universe_df):,} (PR, SHA) pairs "
        f"across {universe_df['full_name'].nunique()} repos"
    )
    return universe_df


def check_processed_repos(data_dir: Path) -> set:
    """Load set of already processed repositories."""
    tracker = data_dir / "processed_repos.txt"
    if not tracker.exists():
        return set()
    with open(tracker, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def mark_repo_processed(data_dir: Path, repo_name: str):
    """Mark a repository as processed."""
    tracker = data_dir / "processed_repos.txt"
    with open(tracker, 'a', encoding='utf-8') as f:
        f.write(f"{repo_name}\n")


def extract_pr_chronology(
    universe_df: pd.DataFrame,
    scratch_dir: Path,
    data_dir: Path,
    workers: int = 1,
):
    """Extract PR commit chronology using Git refs (Stage 0.5).

    For each PR, extract all commits between fork point and PR tip.
    Generates pr_commits.parquet used by subsequent analysis stages.
    """
    logging.info(f"Stage 0.5: Extracting PR chronology (workers={workers})...")

    scratch_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build list of repositories to process
    repo_groups = []
    for repo_name, repo_data in universe_df.groupby('full_name'):
        if not repo_data.empty:
            repo_groups.append((repo_name, repo_data))

    total_repos = len(repo_groups)
    logging.info(f"Extracting chronology for {total_repos} repositories...")

    # Bind scratch_dir to the function
    extract_func = functools.partial(extract_pr_commits, scratch_dir=scratch_dir)

    processed = 0
    if workers == 1:
        for i, repo_info in enumerate(repo_groups):
            repo_name, chronology, errs = extract_func(repo_info)
            logging.info(f"[{i+1}/{total_repos}] {repo_name}")

            if chronology:
                append_jsonl_pr("pr_commits.jsonl", chronology, data_dir)
                processed += 1
            if errs:
                append_jsonl_pr("pr_chronology_errors.jsonl", errs, data_dir)
    else:
        with multiprocessing.Pool(workers) as pool:
            for i, (repo_name, chronology, errs) in enumerate(
                pool.imap_unordered(extract_func, repo_groups)
            ):
                logging.info(f"[{i+1}/{total_repos}] {repo_name}")

                if chronology:
                    append_jsonl_pr("pr_commits.jsonl", chronology, data_dir)
                    processed += 1
                if errs:
                    append_jsonl_pr("pr_chronology_errors.jsonl", errs, data_dir)

    # Aggregate to parquet
    logging.info("Aggregating PR chronology to parquet...")
    aggregate_pr_commits_to_parquet(data_dir)

    logging.info(f"Stage 0.5 complete ({processed} repositories with PR commits)")


def process_repositories(
    universe_df: pd.DataFrame,
    scratch_dir: Path,
    data_dir: Path,
    workers: int = 1,
):
    """Mine internal merge commits in parallel."""
    logging.info(f"Stage 1: Mining merge commits (workers={workers})...")

    scratch_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    processed_repos = check_processed_repos(data_dir)
    logging.info(f"Found {len(processed_repos)} already processed repositories")

    repo_groups = []
    for repo_name, repo_data in universe_df.groupby('full_name'):
        if repo_data.empty or repo_name in processed_repos:
            continue
        repo_groups.append((repo_name, repo_data))

    total_repos = len(repo_groups)
    logging.info(f"Processing {total_repos} repositories...")

    # Bind scratch_dir to the function using functools.partial
    process_func = functools.partial(process_single_repository, scratch_dir=scratch_dir)

    if workers == 1:
        for i, repo_info in enumerate(repo_groups):
            repo_name, im, cc, rc, ca, errs = process_func(repo_info)
            logging.info(f"[{i+1}/{total_repos}] {repo_name}")

            append_jsonl("internal_merges.jsonl", im, data_dir)
            append_jsonl("conflict_chunks.jsonl", cc, data_dir)
            append_jsonl("resolved_chunks.jsonl", rc, data_dir)
            append_jsonl("classified_chunks.jsonl", ca, data_dir)
            append_jsonl("extraction_errors.jsonl", errs, data_dir)
            mark_repo_processed(data_dir, repo_name)

            # Log disk usage periodically
            if (i + 1) % max(1, total_repos // 10) == 0:
                log_disk_usage(data_dir)
    else:
        with multiprocessing.Pool(workers) as pool:
            for i, (repo_name, im, cc, rc, ca, errs) in enumerate(
                pool.imap_unordered(process_func, repo_groups)
            ):
                logging.info(f"[{i+1}/{total_repos}] {repo_name}")

                append_jsonl("internal_merges.jsonl", im, data_dir)
                append_jsonl("conflict_chunks.jsonl", cc, data_dir)
                append_jsonl("resolved_chunks.jsonl", rc, data_dir)
                append_jsonl("classified_chunks.jsonl", ca, data_dir)
                append_jsonl("extraction_errors.jsonl", errs, data_dir)
                mark_repo_processed(data_dir, repo_name)

                # Log disk usage periodically
                if (i + 1) % max(1, total_repos // 10) == 0:
                    log_disk_usage(data_dir)

    logging.info("Stage 1 complete")


def create_resolver_labels(data_dir: Path):
    """Create resolver labels parquet from internal_merges."""
    logging.info("Creating resolver labels...")
    im_path = data_dir / "internal_merges.parquet"
    if im_path.exists():
        df_im = pd.read_parquet(im_path)
        if not df_im.empty and "resolver_type" in df_im.columns:
            df_im[['pr_id', 'merge_sha', 'parent1_sha', 'parent2_sha',
                    'author', 'committer', 'repo_full_name', 'resolver_type']
                   ].to_parquet(data_dir / "resolver_labels.parquet")
            logging.info("Resolver labels created")


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(description='Replication package pipeline')

    parser.add_argument('--analyze-only', action='store_true',
                        help='Run only analysis (requires pre-downloaded parquet files)')
    parser.add_argument('--aidev-dir', type=str, default=None,
                        help='Path to AIDev dataset (required for full/pilot pipeline)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory for outputs (default: ./data)')
    parser.add_argument('--pilot', type=int, default=None,
                        help='Run pilot mode on N repositories')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--cleanup-scratch', action='store_true',
                        help='Aggressively clean scratch after each repo (already done in finally, this is extra)')

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    results_dir = data_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = data_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_{timestamp}.log"
    setup_logging(log_file)

    print("=" * 70)
    print("REPLICATION PACKAGE: Merge Conflict Resolution Study")
    print("=" * 70)

    if args.analyze_only:
        logging.info("\n[MODE] Analysis Only")
        logging.info(f"[DATA] Loading from: {data_dir}")

        try:
            tables = load_tables(str(data_dir))
            logging.info("[DATA] ✓ Loaded tables")

            logging.info("\n[STAGE 1] Dataset Characterization")
            analyze_dataset(tables, output_dir=str(results_dir))

            logging.info("\n[STAGE 2] RQ1 Analysis: Who Resolves?")
            analyze_rq1(tables, output_dir=str(results_dir))

            logging.info("\n[STAGE 3] RQ2 Analysis: How Do They Resolve?")
            analyze_rq2(tables, output_dir=str(results_dir))

            logging.info("\n[STAGE 4] Figure Generation")
            generate_all_figures(tables, output_dir=str(results_dir))

            logging.info("\n" + "=" * 70)
            logging.info("✓ Analysis complete!")
            logging.info(f"  Results saved to: {results_dir}")
            logging.info(f"  Log file: {log_file}")
            logging.info("=" * 70)

        except FileNotFoundError as e:
            logging.error(f"✗ ERROR: {e}")
            logging.error("Required parquet files in data directory:")
            logging.error("  - universe.parquet")
            logging.error("  - internal_merges.parquet")
            logging.error("  - conflict_chunks.parquet")
            logging.error("  - classified_chunks.parquet")
            logging.error("  - resolver_labels.parquet")
            sys.exit(1)

    else:
        if args.aidev_dir is None:
            logging.error("✗ ERROR: --aidev-dir is required for full/pilot pipeline")
            sys.exit(1)

        aidev_dir = Path(args.aidev_dir)
        if not aidev_dir.exists():
            logging.error(f"✗ ERROR: AIDev directory not found: {aidev_dir}")
            sys.exit(1)

        mode = f"PILOT ({args.pilot} repos)" if args.pilot else "FULL"
        logging.info(f"\n[MODE] {mode}")
        logging.info(f"[DATA] AIDev: {aidev_dir}")
        logging.info(f"[DATA] Output: {data_dir}")
        logging.info(f"[WORKERS] {args.workers}")
        logging.info(f"[LOG] {log_file}")

        try:
            universe_df = build_universe(
                aidev_dir,
                use_pilot=(args.pilot is not None),
                pilot_count=args.pilot or 0,
            )

            scratch_dir = data_dir / 'scratch'

            logging.info("\nStage 0.5: Extracting PR Chronology...")
            extract_pr_chronology(universe_df, scratch_dir, data_dir, workers=args.workers)

            process_repositories(universe_df, scratch_dir, data_dir, workers=args.workers)

            logging.info("\nStage 2: Aggregating JSONL to Parquet...")
            aggregate_jsonl_to_parquet(data_dir)

            logging.info("\nStage 2b: Creating final merge audit...")
            create_final_merge_audit(data_dir)

            create_resolver_labels(data_dir)

            logging.info("\n[STAGE 3] Dataset Characterization")
            tables = load_tables(str(data_dir))
            analyze_dataset(tables, output_dir=str(results_dir))

            logging.info("\n[STAGE 4] RQ1 Analysis: Who Resolves?")
            analyze_rq1(tables, output_dir=str(results_dir))
            analyze_rq1_detailed(tables, output_dir=str(results_dir))

            logging.info("\n[STAGE 5] RQ2 Analysis: How Do They Resolve?")
            analyze_rq2(tables, output_dir=str(results_dir))
            analyze_rq2_detailed(tables, output_dir=str(results_dir))

            logging.info("\n[STAGE 6] Figure Generation")
            generate_all_figures(tables, output_dir=str(results_dir))

            logging.info("\n" + "=" * 70)
            logging.info("Pipeline complete!")
            logging.info(f"  Results saved to: {results_dir}")
            logging.info(f"  Log file: {log_file}")
            log_disk_usage(data_dir)
            logging.info("=" * 70)

        except Exception as e:
            logging.error(f"✗ Pipeline failed: {e}", exc_info=True)
            sys.exit(1)


if __name__ == '__main__':
    main()
