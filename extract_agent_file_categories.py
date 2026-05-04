#!/usr/bin/env python3
"""
Extract file-category distribution for agent-resolved chunks.

This script:
1. Loads resolved_chunks (has file paths and chunk info)
2. Loads internal_merges (has merge_sha and resolver_type)
3. Loads universe (has agent info)
4. Joins them to get chunks resolved by agents with their file categories

USAGE:
    python3 extract_agent_file_categories.py [--data-dir DATA_DIR] [--output OUTPUT_CSV]
"""

import sys
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def categorize_filepath(filepath):
    """Categorize a file path into one of 6 categories."""
    if not isinstance(filepath, str):
        return 'source'

    # Extract extension and basename
    basename = Path(filepath).name
    extension = Path(filepath).suffix.lower()
    path_str = filepath.lower()

    # Lock files
    lock_files = {
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'npm-shrinkwrap.json',
        'pipfile.lock', 'poetry.lock', 'uv.lock', 'gemfile.lock', 'composer.lock',
        'cargo.lock', 'go.sum', 'mix.lock', 'flake.lock', 'pubspec.lock',
        'podfile.lock', 'bun.lockb'
    }
    if basename.lower() in lock_files:
        return 'lock'

    # Generated files
    if extension in {'.min.js', '.min.css', '.bundle.js', '.map', '.snap'}:
        return 'generated'
    build_dirs = {'dist/', 'build/', 'out/', 'node_modules/', 'vendor/', 'target/',
                  'coverage/', '__pycache__/', '.next/', '.nuxt/', '.cache/', '.parcel-cache/'}
    if any(f'/{d}' in path_str or f'\\{d.rstrip("/")}\\' in path_str for d in build_dirs):
        return 'generated'

    # Documentation
    if extension in {'.md', '.rst', '.txt', '.adoc'}:
        return 'doc'

    # Config
    if extension in {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env'}:
        return 'config'

    # Migration
    if extension == '.sql' or '/migration' in path_str or '/migrations/' in path_str:
        return 'migration'

    # Default: source
    return 'source'

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',
                       default='./data/nature_of_agent_conflicts',
                       help='Directory containing parquet files')
    parser.add_argument('--output',
                       default='agent_filetype_chunks.csv',
                       help='Output CSV file')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Check if data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info(f"Loading data from: {data_dir}")

    try:
        # Load all required parquet files
        resolved = pd.read_parquet(data_dir / 'resolved_chunks.parquet')
        logger.info(f"✓ Loaded {len(resolved):,} resolved chunks")

        internal_merges = pd.read_parquet(data_dir / 'internal_merges.parquet')
        logger.info(f"✓ Loaded {len(internal_merges):,} internal merges")

        universe = pd.read_parquet(data_dir / 'universe.parquet')
        logger.info(f"✓ Loaded {len(universe):,} universe records")

    except FileNotFoundError as e:
        logger.error(f"ERROR: Missing parquet file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ERROR loading parquet files: {e}")
        sys.exit(1)

    # Step 1: Join resolved_chunks with internal_merges on merge_sha
    try:
        merged = resolved.merge(
            internal_merges[['merge_sha', 'resolver_type', 'pr_id']],
            on='merge_sha',
            how='left'
        )
        logger.info(f"✓ Merged resolved_chunks with internal_merges: {len(merged):,} chunks")
    except Exception as e:
        logger.error(f"ERROR merging resolved_chunks with internal_merges: {e}")
        sys.exit(1)

    # Step 2: Filter to agent-resolved chunks only
    if 'resolver_type' not in merged.columns:
        logger.error("ERROR: 'resolver_type' column not found after merge")
        logger.info(f"Available columns: {merged.columns.tolist()}")
        sys.exit(1)

    logger.info(f"Resolver types: {merged['resolver_type'].unique()}")

    agent_resolved = merged[merged['resolver_type'] == 'agent'].copy()
    logger.info(f"✓ Filtered to {len(agent_resolved):,} agent-resolved chunks")

    if len(agent_resolved) == 0:
        logger.error("ERROR: No agent-resolved chunks found!")
        sys.exit(1)

    # Step 3: Join with universe to get agent name
    try:
        # Merge on pr_id to get agent (use pr_id_x since pr_id got renamed after first merge)
        agent_resolved = agent_resolved.merge(
            universe[['id', 'agent']],
            left_on='pr_id_x',
            right_on='id',
            how='left'
        )
        logger.info(f"✓ Merged with universe to get agent info")
    except Exception as e:
        logger.error(f"ERROR merging with universe: {e}")
        logger.info(f"Columns available: {agent_resolved.columns.tolist()}")
        sys.exit(1)

    # Verify agent column exists
    if 'agent' not in agent_resolved.columns:
        logger.error("ERROR: 'agent' column not found after merge with universe")
        logger.info(f"Available columns: {agent_resolved.columns.tolist()}")
        sys.exit(1)

    # Step 4: Categorize files and group by agent and file_category
    agent_resolved['file_category'] = agent_resolved['file_path'].apply(categorize_filepath)

    grouped = agent_resolved.groupby(['agent', 'file_category']).size().reset_index(name='chunk_count')

    # Rename 'agent' column to 'resolver' for compatibility with table generation script
    grouped = grouped.rename(columns={'agent': 'resolver'})

    # Save to CSV
    grouped.to_csv(args.output, index=False)
    logger.info(f"✓ Saved {len(grouped)} rows to {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: File-category distribution of agent-resolved chunks")
    print("="*80)
    for agent in sorted(grouped['resolver'].unique()):
        agent_data = grouped[grouped['resolver'] == agent]
        total_chunks = agent_data['chunk_count'].sum()
        print(f"\n{agent}:")
        print(f"  Total chunks: {total_chunks:,}")
        print("  By file category:")
        for _, row in agent_data.sort_values('chunk_count', ascending=False).iterrows():
            pct = row['chunk_count'] / total_chunks * 100
            print(f"    {row['file_category']:12} {row['chunk_count']:6,} ({pct:5.1f}%)")

    total_all = grouped['chunk_count'].sum()
    print(f"\n  TOTAL AGENT-RESOLVED CHUNKS: {total_all:,}")

    print("\n" + "="*80)
    print(f"✓ Data saved to: {args.output}")
    print("✓ Next: run 'python3 generate_file_category_table.py " + args.output + "'")
    print("="*80)

if __name__ == '__main__':
    main()