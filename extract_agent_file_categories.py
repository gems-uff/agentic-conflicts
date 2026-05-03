#!/usr/bin/env python3
"""
Extract file-category distribution for agent-resolved chunks.

This script reads the processed parquet files (resolved_chunks.parquet and
resolver_labels.parquet) and generates a CSV with the distribution of
agent-resolved chunks by agent and file category.

USAGE:
    python3 extract_agent_file_categories.py [--data-dir DATA_DIR] [--output OUTPUT_CSV]

REQUIREMENTS:
    - pandas
    - pyarrow (for parquet support)
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
        logger.error(f"Expected parquet files in: {data_dir}")
        sys.exit(1)

    # Load required parquet files
    logger.info(f"Loading data from: {data_dir}")

    try:
        # Load resolved chunks
        resolved_file = data_dir / 'resolved_chunks.parquet'
        if not resolved_file.exists():
            logger.error(f"resolved_chunks.parquet not found in {data_dir}")
            sys.exit(1)

        resolved = pd.read_parquet(resolved_file)
        logger.info(f"✓ Loaded {len(resolved):,} resolved chunks")

        # Load PR universe to get agent info
        universe_file = data_dir / 'aidev_pop_universe.parquet'
        if not universe_file.exists():
            logger.error(f"aidev_pop_universe.parquet not found in {data_dir}")
            sys.exit(1)

        universe = pd.read_parquet(universe_file)
        logger.info(f"✓ Loaded universe data")

    except Exception as e:
        logger.error(f"ERROR loading parquet files: {e}")
        sys.exit(1)

    # Merge resolved chunks with universe to get agent info
    try:
        # Merge on pr_id to get agent information
        merged = resolved.merge(universe[['pr_id', 'bot_author']], on='pr_id', how='left')
        logger.info(f"✓ Merged data: {len(merged):,} chunks")
    except Exception as e:
        logger.error(f"ERROR merging data: {e}")
        logger.info("Columns in resolved_chunks:", resolved.columns.tolist())
        logger.info("Columns in universe:", universe.columns.tolist())
        sys.exit(1)

    # Filter to agent-resolved chunks only
    logger.info(f"Resolver types in data: {merged['resolver_type'].unique() if 'resolver_type' in merged.columns else 'N/A'}")

    if 'resolver_type' not in merged.columns:
        logger.error("ERROR: 'resolver_type' column not found in merged data")
        logger.info(f"Available columns: {merged.columns.tolist()}")
        sys.exit(1)

    agent_resolved = merged[merged['resolver_type'] == 'agent'].copy()
    logger.info(f"✓ Filtered to {len(agent_resolved):,} agent-resolved chunks")

    if len(agent_resolved) == 0:
        logger.error("ERROR: No agent-resolved chunks found!")
        sys.exit(1)

    # Get agent name (PR author agent)
    if 'bot_author' not in agent_resolved.columns:
        logger.error("ERROR: 'bot_author' column not found")
        logger.info(f"Available columns: {agent_resolved.columns.tolist()}")
        sys.exit(1)

    # Rename bot_author to agent for consistency
    agent_resolved = agent_resolved.rename(columns={'bot_author': 'agent'})

    # Get file path and categorize
    if 'file_path' not in agent_resolved.columns:
        logger.error("ERROR: 'file_path' column not found")
        logger.info(f"Available columns: {agent_resolved.columns.tolist()}")
        sys.exit(1)

    agent_resolved['file_category'] = agent_resolved['file_path'].apply(categorize_filepath)

    # Group by agent and file_category, count chunks
    grouped = agent_resolved.groupby(['agent', 'file_category']).size().reset_index(name='chunk_count')

    # Rename 'agent' column to 'resolver' for consistency with table generation script
    grouped = grouped.rename(columns={'agent': 'resolver'})

    # Save to CSV
    grouped.to_csv(args.output, index=False)
    logger.info(f"✓ Saved {len(grouped)} rows to {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
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

    print("\n" + "="*80)
    print(f"✓ Data saved to: {args.output}")
    print("✓ Next: run 'python3 generate_file_category_table.py " + args.output + "'")
    print("="*80)

if __name__ == '__main__':
    main()
