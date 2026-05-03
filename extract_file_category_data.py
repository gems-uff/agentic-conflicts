#!/usr/bin/env python3
"""
Extract file-category distribution data from classified chunks.

This script reads the processed chunks data and generates a CSV with the
file-category distribution per agent, ready for the file_category_table.py script.

REQUIREMENTS:
    - replication-package must be set up in Python path
    - Data files should be in the standard location

USAGE:
    python3 extract_file_category_data.py [--data-dir DATA_DIR] [--output OUTPUT_CSV]

"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Try to import from replication-package
try:
    sys.path.insert(0, str(Path(__file__).parent / 'replication-package'))
    from analysis.common import load_tables, build_chunk_frame
except ImportError as e:
    print(f"ERROR: Could not import replication-package analysis modules")
    print(f"Details: {e}")
    print("\nAlternatively, provide a CSV with columns:")
    print("  resolver, file_category, chunk_count")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',
                       default='./data/nature_of_agent_conflicts',
                       help='Directory containing classified chunks parquet files')
    parser.add_argument('--output',
                       default='agent_filetype_chunks.csv',
                       help='Output CSV file')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print(f"\nTry running:")
        print(f"  python3 extract_file_category_data.py --data-dir /path/to/data")
        sys.exit(1)

    logger.info(f"Loading data from: {data_dir}")

    # Load analysis tables
    try:
        tables = load_tables(data_dir=str(data_dir))
        logger.info("✓ Loaded analysis tables")
    except Exception as e:
        logger.error(f"ERROR loading analysis tables: {e}")
        logger.error(f"Traceback: {type(e).__name__}")
        sys.exit(1)

    # Build chunk frame with file categories
    try:
        chunks = build_chunk_frame(tables)
        if chunks is None or len(chunks) == 0:
            logger.error("ERROR: No chunks found in data")
            sys.exit(1)
        logger.info(f"✓ Loaded {len(chunks):,} chunks")
    except Exception as e:
        logger.error(f"ERROR building chunk frame: {e}")
        sys.exit(1)

    # Filter to agent-resolved chunks only
    if 'resolver' in chunks.columns:
        agent_resolved = chunks[chunks['resolver'] == 'agent'].copy()
    else:
        # Alternative: filter by agent column if resolver not available
        if 'agent' in chunks.columns:
            agent_resolved = chunks[chunks['agent'].notna()].copy()
        else:
            logger.error("ERROR: Could not find 'resolver' or 'agent' column")
            logger.info(f"Available columns: {chunks.columns.tolist()}")
            sys.exit(1)

    logger.info(f"✓ Filtered to {len(agent_resolved):,} agent-resolved chunks")

    # Ensure file_category column exists
    if 'file_category' not in agent_resolved.columns:
        logger.error("ERROR: 'file_category' column not found in chunks")
        logger.info(f"Available columns: {agent_resolved.columns.tolist()}")
        sys.exit(1)

    # Map resolver to agent name
    if 'agent' in agent_resolved.columns:
        agent_col = 'agent'
    elif 'pr_author_agent' in agent_resolved.columns:
        agent_col = 'pr_author_agent'
    else:
        logger.error("ERROR: Could not find agent column")
        sys.exit(1)

    # Group by agent and file_category, count chunks
    grouped = agent_resolved.groupby([agent_col, 'file_category']).size().reset_index(name='chunk_count')

    # Rename agent column to 'resolver' for consistency with main script
    grouped = grouped.rename(columns={agent_col: 'resolver'})

    # Save to CSV
    grouped.to_csv(args.output, index=False)
    logger.info(f"✓ Saved {len(grouped)} rows to {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for agent in grouped['resolver'].unique():
        agent_data = grouped[grouped['resolver'] == agent]
        total_chunks = agent_data['chunk_count'].sum()
        print(f"\n{agent}:")
        print(f"  Total chunks: {total_chunks:,}")
        print("  By file category:")
        for _, row in agent_data.iterrows():
            pct = row['chunk_count'] / total_chunks * 100
            print(f"    {row['file_category']:12} {row['chunk_count']:6,} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print(f"✓ Data saved to: {args.output}")
    print("✓ Next: run 'python3 generate_file_category_table.py " + args.output + "'")
    print("="*80)

if __name__ == '__main__':
    main()
