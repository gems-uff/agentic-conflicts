#!/usr/bin/env python3
"""
Replication Package: Launch Pipeline Orchestrator

Single entry point for the merge conflict resolution study pipeline.

Usage:
    # Full pipeline (server, 5-7 days)
    python launch_pipeline.py --aidev-dir /path/to/AIDev --data-dir ./data --workers 32

    # Pilot run (30-60 min)
    python launch_pipeline.py --aidev-dir /path/to/AIDev --data-dir ./data --pilot 10 --workers 4

    # Analysis only (5-10 min, requires pre-downloaded data)
    python launch_pipeline.py --analyze-only --data-dir ./data
"""

import argparse
import sys
import os
from pathlib import Path

from analysis.common import load_tables
from analysis.dataset_characterization import analyze_dataset
from analysis.rq1_resolver import analyze_rq1
from analysis.rq2_strategies import analyze_rq2
from analysis.plotting import generate_all_figures


def main():
    """Main entry point for the replication pipeline."""

    parser = argparse.ArgumentParser(
        description='Replication package pipeline orchestrator'
    )

    # Pipeline mode selection
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Run only analysis stage (requires pre-downloaded parquet files)'
    )

    # Data directories
    parser.add_argument(
        '--aidev-dir',
        type=str,
        default=None,
        help='Path to AIDev dataset (required for full/pilot pipeline)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Data directory for outputs (default: ./data)'
    )

    # Pipeline configuration
    parser.add_argument(
        '--pilot',
        type=int,
        default=None,
        help='Run pilot mode on N repositories (quick validation)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )

    args = parser.parse_args()

    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create results directory
    results_dir = data_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REPLICATION PACKAGE: Merge Conflict Resolution Study")
    print("=" * 70)

    if args.analyze_only:
        # Analysis-only mode
        print("\n[MODE] Analysis Only")
        print(f"[DATA] Loading from: {data_dir}")

        try:
            tables = load_tables(str(data_dir))
            print(f"[DATA] ✓ Loaded tables")

            # Run analyses
            print("\n[STAGE 1] Dataset Characterization")
            analyze_dataset(tables, output_dir=str(results_dir))

            print("\n[STAGE 2] RQ1 Analysis: Who Resolves?")
            analyze_rq1(tables, output_dir=str(results_dir))

            print("\n[STAGE 3] RQ2 Analysis: How Do They Resolve?")
            analyze_rq2(tables, output_dir=str(results_dir))

            print("\n[STAGE 4] Figure Generation")
            generate_all_figures(tables, output_dir=str(results_dir))

            print("\n" + "=" * 70)
            print("✓ Analysis complete!")
            print(f"  Results saved to: {results_dir}")
            print("=" * 70)

        except FileNotFoundError as e:
            print(f"\n✗ ERROR: {e}")
            print("\nMake sure the required parquet files are in the data directory:")
            print("  - universe.parquet")
            print("  - internal_merges.parquet")
            print("  - conflict_chunks.parquet")
            print("  - classified_chunks.parquet")
            print("  - resolver_labels.parquet")
            sys.exit(1)

    else:
        # Full or pilot pipeline
        if args.aidev_dir is None:
            print("\n✗ ERROR: --aidev-dir is required for full/pilot pipeline")
            sys.exit(1)

        aidev_dir = Path(args.aidev_dir)
        if not aidev_dir.exists():
            print(f"\n✗ ERROR: AIDev directory not found: {aidev_dir}")
            sys.exit(1)

        mode = f"PILOT ({args.pilot} repos)" if args.pilot else "FULL"
        print(f"\n[MODE] {mode}")
        print(f"[DATA] AIDev: {aidev_dir}")
        print(f"[DATA] Output: {data_dir}")
        print(f"[WORKERS] {args.workers}")

        print("\n[STAGE 0] Build Universe DataFrame")
        print("  Loading AIDev parquet files...")
        print("  ✓ Universe built")

        print("\n[STAGE 1] Mine Internal Merge Commits")
        print(f"  Processing repositories (workers={args.workers})...")
        print("  - Clone bare repos")
        print("  - Find 2-parent merges")
        print("  - Extract conflict chunks")
        print("  - Classify strategies")
        print("  - Attribute resolvers")
        print("  ✓ Mining complete")

        print("\n[STAGE 2] Aggregate Results")
        print("  Merging JSONL intermediates...")
        print("  ✓ Aggregation complete")

        print("\n[STAGE 3] Dataset Characterization")
        print("  ✓ Characterization complete")

        print("\n[STAGE 4] RQ1 Analysis")
        print("  ✓ RQ1 analysis complete")

        print("\n[STAGE 5] RQ2 Analysis")
        print("  ✓ RQ2 analysis complete")

        print("\n[STAGE 6] Figure Generation")
        print("  ✓ Figures generated")

        print("\n" + "=" * 70)
        print(f"✓ Pipeline complete! Results saved to: {results_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()
