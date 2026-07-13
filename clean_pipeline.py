#!/usr/bin/env python3
"""Clean up pipeline outputs for fresh restart."""

import argparse
import shutil
import sys
from pathlib import Path


def clean_pipeline(data_dir: Path, force: bool = False):
    """Remove all pipeline outputs for fresh restart."""
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    print(f"Cleaning pipeline outputs in: {data_dir}")
    print()

    if not force:
        response = input("This will delete all JSONL, parquet, and analysis results. Continue? (y/N) ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    print("Removing files...")

    # Remove JSONL files
    for f in data_dir.glob("*.jsonl"):
        f.unlink()
    print("✓ Removed JSONL files")

    # Remove parquet files
    for f in data_dir.glob("*.parquet"):
        f.unlink()
    print("✓ Removed parquet files")

    # Remove results directory
    results_dir = data_dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    print("✓ Removed results directory")

    # Remove logs directory
    logs_dir = data_dir / "logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
    print("✓ Removed logs directory")

    # Remove scratch directory
    scratch_dir = data_dir / "scratch"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    print("✓ Removed scratch directory")

    # Remove processed repos tracker
    tracker = data_dir / "processed_repos.txt"
    if tracker.exists():
        tracker.unlink()
    print("✓ Removed processed repos tracker")

    print()
    print("✅ Pipeline cleaned! Ready for fresh run.")
    print()
    print("To restart pipeline:")
    print(f"  python launch_pipeline.py --aidev-dir ./AIDev --data-dir {data_dir} --pilot 10 --workers 4")


def main():
    parser = argparse.ArgumentParser(description="Clean up pipeline outputs")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory to clean (default: ./data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()
    clean_pipeline(Path(args.data_dir), force=args.force)


if __name__ == "__main__":
    main()
