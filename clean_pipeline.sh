#!/bin/bash
# Clean up pipeline outputs for fresh restart

set -e

DATA_DIR="${1:-.}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

echo "Cleaning pipeline outputs in: $DATA_DIR"
echo ""

# Confirm before deleting
read -p "This will delete all JSONL, parquet, and analysis results. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Removing files..."

# Remove intermediate JSONL files
rm -f "$DATA_DIR"/*.jsonl
echo "✓ Removed JSONL files"

# Remove parquet files (but keep raw data if exists)
rm -f "$DATA_DIR"/*.parquet
echo "✓ Removed parquet files"

# Remove analysis results
rm -rf "$DATA_DIR/results"
echo "✓ Removed results directory"

# Remove logs
rm -rf "$DATA_DIR/logs"
echo "✓ Removed logs directory"

# Remove scratch repos
rm -rf "$DATA_DIR/scratch"
echo "✓ Removed scratch directory"

# Remove processed repos tracker
rm -f "$DATA_DIR/processed_repos.txt"
echo "✓ Removed processed repos tracker"

echo ""
echo "✅ Pipeline cleaned! Ready for fresh run."
echo ""
echo "To restart pipeline:"
echo "  python launch_pipeline.py --aidev-dir ./AIDev --data-dir $DATA_DIR --pilot 10 --workers 4"
