#!/bin/bash
# Convenience script to run both extraction and table generation steps

set -e

echo "=========================================="
echo "File Category Distribution Analysis"
echo "=========================================="
echo ""

# Step 1: Extract data
echo "Step 1: Extracting file-category data from agent-resolved chunks..."
python3 extract_agent_file_categories.py --data-dir ./data/nature_of_agent_conflicts_paper --output agent_filetype_chunks.csv

if [ $? -ne 0 ]; then
    echo "ERROR: Data extraction failed"
    echo ""
    echo "Make sure you have:"
    echo "  - pandas installed: pip3 install pandas pyarrow"
    echo "  - Data files in ./data/nature_of_agent_conflicts/"
    echo "    (resolved_chunks.parquet, resolver_labels.parquet)"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Generating LaTeX table..."
python3 generate_file_category_table.py agent_filetype_chunks.csv

echo ""
echo "=========================================="
echo "DONE! "
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy the LaTeX table from above"
echo "2. Replace the \"File-category conditioning\" paragraph in paper/main.tex"
echo "   with the new table"
echo ""