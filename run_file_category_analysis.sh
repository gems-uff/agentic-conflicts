#!/bin/bash
# Convenience script to run both extraction and table generation steps

set -e

echo "=========================================="
echo "File Category Distribution Analysis"
echo "=========================================="
echo ""

# Step 1: Extract data
echo "Step 1: Extracting file-category data from classified chunks..."
python3 extract_file_category_data.py --data-dir ./data/nature_of_agent_conflicts --output agent_filetype_chunks.csv

if [ $? -ne 0 ]; then
    echo "ERROR: Data extraction failed"
    echo ""
    echo "Possible reasons:"
    echo "  1. Data files not found in ./data/nature_of_agent_conflicts"
    echo "  2. replication-package analysis modules not available"
    echo ""
    echo "Alternative: Manually create agent_filetype_chunks.csv with:"
    echo "  resolver,file_category,chunk_count"
    echo "  OpenAI Codex,config,6590"
    echo "  ..."
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
