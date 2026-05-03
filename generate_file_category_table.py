#!/usr/bin/env python3
"""
Generate file-category distribution table for agents.

This script extracts the file-category composition for each agent's self-resolved
chunks and produces both a percentage table and LaTeX code ready to paste into
the paper.

USAGE:
    python3 generate_file_category_table.py <input_csv> [--output OUTPUT_PREFIX]

INPUT FORMAT:
    CSV file with columns: resolver, file_category, chunk_count

    Example:
    resolver,file_category,chunk_count
    OpenAI Codex,config,6590
    OpenAI Codex,source,104
    OpenAI Codex,lock,1
    Copilot,lock,650
    Copilot,source,126
    ...

OUTPUT:
    - Console: formatted table + LaTeX code
    - file_category_distribution.csv: percentage table
    - file_category_counts.csv: absolute counts
"""

import pandas as pd
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', help='CSV file with resolver, file_category, chunk_count')
    parser.add_argument('--output', default='file_category', help='Output prefix for CSV files')
    args = parser.parse_args()

    # Load data
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{args.input_csv}'")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        sys.exit(1)

    # Validate columns
    required_cols = {'resolver', 'file_category', 'chunk_count'}
    if not required_cols.issubset(set(df.columns)):
        print(f"ERROR: CSV must have columns: {required_cols}")
        print(f"Found: {set(df.columns)}")
        sys.exit(1)

    # Create pivot table: agents × file categories
    pivot_counts = df.pivot_table(
        index='resolver',
        columns='file_category',
        values='chunk_count',
        aggfunc='sum',
        fill_value=0
    )

    # Convert to percentages (each row sums to 100%)
    pivot_pct = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100

    # Define desired agent order
    agent_order = ['OpenAI Codex', 'Copilot', 'Devin', 'Cursor', 'Claude Code']
    agents_in_data = [a for a in agent_order if a in pivot_pct.index]
    if not agents_in_data:
        print("ERROR: No recognized agents found in resolver column")
        print(f"Expected one of: {agent_order}")
        print(f"Found: {list(pivot_pct.index)}")
        sys.exit(1)

    pivot_pct = pivot_pct.reindex(agents_in_data)
    pivot_counts = pivot_counts.reindex(agents_in_data)

    # Define desired file category order
    file_order = ['source', 'lock', 'config', 'generated', 'doc', 'migration']
    files_in_data = [f for f in file_order if f in pivot_pct.columns]
    pivot_pct = pivot_pct[files_in_data]
    pivot_counts = pivot_counts[files_in_data]

    # Print summary
    print("\n" + "="*80)
    print("FILE CATEGORY DISTRIBUTION (percentages by agent)")
    print("="*80 + "\n")
    print(pivot_pct.to_string(float_format=lambda x: f'{x:.1f}%'))

    print("\n" + "="*80)
    print("ABSOLUTE COUNTS (chunks)")
    print("="*80 + "\n")
    print(pivot_counts.to_string(float_format=lambda x: f'{int(x):,}'))

    # Generate LaTeX table
    print("\n" + "="*80)
    print("LaTeX TABLE (copy from line below to paste in paper)")
    print("="*80 + "\n")

    # Create LaTeX with proper formatting
    latex_lines = []
    latex_lines.append(r"\begin{table}")
    latex_lines.append(r"  \caption{File-category distribution of self-resolved chunks per agent (\%).}")
    latex_lines.append(r"  \label{tab:file-category-by-agent}")
    latex_lines.append(r"  \small\setlength{\tabcolsep}{3pt}")

    # Build tabular header
    col_spec = "l" + "r" * len(files_in_data)
    latex_lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append(r"    \toprule")

    # Header row
    header = "    Agent"
    for col in files_in_data:
        header += f" & {col.capitalize()}"
    header += r" \\"
    latex_lines.append(header)

    latex_lines.append(r"    \midrule")

    # Data rows
    for agent in agents_in_data:
        row = f"    {agent}"
        for col in files_in_data:
            val = pivot_pct.loc[agent, col]
            if pd.isna(val) or val == 0:
                row += " & ---"
            else:
                row += f" & {val:.1f}\\%"
        row += r" \\"
        latex_lines.append(row)

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_table = "\n".join(latex_lines)
    print(latex_table)

    # Save to CSV files
    pct_file = f"{args.output}_distribution.csv"
    count_file = f"{args.output}_counts.csv"

    pivot_pct.to_csv(pct_file)
    pivot_counts.to_csv(count_file)

    print(f"\n✓ Saved percentage table to: {pct_file}")
    print(f"✓ Saved absolute counts to: {count_file}")

    print("\n" + "="*80)
    print("COPY THE LaTeX TABLE ABOVE AND PASTE IT IN paper/main.tex")
    print("="*80)

if __name__ == '__main__':
    main()
