import json
import pandas as pd
from pathlib import Path

# Expected values from the paper (Table per-agent-vs-human)
EXPECTED_CHUNKS_BY_AGENT = {
    "OpenAI Codex": 6695,
    "Devin": 1235,
    "Copilot": 776,
    "Cursor": 165,
    "Claude Code": 108,
}
EXPECTED_TOTAL_AGENT_CHUNKS = 8979

# Load the JSON results
results_path = Path("./analysis/results/results_main3_codex.json")
if not results_path.exists():
    print(f"ERROR: {results_path} not found")
    exit(1)

with open(results_path, 'r') as f:
    results = json.load(f)

# Extract H_path_breakdown data
h_data = results.get("H_path_breakdown", {})
agent_data = h_data.get("by_agent_self_resolved", {})

if not agent_data:
    print("ERROR: No agent data found in H_path_breakdown")
    exit(1)

# Build a list of rows for the CSV: agent, file_category, chunk_count
rows = []
agent_totals = {}

for agent_name, agent_info in agent_data.items():
    if agent_info is None:
        continue
    agent_display = agent_name.replace("_", " ")
    cat_counts = agent_info.get("category_counts", {})
    agent_total = sum(cat_counts.values())
    agent_totals[agent_display] = agent_total

    for category, count in cat_counts.items():
        rows.append({
            "resolver": agent_display,
            "file_category": category,
            "chunk_count": count
        })

df = pd.DataFrame(rows)

if len(df) == 0:
    print("ERROR: No data extracted")
    exit(1)

# ============================================================
# SANITY CHECKS
# ============================================================
print("\n" + "="*80)
print("SANITY CHECKS - Validating against expected paper values")
print("="*80 + "\n")

# Check 1: Total chunks
total_chunks = df['chunk_count'].sum()
print(f"✓ Total agent-resolved chunks: {total_chunks:,}")
if total_chunks == EXPECTED_TOTAL_AGENT_CHUNKS:
    print(f"  ✓ MATCHES expected {EXPECTED_TOTAL_AGENT_CHUNKS:,}")
else:
    print(f"  ⚠ WARNING: Expected {EXPECTED_TOTAL_AGENT_CHUNKS:,}, got {total_chunks:,}")
    print(f"    Difference: {total_chunks - EXPECTED_TOTAL_AGENT_CHUNKS:+,}")

# Check 2: Per-agent totals
print(f"\n✓ Per-agent chunk counts:")
all_match = True
for agent in ["OpenAI Codex", "Copilot", "Devin", "Cursor", "Claude Code"]:
    actual = agent_totals.get(agent, 0)
    expected = EXPECTED_CHUNKS_BY_AGENT.get(agent, 0)
    status = "✓" if actual == expected else "⚠"
    print(f"  {status} {agent:20s}: {actual:6,} (expected {expected:6,})", end="")
    if actual != expected:
        print(f" DIFF: {actual - expected:+,}", end="")
        all_match = False
    print()

if all_match:
    print(f"\n✓ All per-agent counts MATCH expected values!")
else:
    print(f"\n⚠ Some per-agent counts DO NOT match. Check data.")

# Check 3: Categories present
print(f"\n✓ File categories found: {sorted(df['file_category'].unique())}")

# Create pivot table: agents × file categories
pivot_counts = df.pivot_table(
    index='resolver',
    columns='file_category',
    values='chunk_count',
    aggfunc='sum',
    fill_value=0
)

# Convert to percentages
pivot_pct = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100

# Check 4: Percentages sum to 100% per agent
print(f"\n✓ Verifying percentages sum to 100% per agent:")
for agent in pivot_pct.index:
    pct_sum = pivot_pct.loc[agent].sum()
    status = "✓" if abs(pct_sum - 100.0) < 0.1 else "⚠"
    print(f"  {status} {agent:20s}: {pct_sum:7.1f}%")

# Define desired agent order
agent_order = ['OpenAI Codex', 'Copilot', 'Devin', 'Cursor', 'Claude Code']
agents_in_data = [a for a in agent_order if a in pivot_pct.index]

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

print("\n" + "="*80)
print("✓ Copy the LaTeX table above and paste it in paper/main.tex")
print("="*80)