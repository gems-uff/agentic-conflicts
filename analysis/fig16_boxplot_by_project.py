"""
Reprodução da Fig. 16 de Ghiotto et al. com os dados do estudo.

Para cada projeto, calcula a porcentagem de chunks conflitantes resolvidos
por cada estratégia, e plota box plots da distribuição dessas porcentagens.

Uso:
    python fig16_boxplot_by_project.py

Saída:
    analysis/figures/rq2_boxplot_by_project.png
    analysis/figures/rq2_boxplot_by_project.pdf
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "nature_of_agent_conflicts"
FIGURES_DIR  = PROJECT_ROOT / "analysis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Strategy mapping (raw → canonical, following PLAN.md §5.4) ─────────────
RAW_TO_CANONICAL = {
    "V1":         "V1",
    "V2":         "V2",
    "ConcatV1V2": "CC",
    "ConcatV2V1": "CC",
    "Combination": "CB",
    "New code":   "NC",
    "None":       "NN",
    "Imprecise":  "Imprecise",
    "Postponed":  "Imprecise",
}

# STRATEGY_ORDER = ["V1", "V2", "CC", "CB", "NC", "NN", "Imprecise"]
STRATEGY_ORDER = ["V1", "V2", "CC", "CB", "NC", "NN"]

STRATEGY_PALETTE = {
    "V1":        "#0072B2",
    "V2":        "#D55E00",
    "CC":        "#009E73",
    "CB":        "#CC79A7",
    "NC":        "#E69F00",
    "NN":        "#56B4E9",
    "Imprecise": "#BFBFBF",
}

# ── Load & deduplicate ──────────────────────────────────────────────────────
print("Loading classified_chunks.parquet ...")
chunks = pd.read_parquet(DATA_DIR / "classified_chunks.parquet")

# Deduplicate on natural key (same logic as common.load_tables)
CHUNK_KEY = ["repo_full_name", "merge_sha", "file_path", "chunk_index"]
chunks = (
    chunks
    .sort_values(CHUNK_KEY + (["pr_id"] if "pr_id" in chunks.columns else []))
    .drop_duplicates(subset=CHUNK_KEY, keep="first")
)
print(f"  {len(chunks):,} unique chunks across {chunks['repo_full_name'].nunique():,} repos")

# ── Map to canonical labels ─────────────────────────────────────────────────
chunks["strategy_canon"] = chunks["strategy"].map(RAW_TO_CANONICAL).fillna("Imprecise")

# ── Per-repo percentage distribution ───────────────────────────────────────
# For each repo: what % of its chunks were resolved by each strategy?
repo_counts = (
    chunks
    .groupby(["repo_full_name", "strategy_canon"])
    .size()
    .rename("n")
    .reset_index()
)
repo_totals = chunks.groupby("repo_full_name").size().rename("total").reset_index()
repo_counts = repo_counts.merge(repo_totals, on="repo_full_name")
repo_counts["pct"] = repo_counts["n"] / repo_counts["total"] * 100

# Pivot to wide format; repos with 0 chunks for a strategy get 0%
repo_wide = (
    repo_counts
    .pivot(index="repo_full_name", columns="strategy_canon", values="pct")
    .reindex(columns=STRATEGY_ORDER)
    .fillna(0)
)

print(f"  Per-repo matrix: {repo_wide.shape[0]} repos × {repo_wide.shape[1]} strategies")
print(repo_wide.describe().round(1).to_string())

# ── Plot ────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(8, 5))

data = [repo_wide[s].values for s in STRATEGY_ORDER]

bplot = ax.boxplot(
    data,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=0.9),
    capprops=dict(linewidth=0.9),
    boxprops=dict(linewidth=0.9),
    flierprops=dict(
        marker="o", markersize=2.5, linestyle="none",
        markerfacecolor="none", markeredgewidth=0.5,
    ),
    showfliers=True,
)

for patch, strat in zip(bplot["boxes"], STRATEGY_ORDER):
    patch.set_facecolor(STRATEGY_PALETTE[strat])
    patch.set_alpha(0.85)

ax.set_xticks(range(1, len(STRATEGY_ORDER) + 1))
ax.set_xticklabels(STRATEGY_ORDER, fontsize=10)
ax.set_xlabel("Developer decision", fontsize=11)
ax.set_ylabel("Conflicting chunks (%)", fontsize=11)
ax.set_ylim(-2, 102)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = FIGURES_DIR / f"rq2_boxplot_by_project.{ext}"
    dpi = 150 if ext == "png" else None
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
