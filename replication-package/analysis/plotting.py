"""Figure generation utilities for the replication package."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

from .common import (
    AnalysisTables, STRATEGY_PALETTE,
    build_chunk_frame, build_merge_frame,
    stratum_order, setup_style, save_fig,
)


def plot_rq1_chunks_per_merge_global(tables: AnalysisTables, output_dir: str = './results'):
    """RQ1 Figure: Chunks per failed merge (histogram + CDF)."""
    if not plt or tables.internal_merges is None:
        logging.warning("Cannot plot RQ1 chunks per merge global")
        return

    merges = build_merge_frame(tables)
    failed_merges = merges[merges["has_conflict"]].copy()

    if failed_merges.empty:
        logging.warning("No failed merges for RQ1 chunks plot")
        return

    failed_merges["n_chunks_capped"] = failed_merges["n_chunks"].clip(upper=18)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))

    sns.histplot(
        failed_merges["n_chunks_capped"],
        bins=np.arange(0.5, 19.5, 1),
        ax=axes[0],
        color=STRATEGY_PALETTE.get("V1", "#0072B2"),
    )
    xticks = list(range(1, 19))
    xticklabels = [str(i) for i in xticks]
    xticklabels[-1] = "18+"
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels)
    axes[0].set_xlabel("(a) Conflicting chunks per merge")
    axes[0].set_ylabel("Number of merges")

    sorted_n = np.sort(failed_merges["n_chunks"].values)
    cdf = np.arange(1, len(sorted_n) + 1) / len(sorted_n)
    axes[1].plot(sorted_n, cdf, color=STRATEGY_PALETTE.get("V1", "#0072B2"), lw=1.6)
    axes[1].set_xscale("symlog")
    axes[1].set_xlabel("(b) Conflicting chunks per merge (symlog)")
    axes[1].set_ylabel("Empirical CDF")

    save_fig(fig, "rq1_chunks_per_merge_global", output_dir)


def plot_rq1_loc_distributions_global(tables: AnalysisTables, output_dir: str = './results'):
    """RQ1 Figure: Chunk-level LOC distributions (v1, v2, resolution)."""
    if not plt or tables.classified_chunks is None:
        logging.warning("Cannot plot RQ1 LOC distributions global")
        return

    chunks = build_chunk_frame(tables)
    if chunks.empty:
        logging.warning("No chunks for RQ1 LOC plot")
        return

    metrics = [("v1_loc", "v1 lines of code"),
               ("v2_loc", "v2 lines of code"),
               ("resolution_loc", "resolution lines of code")]
    metrics = [(c, lab) for c, lab in metrics if c in chunks.columns]

    if not metrics:
        logging.warning("Required LOC columns not found")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.2), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics):
        data = chunks[col].clip(upper=chunks[col].quantile(0.99))
        sns.histplot(data, bins=40, ax=ax, color=STRATEGY_PALETTE.get("V2", "#D55E00"))
        ax.set_yscale("log")
        ax.set_xlabel(label)
    axes[0].set_ylabel("Number of chunks (log)")

    save_fig(fig, "rq1_loc_distributions_global", output_dir)


def plot_rq1_chunks_per_merge_by_agent(tables: AnalysisTables, output_dir: str = './results'):
    """RQ1 Figure: Chunks per merge stratified by agent."""
    if not plt or tables.internal_merges is None:
        logging.warning("Cannot plot RQ1 chunks by agent")
        return

    merges = build_merge_frame(tables)
    failed_merges = merges[merges["has_conflict"] & merges["agent"].notna()].copy()

    if failed_merges.empty:
        logging.warning("No failed merges with agent data")
        return

    agent_order = stratum_order(failed_merges, "agent")
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(agent_order) + 2), 3.4))
    sns.boxplot(
        data=failed_merges, x="agent", y="n_chunks",
        order=agent_order, showfliers=False, ax=ax, color=STRATEGY_PALETTE.get("V1", "#0072B2"),
    )
    ax.set_xlabel("Agent")
    ax.set_ylabel("Chunks per merge")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    save_fig(fig, "rq1_chunks_per_merge_by_agent", output_dir)


def plot_rq1_loc_by_agent(tables: AnalysisTables, output_dir: str = './results'):
    """RQ1 Figure: Chunk LOC by agent (faceted)."""
    if not plt or tables.classified_chunks is None:
        logging.warning("Cannot plot RQ1 LOC by agent")
        return

    chunks = build_chunk_frame(tables)
    if chunks.empty or "agent" not in chunks.columns:
        logging.warning("Chunks missing agent column")
        return

    metrics = [("v1_loc", "V1 LoC"),
               ("v2_loc", "V2 LoC"),
               ("resolution_loc", "Resolution LoC")]
    metrics = [(c, lab) for c, lab in metrics if c in chunks.columns]

    if not metrics:
        logging.warning("LOC columns not found")
        return

    long_df = chunks.melt(
        id_vars=["agent"],
        value_vars=[c for c, _ in metrics],
        var_name="metric", value_name="loc",
    ).dropna()

    agent_order = stratum_order(chunks, "agent")
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.4), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics):
        d = long_df[long_df["metric"] == col]
        sns.boxplot(data=d, x="agent", y="loc", order=agent_order,
                    showfliers=False, ax=ax, color=STRATEGY_PALETTE.get("V2", "#D55E00"))
        ax.set_xlabel(label)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    axes[0].set_ylabel("LOC")

    save_fig(fig, "rq1_loc_by_agent", output_dir)


def plot_rq1_chunks_per_merge_by_language(tables: AnalysisTables, output_dir: str = './results'):
    """RQ1 Figure: Chunks per merge stratified by language (top-N)."""
    if not plt or tables.internal_merges is None:
        logging.warning("Cannot plot RQ1 chunks by language")
        return

    merges = build_merge_frame(tables)
    failed_merges = merges[merges["has_conflict"] & merges["language_top"].notna()].copy()

    if failed_merges.empty:
        logging.warning("No failed merges with language data")
        return

    lang_order = stratum_order(failed_merges, "language_top")
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(lang_order) + 2), 3.4))
    sns.boxplot(
        data=failed_merges, x="language_top", y="n_chunks",
        order=lang_order, showfliers=False, ax=ax, color=STRATEGY_PALETTE.get("CC", "#009E73"),
    )
    ax.set_xlabel("Language (top)")
    ax.set_ylabel("Chunks per merge")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    save_fig(fig, "rq1_chunks_per_merge_by_language", output_dir)


def plot_rq1_chunks_per_merge_by_task_type(tables: AnalysisTables, output_dir: str = './results'):
    """RQ1 Figure: Chunks per merge stratified by PR task type."""
    if not plt or tables.internal_merges is None:
        logging.warning("Cannot plot RQ1 chunks by task type")
        return

    merges = build_merge_frame(tables)
    subset = merges[merges["has_conflict"] & merges["pr_task_type"].notna()].copy()

    if subset.empty:
        logging.info("pr_task_type not populated (requires AIDev-pop)")
        return

    task_order = stratum_order(subset, "pr_task_type")
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(task_order) + 2), 3.4))
    sns.boxplot(
        data=subset, x="pr_task_type", y="n_chunks",
        order=task_order, showfliers=False, ax=ax, color=STRATEGY_PALETTE.get("CB", "#CC79A7"),
    )
    ax.set_xlabel("PR task type")
    ax.set_ylabel("Chunks per merge")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    save_fig(fig, "rq1_chunks_per_merge_by_task_type", output_dir)


def plot_rq2_size_distributions(tables: AnalysisTables, output_dir: str = './results'):
    """RQ2 Figure: Size distributions (chunks + LOC)."""
    if not plt or tables.classified_chunks is None:
        logging.warning("Cannot plot RQ2 size distributions")
        return

    chunks = build_chunk_frame(tables)
    merges = build_merge_frame(tables)
    conflict_merges = merges[merges["n_chunks"] > 0]

    if chunks.empty or conflict_merges.empty:
        logging.warning("Missing data for RQ2 size plot")
        return

    # Color palette
    _BLUE = "#0072B2"
    _VERMILLION = "#D55E00"
    _GREEN = "#009E73"

    loc_cfg = [
        ("v1_loc", "V1 LoC", _BLUE),
        ("v2_loc", "V2 LoC", _VERMILLION),
        ("resolution_loc", "Resolution LoC", _GREEN),
    ]
    loc_series = []
    loc_labels = []
    loc_colors = []
    for col, label, color in loc_cfg:
        if col in chunks.columns:
            vals = chunks[col].dropna()
            if not vals.empty:
                loc_series.append(vals.values)
                loc_labels.append(label)
                loc_colors.append(color)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.2, 3.6))

    # Left — chunks per merge
    bp_l = ax_l.boxplot(
        conflict_merges["n_chunks"].values,
        patch_artist=True,
        showfliers=False,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker=".", markersize=3, alpha=0.4,
                        markerfacecolor=_BLUE, markeredgecolor=_BLUE),
    )
    bp_l["boxes"][0].set_facecolor(_BLUE)
    bp_l["boxes"][0].set_alpha(0.7)
    bp_l["boxes"][0].set_edgecolor("black")
    bp_l["boxes"][0].set_linewidth(0.8)
    ax_l.set_xticks([1])
    ax_l.set_xticklabels(["Chunks"])
    ax_l.set_ylabel("Chunks per merge")
    ax_l.grid(axis="x", visible=False)

    # Right — LoC distributions
    if loc_series:
        bp_r = ax_r.boxplot(
            loc_series,
            patch_artist=True,
            showfliers=False,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, color in zip(bp_r["boxes"], loc_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        for fliers, color in zip(bp_r["fliers"], loc_colors):
            fliers.set_markerfacecolor(color)
            fliers.set_markeredgecolor(color)
        ax_r.set_xticks(range(1, len(loc_labels) + 1))
        ax_r.set_xticklabels(loc_labels, rotation=15, ha="right")
        ax_r.set_ylabel("Lines of code (LoC)")
        ax_r.grid(axis="x", visible=False)

    save_fig(fig, "rq2_size_distributions", output_dir)


def generate_all_figures(tables: AnalysisTables, output_dir: str = './results'):
    """Generate all publication figures from the analysis notebooks."""
    logging.info("Generating figures...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        setup_style()

        logging.info("Generating RQ1 figures...")
        plot_rq1_chunks_per_merge_global(tables, output_dir)
        plot_rq1_loc_distributions_global(tables, output_dir)
        plot_rq1_chunks_per_merge_by_agent(tables, output_dir)
        plot_rq1_loc_by_agent(tables, output_dir)
        plot_rq1_chunks_per_merge_by_language(tables, output_dir)
        plot_rq1_chunks_per_merge_by_task_type(tables, output_dir)

        logging.info("Generating RQ2 figures...")
        plot_rq2_size_distributions(tables, output_dir)

        logging.info("Figure generation complete")
    except Exception as e:
        logging.warning(f"Figure generation failed: {e}")


__all__ = [
    'setup_style',
    'plot_rq1_chunks_per_merge_global',
    'plot_rq1_loc_distributions_global',
    'plot_rq1_chunks_per_merge_by_agent',
    'plot_rq1_loc_by_agent',
    'plot_rq1_chunks_per_merge_by_language',
    'plot_rq1_chunks_per_merge_by_task_type',
    'plot_rq2_size_distributions',
    'generate_all_figures',
]
