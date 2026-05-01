"""Shared helpers for the RQ1 / RQ2 / RQ3 analysis notebooks.

The pipeline (``extract_aidev_nature.py``) writes a set of Parquet tables
under ``data/nature_of_agent_conflicts/``. This module centralises:

* loading those tables into a :class:`AnalysisTables` bundle,
* joining chunk-level data with PR- and merge-level context (``agent``,
  ``language``, ``pr_task_type``, ``state``, ``resolver_type``) so every
  notebook starts from the same analysis frame,
* canonical strategy labels and colour palette (PLAN.md §5.4),
* the three cross-cutting stratifications (agent, language, pr_task_type)
  with top-N collapsing for language,
* consistent Matplotlib / Seaborn styling,
* a :func:`save_fig` helper that writes every figure as both PDF (for the
  paper) and PNG (for quick previews).

Usage from a notebook::

    from analysis.common import (
        load_tables, build_chunk_frame, build_merge_frame, build_pr_frame,
        setup_style, save_fig, stratify, STRATEGY_ORDER, STRATEGY_PALETTE,
    )

    setup_style()
    tables = load_tables()
    chunks = build_chunk_frame(tables)
    # ... plot ...
    save_fig(fig, "rq1_chunks_per_merge_global")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Resolve project root so notebooks work regardless of the active CWD
# (Jupyter commonly launches the kernel in the notebook's own folder).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nature_of_agent_conflicts"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Canonical labels (PLAN.md §5.4)
# --------------------------------------------------------------------------- #

# The seven buckets the study reports. ``identify_resolution`` emits finer
# labels (ConcatV1V2 / ConcatV2V1, Combination, New code, None, Postponed)
# which we fold here to stay comparable with Ghiotto et al. (TSE 2020).
STRATEGY_ORDER = ["V1", "V2", "CC", "CB", "NC", "NN", "Imprecise"]

_RAW_TO_CANONICAL = {
    "V1": "V1",
    "V2": "V2",
    "ConcatV1V2": "CC",
    "ConcatV2V1": "CC",
    "Combination": "CB",
    "New code": "NC",
    "None": "NN",
    "Imprecise": "Imprecise",
    # ``Postponed`` (resolution still contains conflict markers) is rare in
    # practice and has no Ghiotto counterpart; folded into Imprecise so it
    # stays visible as "not a conventional resolution".
    "Postponed": "Imprecise",
}

# Qualitative palette (Okabe-Ito: colour-blind-safe) stable across
# figures. The strategies are additionally distinguished by hatches
# (``STRATEGY_HATCH``) so every figure remains readable when the paper
# is printed in black & white; the luminance order of the colours
# (V1 darkest → Imprecise lightest) reinforces the grayscale cue.
STRATEGY_PALETTE = {
    "V1": "#0072B2",  # deep blue     (darkest)
    "V2": "#D55E00",  # vermillion
    "CC": "#009E73",  # bluish-green
    "CB": "#CC79A7",  # reddish-purple
    "NC": "#E69F00",  # orange
    "NN": "#56B4E9",  # sky blue      (light)
    "Imprecise": "#BFBFBF",  # neutral gray (lightest)
}

# Hatch pattern per strategy. Empty string = solid fill. These are
# applied on top of the fill colour so the figures stay readable in
# grayscale / photocopy reproduction.
STRATEGY_HATCH = {
    "V1": "",
    "V2": "",
    "CC": "//",
    "CB": "\\\\",
    "NC": "xx",
    "NN": "++",
    "Imprecise": "..",
}

RESOLVER_ORDER = ["agent", "human"]
RESOLVER_PALETTE = {"agent": "#0072B2", "human": "#D55E00"}
RESOLVER_HATCH = {"agent": "", "human": "//"}

PR_OUTCOME_ORDER = ["resolved", "abandoned-with-conflict", "open-with-conflict"]
PR_OUTCOME_PALETTE = {
    "resolved": "#009E73",
    "abandoned-with-conflict": "#D55E00",
    "open-with-conflict": "#BFBFBF",
}
PR_OUTCOME_HATCH = {
    "resolved": "",
    "abandoned-with-conflict": "xx",
    "open-with-conflict": "..",
}

# How many languages to show before folding the long tail into "Other".
TOP_N_LANGUAGES = 8


# --------------------------------------------------------------------------- #
# Table loading
# --------------------------------------------------------------------------- #


@dataclass
class AnalysisTables:
    """Raw Parquet tables produced by ``extract_aidev_nature.py``."""

    universe: pd.DataFrame
    internal_merges: pd.DataFrame
    classified_chunks: pd.DataFrame
    resolved_chunks: pd.DataFrame
    extraction_errors: pd.DataFrame


def _read_parquet_optional(path: Path) -> pd.DataFrame:
    """Return an empty DataFrame if the file hasn't been produced yet."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_tables(data_dir: Path | str = DATA_DIR, deduplicate: bool = True) -> AnalysisTables:
    """Read every Parquet output of Pass B into a bundle.

    Missing files are returned as empty DataFrames so a notebook can still
    render partial results when the pipeline is in progress.

    When ``deduplicate=True`` (the default), tables are collapsed on their
    natural keys so that the same physical merge commit is counted once
    regardless of how many PRs reference it. This is important because a
    merge commit can appear on the head branches of several PRs
    (force-push / re-open / cherry-pick), and Pass B's JSONL→Parquet
    aggregation does not always remove these duplicates.

    Natural keys used:

    * ``internal_merges``   → ``(repo_full_name, merge_sha)``
    * ``conflict_chunks``   → ``(repo_full_name, merge_sha, file_path, chunk_index)``
    * ``resolved_chunks``   → ``(repo_full_name, merge_sha, file_path, chunk_index)``
    * ``classified_chunks`` → ``(repo_full_name, merge_sha, file_path, chunk_index)``

    We keep ``pr_id`` on the kept row (one representative PR, chosen by
    smallest ``pr_id`` to be deterministic) so the notebooks can still
    join PR-level context (agent, language, pr_task_type). When finer
    analyses need the full list of PRs that referenced a merge, use
    :func:`pr_fanout` to materialise it from the raw tables.
    """
    data_dir = Path(data_dir)
    tables = AnalysisTables(
        universe=_read_parquet_optional(data_dir / "universe.parquet"),
        internal_merges=_read_parquet_optional(data_dir / "internal_merges.parquet"),
        classified_chunks=_read_parquet_optional(data_dir / "classified_chunks.parquet"),
        resolved_chunks=_read_parquet_optional(data_dir / "resolved_chunks.parquet"),
        extraction_errors=_read_parquet_optional(data_dir / "extraction_errors.parquet"),
    )
    if deduplicate:
        tables = _deduplicate_tables(tables)
    return tables


_MERGE_KEY = ["repo_full_name", "merge_sha"]
_CHUNK_KEY = ["repo_full_name", "merge_sha", "file_path", "chunk_index"]


def _dedup_on(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Sort by (subset + pr_id) and keep the first row per ``subset``.

    Sorting by pr_id makes the kept row deterministic across runs when
    multiple PRs reference the same merge commit.
    """
    if df.empty or not all(c in df.columns for c in subset):
        return df
    sort_cols = list(subset)
    if "pr_id" in df.columns:
        sort_cols.append("pr_id")
    return df.sort_values(sort_cols).drop_duplicates(subset=subset, keep="first")


def _deduplicate_tables(tables: AnalysisTables) -> AnalysisTables:
    return AnalysisTables(
        universe=tables.universe,  # PR-level table; not keyed on merges
        internal_merges=_dedup_on(tables.internal_merges, _MERGE_KEY),
        classified_chunks=_dedup_on(tables.classified_chunks, _CHUNK_KEY),
        resolved_chunks=_dedup_on(tables.resolved_chunks, _CHUNK_KEY),
        extraction_errors=tables.extraction_errors,  # audit table; PR-level entries kept as-is
    )


def pr_fanout(internal_merges_raw: pd.DataFrame) -> pd.DataFrame:
    """Return ``(repo_full_name, merge_sha, n_prs, pr_ids)`` from the raw
    (un-deduplicated) ``internal_merges`` table.

    Useful to audit how many PRs reference each merge commit.
    """
    if internal_merges_raw.empty:
        return internal_merges_raw
    grp = (
        internal_merges_raw
        .groupby(_MERGE_KEY)["pr_id"]
        .agg(list)
        .rename("pr_ids")
        .reset_index()
    )
    grp["n_prs"] = grp["pr_ids"].str.len()
    return grp


# --------------------------------------------------------------------------- #
# Analysis frames
# --------------------------------------------------------------------------- #


def _pr_context(universe: pd.DataFrame) -> pd.DataFrame:
    """One row per ``pr_id`` carrying the cross-cutting strata.

    ``universe`` carries one row per ``(pr_id, sha)``; PR-level attributes
    (``agent``, ``language``, ``state``, ``merged_at``, ``pr_task_type``)
    are constant within a PR, so we just drop the SHA dimension.
    """
    if universe.empty:
        return universe
    cols = [c for c in (
        "pr_id", "full_name", "agent", "language",
        "pr_task_type", "state", "merged_at",
    ) if c in universe.columns]
    ctx = universe[cols].drop_duplicates(subset=["pr_id"])
    return ctx


def _merge_context(internal_merges: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(repo_full_name, merge_sha)`` carrying the resolver label.

    When ``load_tables`` is called with ``deduplicate=True`` (default) the
    input frame already carries a single row per physical merge; this
    helper is therefore idempotent under the default workflow. It keeps
    the defensive ``drop_duplicates`` so it remains safe if a caller
    builds an ``AnalysisTables`` bundle from raw data.
    """
    if internal_merges.empty:
        return internal_merges
    cols = [c for c in (
        "pr_id", "merge_sha", "repo_full_name", "author",
        "committer", "resolver_type",
    ) if c in internal_merges.columns]
    return _dedup_on(internal_merges[cols], _MERGE_KEY)


def _canonicalize_strategy(df: pd.DataFrame) -> pd.DataFrame:
    if "strategy" not in df.columns:
        return df
    df = df.copy()
    df["strategy_raw"] = df["strategy"]
    df["strategy"] = df["strategy"].map(_RAW_TO_CANONICAL).fillna("Imprecise")
    df["strategy"] = pd.Categorical(df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    return df


def build_chunk_frame(tables: AnalysisTables) -> pd.DataFrame:
    """Chunk-level analysis frame.

    Joins ``classified_chunks`` with:

    * ``resolver_type`` from ``internal_merges``, and
    * ``agent`` / ``language`` / ``pr_task_type`` / ``state`` / ``merged_at``
      from ``universe``.

    Canonicalizes the strategy label to the seven-bucket scheme
    (PLAN.md §5.4) and applies language top-N folding.

    Returned columns (present only when the underlying data is):

        repo_full_name, pr_id, merge_sha, file_path, chunk_index,
        v1, base, v2, v1_loc, v2_loc, base_loc,
        resolution, resolution_loc, localized_ok,
        strategy, strategy_raw,
        resolver_type, agent, language, language_top, pr_task_type,
        state, merged_at
    """
    chunks = tables.classified_chunks
    if chunks.empty:
        return chunks

    chunks = _canonicalize_strategy(chunks)

    mctx = _merge_context(tables.internal_merges)
    if not mctx.empty:
        # Join on the physical merge (repo, sha) rather than on pr_id so
        # chunks inherit a single canonical resolver label even if the
        # raw tables still contained duplicated pr_id rows. Drop any
        # stale pr_id column coming from mctx to keep the chunk-side
        # pr_id as the ground truth.
        chunks = chunks.merge(
            mctx[[c for c in ("repo_full_name", "merge_sha", "resolver_type") if c in mctx.columns]],
            on=[c for c in ("repo_full_name", "merge_sha") if c in mctx.columns and c in chunks.columns],
            how="left",
        )

    pctx = _pr_context(tables.universe)
    if not pctx.empty:
        chunks = chunks.merge(pctx, on="pr_id", how="left")
        chunks = _apply_language_topn(chunks)

    return chunks


def build_merge_frame(tables: AnalysisTables) -> pd.DataFrame:
    """Merge-level analysis frame: one row per internal merge commit.

    The frame is keyed on ``(repo_full_name, merge_sha)`` -- a single
    physical merge commit contributes exactly one row even if several
    PRs reference it (force-push / re-open / cherry-pick). Adds
    per-merge aggregates (``n_chunks``, ``has_conflict``) plus the
    PR-level strata (via one representative ``pr_id``). This is the
    frame used for RQ1's "#chunks per merge" distribution and for
    RQ3's per-merge resolver breakdown.
    """
    merges = tables.internal_merges
    if merges.empty:
        return merges

    # Enforce the merge-level key. ``load_tables`` already does this for
    # the default code path; repeating the step here keeps the function
    # robust when called with a non-deduplicated bundle.
    merges = _dedup_on(merges, _MERGE_KEY).copy()

    # Chunks per merge -- zero when the merge produced no textual conflict
    # (the merge row exists but there's no matching classified_chunks row).
    # Counted on the chunk-level natural key so duplicated chunk rows (if
    # any survived) do not inflate the per-merge count.
    if not tables.classified_chunks.empty:
        chunk_src = _dedup_on(tables.classified_chunks, _CHUNK_KEY)
        per_merge = (
            chunk_src
            .groupby(["repo_full_name", "merge_sha"])
            .size()
            .rename("n_chunks")
            .reset_index()
        )
        merges = merges.merge(per_merge, on=["repo_full_name", "merge_sha"], how="left")
        merges["n_chunks"] = merges["n_chunks"].fillna(0).astype(int)
    else:
        merges["n_chunks"] = 0

    merges["has_conflict"] = merges["n_chunks"] > 0

    pctx = _pr_context(tables.universe)
    if not pctx.empty:
        merges = merges.merge(pctx, on="pr_id", how="left")
        merges = _apply_language_topn(merges)

    return merges


def build_pr_frame(tables: AnalysisTables) -> pd.DataFrame:
    """PR-level analysis frame: one row per PR in the universe.

    Adds:

    * ``has_internal_merge`` -- the PR has at least one two-parent commit
      that reached Stage 2.
    * ``has_conflict``       -- at least one of its internal merges
      produced a conflict chunk.
    * ``pr_outcome`` (PLAN.md §5.6) -- one of:

        * ``resolved``                 (merged, or closed with no conflict)
        * ``abandoned-with-conflict``  (closed, ``merged_at`` null, has conflict)
        * ``open-with-conflict``       (open, has conflict)
    """
    pctx = _pr_context(tables.universe)
    if pctx.empty:
        return pctx

    prs = pctx.copy()

    if not tables.internal_merges.empty:
        # At PR level we intentionally do NOT deduplicate by physical merge:
        # if a PR references five merge commits, it should count five -- the
        # duplication we're guarding against is at the "same sha appears
        # under several pr_ids" axis, which does not affect this per-PR
        # aggregation.
        im_per_pr = (
            tables.internal_merges
            .groupby("pr_id")
            .size()
            .rename("n_internal_merges")
            .reset_index()
        )
        prs = prs.merge(im_per_pr, on="pr_id", how="left")
    else:
        prs["n_internal_merges"] = 0
    prs["n_internal_merges"] = prs["n_internal_merges"].fillna(0).astype(int)
    prs["has_internal_merge"] = prs["n_internal_merges"] > 0

    if not tables.classified_chunks.empty:
        conflicting_prs = tables.classified_chunks["pr_id"].drop_duplicates()
        prs["has_conflict"] = prs["pr_id"].isin(conflicting_prs)
    else:
        prs["has_conflict"] = False

    def _outcome(row: pd.Series) -> str:
        state = str(row.get("state", "")).lower()
        has_conflict = bool(row.get("has_conflict", False))
        merged_at = row.get("merged_at", None)
        if not has_conflict:
            return "resolved"
        if state == "closed" and pd.isna(merged_at):
            return "abandoned-with-conflict"
        if state == "open":
            return "open-with-conflict"
        # Merged PRs with conflict chunks are still "resolved" -- the conflict
        # was handled during the PR and the PR landed on base.
        return "resolved"

    prs["pr_outcome"] = prs.apply(_outcome, axis=1)
    prs["pr_outcome"] = pd.Categorical(
        prs["pr_outcome"], categories=PR_OUTCOME_ORDER, ordered=True,
    )

    prs = _apply_language_topn(prs)
    return prs


# --------------------------------------------------------------------------- #
# Stratification
# --------------------------------------------------------------------------- #


def _apply_language_topn(df: pd.DataFrame, n: int = TOP_N_LANGUAGES) -> pd.DataFrame:
    """Collapse the language long tail into ``"Other"``.

    The untouched column is kept under ``language``; the folded column is
    added as ``language_top``. Empty / null languages are rendered as
    ``"Unknown"`` for consistency.
    """
    if "language" not in df.columns:
        return df
    df = df.copy()
    lang = df["language"].fillna("Unknown").replace("", "Unknown")
    top = lang.value_counts().head(n).index.tolist()
    df["language_top"] = lang.where(lang.isin(top), other="Other")
    return df


STRATA = {
    "agent": "agent",
    "language": "language_top",
    "pr_task_type": "pr_task_type",
}


def stratify(df: pd.DataFrame, axis: str) -> Iterator[tuple[str, pd.DataFrame]]:
    """Iterate over ``(stratum_value, sub_df)`` pairs for an axis.

    ``axis`` is one of ``agent`` / ``language`` / ``pr_task_type``. Strata are
    ordered by descending row count so the most important slices appear
    first in stratified figures. Null values become the string
    ``"Unknown"`` so they render on the plot rather than being dropped.
    """
    col = STRATA.get(axis, axis)
    if col not in df.columns:
        return iter([])
    values = df[col].fillna("Unknown").astype(str)
    order = values.value_counts().index.tolist()
    for value in order:
        yield value, df[values == value]


def stratum_order(df: pd.DataFrame, axis: str) -> list[str]:
    col = STRATA.get(axis, axis)
    if col not in df.columns:
        return []
    values = df[col].fillna("Unknown").astype(str)
    return values.value_counts().index.tolist()


# --------------------------------------------------------------------------- #
# Styling & figure saving
# --------------------------------------------------------------------------- #


def setup_style() -> None:
    """Apply the paper-figure defaults.

    The defaults are deliberately austere: serif font, no chart titles
    (we never set them -- callers drop the title and let the LaTeX
    caption do the labelling), thin hatches that still reproduce after
    photocopying, and PDF/TTF embedding so the figures stay editable
    in the final paper.
    """
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        # We never set chart titles in the notebooks (see comment above);
        # the LaTeX caption carries that text. The rc entries below keep
        # the size consistent if any stray title does appear.
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
        # Keep hatches thin so stacked bars stay legible after photocopy
        # reproduction and so the colour underneath is still visible.
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,   # embed TrueType (editable in LaTeX / Illustrator)
        "ps.fonttype": 42,
    })


def _apply_hatches(ax: plt.Axes, order: list[str], hatches: dict[str, str]) -> None:
    """Apply ``hatches`` to the segments of a stacked bar chart.

    Matplotlib's ``DataFrame.plot(kind="bar", stacked=True)`` produces
    one ``BarContainer`` per stacked segment (in the same order as the
    columns). We walk them in ``order`` and assign the matching hatch
    to every patch. Each patch also gets a thin black edge so the
    segments remain separated in grayscale.
    """
    containers = [c for c in ax.containers if hasattr(c, "patches")]
    for segment, container in zip(order, containers):
        h = hatches.get(segment, "")
        for patch in container.patches:
            patch.set_hatch(h)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)


def plot_strategy_stacked(
    df: pd.DataFrame,
    order: list[str] | None = None,
    orientation: str = "vertical",
    ax: plt.Axes | None = None,
    palette: dict[str, str] | None = None,
    hatches: dict[str, str] | None = None,
    annotate_n: bool = True,
    legend: bool = True,
) -> plt.Axes:
    """Stacked-bar plot of a strategy distribution.

    ``df`` is expected to be a row-normalised frame with one row per
    stratum and columns = strategies (the output of
    :func:`strategy_distribution` with a ``group_col``).

    ``orientation='horizontal'`` produces Ghiotto-Figure-16-style
    horizontal 100%-stacked bars (one per stratum, stratum name on the
    y-axis, strategy segments colour- and hatch-coded). ``'vertical'``
    keeps the older notebook layout.

    The helper is deliberately style-free w.r.t. titles: callers that
    want a title should add it afterwards. Keeping the helper
    title-less matches the paper-figure convention (LaTeX caption
    carries the label).
    """
    if order is None:
        order = [c for c in STRATEGY_ORDER if c in df.columns]
    if palette is None:
        palette = STRATEGY_PALETTE
    if hatches is None:
        hatches = STRATEGY_HATCH

    df = df.reindex(columns=order).fillna(0)

    if ax is None:
        if orientation == "horizontal":
            height = max(2.2, 0.45 * len(df) + 1.2)
            fig, ax = plt.subplots(figsize=(7.2, height))
        else:
            width = max(4.5, 0.9 * len(df) + 2)
            fig, ax = plt.subplots(figsize=(width, 3.6))

    kind = "barh" if orientation == "horizontal" else "bar"
    (df * 100).plot(
        kind=kind, stacked=True,
        color=[palette[s] for s in order],
        ax=ax,
        width=0.8,
        legend=False,
    )
    _apply_hatches(ax, order, hatches)

    if orientation == "horizontal":
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of chunks")
        ax.set_ylabel("")
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    else:
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of chunks")
        ax.set_xlabel("")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    if annotate_n and "n" in df.attrs:
        counts = df.attrs["n"]
        labels = [f"n={counts.get(idx, 0):,}" for idx in df.index]
        if orientation == "horizontal":
            for i, lbl in enumerate(labels):
                ax.text(101, i, lbl, va="center", ha="left", fontsize=8,
                        color="dimgray")
        else:
            for i, lbl in enumerate(labels):
                ax.text(i, 101, lbl, ha="center", va="bottom", fontsize=8,
                        color="dimgray")

    if legend:
        handles = [
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=palette[s],
                          hatch=hatches.get(s, ""),
                          edgecolor="black", linewidth=0.5)
            for s in order
        ]
        if orientation == 'horizontal' and annotate_n:
            ax.legend(
                handles, order,
                title="Strategy",
                bbox_to_anchor=(1.12, 1), loc="upper left",
                frameon=False,
            )
        else:
            ax.legend(
                handles, order,
                title="Strategy",
                bbox_to_anchor=(1.02, 1), loc="upper left",
                frameon=False,
            )
        

    return ax


def strategy_frame_for_plot(
    chunks: pd.DataFrame,
    group_col: str,
    exclude_imprecise: bool = True,
    sort_by_count: bool = True,
    exclude_unknown: bool = True,
) -> pd.DataFrame:
    """Build a plot-ready row-normalised strategy frame.

    Wraps :func:`strategy_distribution` and attaches per-stratum chunk
    counts under ``df.attrs['n']`` so :func:`plot_strategy_stacked`
    can annotate each bar with ``n=...``. When
    ``sort_by_count=True`` rows are ordered by total (classifiable)
    chunk count, descending — the same ordering Ghiotto et~al.\\ use
    in their Figure 16.
    """
    dist = strategy_distribution(chunks, group_col=group_col,
                                 exclude_imprecise=exclude_imprecise)
    if exclude_imprecise:
        counted = chunks[chunks["strategy"] != "Imprecise"]
    else:
        counted = chunks
    if exclude_unknown:
        counted = counted[counted['language_top']!='Unknown']
    
    counts = counted.groupby(group_col).size()
    if sort_by_count:
        dist = dist.reindex(counts.sort_values(ascending=False).index)
    dist.attrs["n"] = counts.to_dict()
    return dist


def save_fig(fig: plt.Figure, name: str, directory: Path | str = FIGURES_DIR) -> tuple[Path, Path]:
    """Save ``fig`` as both ``{name}.pdf`` and ``{name}.png``.

    Returns the two written paths. ``name`` should not include the
    extension. A gentle ``tight_layout()`` is applied before saving.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / f"{name}.pdf"
    png_path = directory / f"{name}.png"
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    return pdf_path, png_path


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def descriptive_table(series: pd.Series) -> pd.Series:
    """Ghiotto-style descriptive stats for a numeric series."""
    return pd.Series({
        "n": series.size,
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "p25": series.quantile(0.25),
        "median": series.median(),
        "p75": series.quantile(0.75),
        "p90": series.quantile(0.90),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99),
        "max": series.max(),
    })


def strategy_distribution(
    chunks: pd.DataFrame,
    group_col: str | None = None,
    exclude_imprecise: bool = False,
) -> pd.DataFrame:
    """Row-normalised strategy distribution, optionally per-group.

    Returns a frame with columns = strategies, rows = groups (or a single
    row called ``"all"`` when ``group_col`` is None).

    When ``exclude_imprecise=True`` the ``Imprecise`` bucket is dropped
    from both the numerator and the denominator, so the reported
    fractions describe the distribution *among chunks we could
    classify*. Use :func:`imprecise_share` to report the excluded
    fraction alongside.
    """
    if chunks.empty:
        cols = [s for s in STRATEGY_ORDER if s != "Imprecise"] if exclude_imprecise else STRATEGY_ORDER
        return pd.DataFrame(columns=cols)
    if exclude_imprecise:
        chunks = chunks[chunks["strategy"] != "Imprecise"]
        order = [s for s in STRATEGY_ORDER if s != "Imprecise"]
    else:
        order = STRATEGY_ORDER
    if group_col is None:
        counts = chunks["strategy"].value_counts().reindex(order, fill_value=0)
        total = counts.sum()
        if total == 0:
            return pd.DataFrame(0, index=["all"], columns=order)
        return (counts / total).to_frame("all").T
    grouped = (
        chunks.groupby(group_col)["strategy"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )
    row_sums = grouped.sum(axis=1).replace(0, pd.NA)
    return grouped.div(row_sums, axis=0).fillna(0)


def imprecise_share(chunks: pd.DataFrame, group_col: str | None = None) -> pd.Series | float:
    """Fraction of chunks whose strategy is ``Imprecise``.

    Returned as a scalar when ``group_col`` is None, or as a per-group
    Series otherwise. Useful to report alongside a distribution produced
    with ``exclude_imprecise=True``.
    """
    if chunks.empty:
        return 0.0 if group_col is None else pd.Series(dtype=float)
    if group_col is None:
        total = len(chunks)
        if total == 0:
            return 0.0
        return float((chunks["strategy"] == "Imprecise").sum() / total)
    return (
        chunks.assign(_is_imp=(chunks["strategy"] == "Imprecise").astype(int))
        .groupby(group_col)["_is_imp"]
        .mean()
    )
