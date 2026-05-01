"""Shared analysis utilities and data loading for the replication package."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterator
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

STRATEGY_ORDER = ["V1", "V2", "CC", "CB", "NC", "NN", "Imprecise", "Postponed"]

STRATEGY_PALETTE = {
    "V1": "#0072B2",
    "V2": "#D55E00",
    "CC": "#009E73",
    "CB": "#CC79A7",
    "NC": "#E69F00",
    "NN": "#56B4E9",
    "Imprecise": "#BFBFBF",
    "Postponed": "#BFBFBF",
}

RESOLVER_ORDER = ["agent", "human"]
RESOLVER_PALETTE = {"agent": "#0072B2", "human": "#D55E00"}

TOP_N_LANGUAGES = 8

# Natural keys for deduplication
_MERGE_KEY = ["repo_full_name", "merge_sha"]
_CHUNK_KEY = ["repo_full_name", "merge_sha", "file_path", "chunk_index"]

# Stratification axes
STRATA = {
    "agent": "agent",
    "language": "language_top",
    "pr_task_type": "pr_task_type",
}

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "results"


@dataclass
class AnalysisTables:
    """Raw Parquet tables produced by the pipeline."""

    universe: pd.DataFrame
    internal_merges: pd.DataFrame
    classified_chunks: pd.DataFrame
    resolved_chunks: pd.DataFrame = None
    extraction_errors: pd.DataFrame = None
    resolver_labels: pd.DataFrame = None

    def __post_init__(self):
        # Fill in None values with empty DataFrames
        if self.resolved_chunks is None:
            self.resolved_chunks = pd.DataFrame()
        if self.extraction_errors is None:
            self.extraction_errors = pd.DataFrame()
        if self.resolver_labels is None:
            self.resolver_labels = pd.DataFrame()


def _read_parquet_optional(path: Path) -> pd.DataFrame:
    """Return an empty DataFrame if the file doesn't exist."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _dedup_on(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Sort by (subset + pr_id) and keep the first row per subset."""
    if df.empty or not all(c in df.columns for c in subset):
        return df
    sort_cols = list(subset)
    if "pr_id" in df.columns:
        sort_cols.append("pr_id")
    return df.sort_values(sort_cols).drop_duplicates(subset=subset, keep="first")


def _pr_context(universe: pd.DataFrame) -> pd.DataFrame:
    """One row per pr_id carrying cross-cutting strata."""
    if universe.empty:
        return universe
    cols = [c for c in (
        "pr_id", "full_name", "agent", "language",
        "pr_task_type", "state", "merged_at",
    ) if c in universe.columns]
    ctx = universe[cols].drop_duplicates(subset=["pr_id"])
    return ctx


def _apply_language_topn(df: pd.DataFrame, n: int = TOP_N_LANGUAGES) -> pd.DataFrame:
    """Collapse the language long tail into 'Other'."""
    if "language" not in df.columns:
        return df
    df = df.copy()
    lang = df["language"].fillna("Unknown").replace("", "Unknown")
    top = lang.value_counts().head(n).index.tolist()
    df["language_top"] = lang.where(lang.isin(top), other="Other")
    return df


def load_tables(data_dir: str = None, deduplicate: bool = True) -> AnalysisTables:
    """Load analysis tables from data directory."""
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    universe = _read_parquet_optional(data_dir / "universe.parquet")
    internal_merges = _read_parquet_optional(data_dir / "internal_merges.parquet")
    classified_chunks = _read_parquet_optional(data_dir / "classified_chunks.parquet")
    resolved_chunks = _read_parquet_optional(data_dir / "resolved_chunks.parquet")
    extraction_errors = _read_parquet_optional(data_dir / "extraction_errors.parquet")

    if deduplicate:
        internal_merges = _dedup_on(internal_merges, _MERGE_KEY)
        classified_chunks = _dedup_on(classified_chunks, _CHUNK_KEY)

    tables = AnalysisTables(
        universe=universe,
        internal_merges=internal_merges,
        classified_chunks=classified_chunks,
        resolved_chunks=resolved_chunks,
        extraction_errors=extraction_errors,
    )

    # Add resolver_labels attribute
    resolver_labels = _read_parquet_optional(data_dir / "resolver_labels.parquet")
    tables.resolver_labels = resolver_labels

    return tables


def build_chunk_frame(tables: AnalysisTables) -> pd.DataFrame:
    """Chunk-level analysis frame with strategy and context."""
    chunks = tables.classified_chunks
    if chunks.empty:
        return chunks

    chunks = chunks.copy()
    # Canonicalize strategy labels
    if "strategy" in chunks.columns:
        chunks["strategy_raw"] = chunks["strategy"]

    # Join with PR context
    pctx = _pr_context(tables.universe)
    if not pctx.empty:
        chunks = chunks.merge(pctx, on="pr_id", how="left")
        chunks = _apply_language_topn(chunks)

    return chunks


def build_merge_frame(tables: AnalysisTables) -> pd.DataFrame:
    """Merge-level analysis frame with chunk counts and context."""
    merges = tables.internal_merges
    if merges.empty:
        return merges

    merges = _dedup_on(merges, _MERGE_KEY).copy()

    # Count chunks per merge
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

    # Join with PR context
    pctx = _pr_context(tables.universe)
    if not pctx.empty:
        merges = merges.merge(pctx, on="pr_id", how="left")
        merges = _apply_language_topn(merges)

    return merges


def stratum_order(df: pd.DataFrame, axis: str) -> list[str]:
    """Get ordering of strata by frequency."""
    col = STRATA.get(axis, axis)
    if col not in df.columns:
        return []
    values = df[col].fillna("Unknown").astype(str)
    return values.value_counts().index.tolist()


def stratify(df: pd.DataFrame, axis: str) -> Iterator[tuple[str, pd.DataFrame]]:
    """Iterate over (stratum_value, sub_df) pairs."""
    col = STRATA.get(axis, axis)
    if col not in df.columns:
        return iter([])
    values = df[col].fillna("Unknown").astype(str)
    order = values.value_counts().index.tolist()
    for value in order:
        yield value, df[values == value]


def setup_style() -> None:
    """Configure matplotlib/seaborn for publication figures."""
    if not sns or not plt:
        return
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
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
    })


def save_fig(fig: plt.Figure, name: str, directory: Path | str = None) -> tuple[Path, Path]:
    """Save figure as both PDF and PNG."""
    if directory is None:
        directory = FIGURES_DIR
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    pdf_path = directory / f"{name}.pdf"
    png_path = directory / f"{name}.png"

    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path, png_path


__all__ = [
    'AnalysisTables',
    'load_tables',
    'build_chunk_frame',
    'build_merge_frame',
    'stratum_order',
    'stratify',
    'setup_style',
    'save_fig',
    'STRATEGY_ORDER',
    'STRATEGY_PALETTE',
    'RESOLVER_ORDER',
    'RESOLVER_PALETTE',
    'TOP_N_LANGUAGES',
    'STRATA',
    'DATA_DIR',
    'FIGURES_DIR',
]
