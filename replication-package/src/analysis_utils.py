"""Shared analysis utilities for merge conflict resolution study."""

import pandas as pd
from typing import Dict, List, Optional


def load_results(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all parquet result files from data directory.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary mapping file names to DataFrames
    """
    results = {}

    files_to_load = [
        'universe.parquet',
        'internal_merges.parquet',
        'conflict_chunks.parquet',
        'classified_chunks.parquet',
        'resolver_labels.parquet',
    ]

    for fname in files_to_load:
        path = f'{data_dir}/{fname}'
        try:
            results[fname.replace('.parquet', '')] = pd.read_parquet(path)
        except FileNotFoundError:
            pass

    return results


def compute_self_resolution_rate(
    resolver_labels: pd.DataFrame,
    by: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute self-resolution rate (conflicts resolved by agent).

    Args:
        resolver_labels: DataFrame with resolver information
        by: Column to group by (e.g., 'agent_name')

    Returns:
        Summary DataFrame with self-resolution rates
    """
    if by is None:
        total = len(resolver_labels)
        agent_resolved = (resolver_labels['resolver'] == 'agent').sum()
        return pd.DataFrame({
            'self_resolution_rate': [agent_resolved / total if total > 0 else 0]
        })

    return resolver_labels.groupby(by).apply(
        lambda g: (g['resolver'] == 'agent').sum() / len(g)
    ).reset_index(name='self_resolution_rate')


def compute_strategy_distribution(
    classified_chunks: pd.DataFrame,
    by: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute distribution of resolution strategies.

    Args:
        classified_chunks: DataFrame with strategy classifications
        by: Column to group by (e.g., 'agent_name')

    Returns:
        Summary DataFrame with strategy counts
    """
    if by is None:
        return classified_chunks['strategy'].value_counts().to_frame('count')

    return classified_chunks.groupby([by, 'strategy']).size().unstack(fill_value=0)


__all__ = [
    'load_results',
    'compute_self_resolution_rate',
    'compute_strategy_distribution',
]
