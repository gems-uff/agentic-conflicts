"""Shared analysis utilities and data loading for the replication package."""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


class AnalysisTables:
    """Container for loaded analysis data and common operations."""

    def __init__(self, data_dir: str):
        """
        Initialize with data directory.

        Args:
            data_dir: Path to directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        self.tables = self._load_tables()

    def _load_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all parquet files from data directory."""
        tables = {}

        files = {
            'universe': 'universe.parquet',
            'internal_merges': 'internal_merges.parquet',
            'conflict_chunks': 'conflict_chunks.parquet',
            'classified_chunks': 'classified_chunks.parquet',
            'resolver_labels': 'resolver_labels.parquet',
        }

        for name, filename in files.items():
            path = self.data_dir / filename
            if path.exists():
                tables[name] = pd.read_parquet(path)

        return tables

    def get(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get a table by name."""
        return self.tables.get(table_name)

    @property
    def universe(self) -> Optional[pd.DataFrame]:
        """Agent PR metadata."""
        return self.tables.get('universe')

    @property
    def internal_merges(self) -> Optional[pd.DataFrame]:
        """Internal merge commits."""
        return self.tables.get('internal_merges')

    @property
    def conflict_chunks(self) -> Optional[pd.DataFrame]:
        """Conflict chunks (all)."""
        return self.tables.get('conflict_chunks')

    @property
    def classified_chunks(self) -> Optional[pd.DataFrame]:
        """Conflict chunks with strategy classification."""
        return self.tables.get('classified_chunks')

    @property
    def resolver_labels(self) -> Optional[pd.DataFrame]:
        """Resolver attribution labels."""
        return self.tables.get('resolver_labels')

    def merge_all(self) -> pd.DataFrame:
        """
        Merge all tables for comprehensive analysis.

        Returns:
            Single DataFrame with all columns
        """
        result = self.classified_chunks.copy()

        if self.resolver_labels is not None:
            result = result.merge(
                self.resolver_labels[['chunk_id', 'resolver', 'resolver_type']],
                on='chunk_id',
                how='left'
            )

        return result


def load_tables(data_dir: str) -> AnalysisTables:
    """
    Convenience function to load analysis tables.

    Args:
        data_dir: Path to data directory

    Returns:
        AnalysisTables object
    """
    return AnalysisTables(data_dir)


__all__ = [
    'AnalysisTables',
    'load_tables',
]
