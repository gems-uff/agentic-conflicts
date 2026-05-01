"""Dataset characterization analysis.

Generates summary statistics for the study dataset:
- Number of repositories, agents, conflicts
- Language distribution
- Conflict distribution by agent
"""

import pandas as pd
from .common import AnalysisTables


def analyze_dataset(tables: AnalysisTables, output_dir: str = './results'):
    """
    Analyze and characterize the dataset.

    Args:
        tables: AnalysisTables object
        output_dir: Where to save results
    """
    print("Dataset Characterization")
    print("=" * 50)

    universe = tables.universe
    internal_merges = tables.internal_merges
    conflict_chunks = tables.conflict_chunks

    if universe is None:
        print("ERROR: universe table not found")
        return

    # Basic counts
    num_repos = len(universe['repo'].unique())
    num_agents = len(universe['agent_name'].unique())
    num_merges = len(internal_merges) if internal_merges is not None else 0
    num_conflicts = len(conflict_chunks) if conflict_chunks is not None else 0

    print(f"\nRepositories: {num_repos:,}")
    print(f"AI Agents: {num_agents}")
    print(f"Internal Merge Commits: {num_merges:,}")
    print(f"Conflict Chunks: {num_conflicts:,}")

    # Agent distribution
    if universe is not None:
        print("\nConflicts by Agent:")
        agent_dist = universe['agent_name'].value_counts()
        for agent, count in agent_dist.items():
            print(f"  {agent}: {count:,}")

    # Language distribution
    if universe is not None and 'language' in universe.columns:
        print("\nTop Languages by Conflict Count:")
        lang_dist = universe['language'].value_counts().head(10)
        for lang, count in lang_dist.items():
            print(f"  {lang}: {count:,}")

    print("\n" + "=" * 50)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './data'

    tables = AnalysisTables(data_dir)
    analyze_dataset(tables, output_dir=f'{data_dir}/results')
