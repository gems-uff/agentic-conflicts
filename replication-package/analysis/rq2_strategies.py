"""RQ2 Analysis: How Do Agents Resolve Conflicts?

Analyzes resolution strategy distributions:
- Overall strategy distribution
- Per-agent strategy preferences
- Heterogeneity analysis (V statistic)
- Patterns by language and task type
"""

import pandas as pd
from .common import AnalysisTables


def analyze_rq2(tables: AnalysisTables, output_dir: str = './results'):
    """
    Analyze RQ2: How do agents resolve merge conflicts?

    Args:
        tables: AnalysisTables object
        output_dir: Where to save results
    """
    print("\nRQ2: How Do Agents Resolve Merge Conflicts?")
    print("=" * 50)

    classified = tables.classified_chunks

    if classified is None:
        print("ERROR: classified_chunks table not found")
        return

    # Overall strategy distribution
    print(f"\nOverall Strategy Distribution:")
    strategy_dist = classified['strategy'].value_counts()
    total = len(classified)

    for strategy, count in strategy_dist.items():
        pct = count / total * 100
        print(f"  {strategy}: {count:,} ({pct:.1f}%)")

    # Per-agent strategy distribution
    if 'agent_name' in classified.columns:
        print(f"\nStrategy Distribution by Agent:")

        agent_strategies = classified.groupby(['agent_name', 'strategy']).size().unstack(fill_value=0)
        agent_pcts = agent_strategies.div(agent_strategies.sum(axis=1), axis=0) * 100

        for agent in agent_pcts.index:
            print(f"\n  {agent}:")
            for strategy in agent_pcts.columns:
                pct = agent_pcts.loc[agent, strategy]
                count = agent_strategies.loc[agent, strategy]
                print(f"    {strategy}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 50)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './data'

    tables = AnalysisTables(data_dir)
    analyze_rq2(tables, output_dir=f'{data_dir}/results')
