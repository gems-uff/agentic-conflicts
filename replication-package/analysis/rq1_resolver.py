"""RQ1 Analysis: Who Resolves Merge Conflicts?

Determines resolver attribution (agent vs. human) and analyzes:
- Overall human vs. agent resolution rates
- Per-agent self-resolution rates
- Distribution by language and task type
"""

import pandas as pd
from .common import AnalysisTables


def analyze_rq1(tables: AnalysisTables, output_dir: str = './results'):
    """
    Analyze RQ1: Who resolves merge conflicts?

    Args:
        tables: AnalysisTables object
        output_dir: Where to save results
    """
    print("\nRQ1: Who Resolves Merge Conflicts?")
    print("=" * 50)

    resolver_labels = tables.resolver_labels

    if resolver_labels is None:
        print("ERROR: resolver_labels table not found")
        return

    # Overall resolution rate
    total = len(resolver_labels)
    agent_resolved = (resolver_labels['resolver'] == 'agent').sum()
    human_resolved = (resolver_labels['resolver'] == 'human').sum()

    agent_rate = agent_resolved / total if total > 0 else 0
    human_rate = human_resolved / total if total > 0 else 0

    print(f"\nOverall Resolution Attribution:")
    print(f"  Total Conflicts: {total:,}")
    print(f"  Agent-Resolved: {agent_resolved:,} ({agent_rate:.1%})")
    print(f"  Human-Resolved: {human_resolved:,} ({human_rate:.1%})")

    # Per-agent self-resolution rates
    if 'agent_name' in resolver_labels.columns:
        print(f"\nSelf-Resolution Rates by Agent:")
        per_agent = resolver_labels.groupby('agent_name').apply(
            lambda g: (g['resolver'] == 'agent').sum() / len(g)
        ).sort_values(ascending=False)

        for agent, rate in per_agent.items():
            count = (resolver_labels[resolver_labels['agent_name'] == agent]['resolver'] == 'agent').sum()
            total_agent = len(resolver_labels[resolver_labels['agent_name'] == agent])
            print(f"  {agent}: {rate:.1%} ({count}/{total_agent})")

    print("\n" + "=" * 50)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './data'

    tables = AnalysisTables(data_dir)
    analyze_rq1(tables, output_dir=f'{data_dir}/results')
