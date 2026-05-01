"""Figure generation utilities for the replication package.

Provides functions to generate publication-quality figures for:
- RQ1: Resolver attribution
- RQ2: Resolution strategies
- Dataset overview
"""

import matplotlib.pyplot as plt
import seaborn as sns
from .common import AnalysisTables
import pandas as pd


def setup_style():
    """Configure matplotlib/seaborn style for publication."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11


def plot_resolver_by_agent(tables: AnalysisTables, output_path: str = 'rq1_resolver_by_agent.png'):
    """
    Create figure: RQ1 Resolver attribution by agent.

    Args:
        tables: AnalysisTables object
        output_path: Where to save figure
    """
    setup_style()

    resolver = tables.resolver_labels
    if resolver is None or 'agent_name' not in resolver.columns:
        print("ERROR: Cannot plot resolver by agent")
        return

    # Compute per-agent rates
    per_agent = resolver.groupby('agent_name')['resolver'].apply(
        lambda x: (x == 'agent').sum() / len(x)
    ).sort_values(ascending=False)

    # Create figure
    fig, ax = plt.subplots()
    per_agent.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Self-Resolution Rate by Agent')
    ax.set_xlabel('Agent')
    ax.set_ylabel('Self-Resolution Rate')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_strategies_by_agent(tables: AnalysisTables, output_path: str = 'rq2_strategies_by_agent.png'):
    """
    Create figure: RQ2 Strategy distribution by agent.

    Args:
        tables: AnalysisTables object
        output_path: Where to save figure
    """
    setup_style()

    classified = tables.classified_chunks
    if classified is None or 'agent_name' not in classified.columns:
        print("ERROR: Cannot plot strategies by agent")
        return

    # Compute per-agent strategy distribution
    agent_strat = classified.groupby(['agent_name', 'strategy']).size().unstack(fill_value=0)

    # Create stacked bar chart
    fig, ax = plt.subplots()
    agent_strat.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Resolution Strategy Distribution by Agent')
    ax.set_xlabel('Agent')
    ax.set_ylabel('Number of Conflicts')
    ax.legend(title='Strategy')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_all_figures(tables: AnalysisTables, output_dir: str = './results'):
    """
    Generate all publication figures.

    Args:
        tables: AnalysisTables object
        output_dir: Where to save figures
    """
    print("Generating figures...")
    plot_resolver_by_agent(tables, f'{output_dir}/rq1_resolver_by_agent.png')
    plot_strategies_by_agent(tables, f'{output_dir}/rq2_strategies_by_agent.png')
    print("Figure generation complete")


__all__ = [
    'setup_style',
    'plot_resolver_by_agent',
    'plot_strategies_by_agent',
    'generate_all_figures',
]
