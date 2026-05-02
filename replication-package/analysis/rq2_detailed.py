"""Detailed RQ2 analysis: How do they resolve conflicts?

Provides comprehensive resolution strategy analysis with:
- Per-agent strategy distribution
- Pairwise contrasts with effect sizes
- Repository concentration analysis
- File-category stratification
- Postponed sub-category analysis
- Sensitivity analysis for Imprecise chunks
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from .common import AnalysisTables, build_chunk_frame, STRATEGY_ORDER
from .statistical_utils import (
    chi2_test_with_cramers,
    cramers_v_bias_corrected,
    bonferroni_correction,
    sensitivity_analysis_imprecise,
)


def compute_strategy_per_agent(
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    """Compute strategy distribution per agent.

    Args:
        chunks: Classified chunks DataFrame with 'agent' and 'strategy' columns

    Returns:
        DataFrame with columns: agent, strategy, count, proportion, percentage
    """
    if chunks.empty or 'agent' not in chunks.columns or 'strategy' not in chunks.columns:
        return pd.DataFrame()

    chunks = chunks.dropna(subset=['agent', 'strategy'])

    # Crosstab
    crosstab = pd.crosstab(chunks['agent'], chunks['strategy'], margins=False)
    crosstab = crosstab.fillna(0).astype(int)

    # Compute proportions
    results = []
    for agent in crosstab.index:
        row_total = crosstab.loc[agent].sum()
        for strategy in crosstab.columns:
            count = crosstab.loc[agent, strategy]
            prop = count / row_total if row_total > 0 else 0
            results.append({
                'agent': agent,
                'strategy': strategy,
                'count': int(count),
                'proportion': prop,
                'percentage': prop * 100,
            })

    df = pd.DataFrame(results)
    # Order by strategy
    strategy_order = [s for s in STRATEGY_ORDER if s in df['strategy'].unique()]
    df['strategy'] = pd.Categorical(df['strategy'], categories=strategy_order, ordered=True)
    return df.sort_values(['agent', 'strategy'])


def compute_pairwise_contrasts(
    chunks: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Compute pairwise Cramér's V contrasts between agents.

    Args:
        chunks: Classified chunks DataFrame
        alpha: Significance level (default 0.05)

    Returns:
        DataFrame with pairwise contrasts
    """
    if chunks.empty or 'agent' not in chunks.columns or 'strategy' not in chunks.columns:
        return pd.DataFrame()

    chunks = chunks.dropna(subset=['agent', 'strategy'])
    agents = chunks['agent'].unique()

    if len(agents) < 2:
        return pd.DataFrame()

    results = []
    n_pairs = len(agents) * (len(agents) - 1) / 2
    bonferroni_alpha = alpha / n_pairs

    for i, agent1 in enumerate(agents):
        for agent2 in agents[i+1:]:
            # Subset data
            pair_chunks = chunks[chunks['agent'].isin([agent1, agent2])].copy()
            pair_chunks['is_agent1'] = pair_chunks['agent'] == agent1

            # Chi-squared test
            test_result = chi2_test_with_cramers(
                pair_chunks,
                'is_agent1',
                'strategy'
            )

            results.append({
                'agent1': agent1,
                'agent2': agent2,
                'cramers_v': test_result['cramers_v'],
                'chi2': test_result['chi2'],
                'p_value': test_result['p_value'],
                'p_bonferroni': min(test_result['p_value'] * n_pairs, 1.0),
                'significant': test_result['p_value'] < bonferroni_alpha,
                'n_chunks': len(pair_chunks),
            })

    return pd.DataFrame(results).sort_values('cramers_v', ascending=False)


def compute_repository_concentration(
    merges: pd.DataFrame,
) -> pd.DataFrame:
    """Compute top-1 and top-3 repository concentration per agent.

    Args:
        merges: Internal merges DataFrame

    Returns:
        DataFrame with top-1 and top-3 repository shares per agent
    """
    if merges.empty or 'agent' not in merges.columns or 'repo_full_name' not in merges.columns:
        return pd.DataFrame()

    merges = merges.dropna(subset=['agent', 'repo_full_name'])

    results = []
    for agent, agent_data in merges.groupby('agent'):
        repo_counts = agent_data['repo_full_name'].value_counts()
        total = repo_counts.sum()

        top1_share = repo_counts.iloc[0] / total if len(repo_counts) > 0 else 0
        top3_share = repo_counts.head(3).sum() / total if len(repo_counts) >= 3 else repo_counts.sum() / total

        results.append({
            'agent': agent,
            'n_merges': int(total),
            'n_unique_repos': len(repo_counts),
            'top1_repo': repo_counts.index[0] if len(repo_counts) > 0 else None,
            'top1_count': int(repo_counts.iloc[0]) if len(repo_counts) > 0 else 0,
            'top1_share': top1_share,
            'top3_share': top3_share,
            'top1_percentage': top1_share * 100,
            'top3_percentage': top3_share * 100,
        })

    return pd.DataFrame(results).sort_values('n_merges', ascending=False)


def analyze_file_categories(
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze strategy distribution stratified by file category.

    Args:
        chunks: Classified chunks with file_category column

    Returns:
        DataFrame with strategy distribution per (agent, file_category) pair
    """
    if (
        chunks.empty
        or 'file_category' not in chunks.columns
        or 'agent' not in chunks.columns
        or 'strategy' not in chunks.columns
    ):
        return pd.DataFrame()

    chunks = chunks.dropna(subset=['agent', 'file_category', 'strategy'])

    results = []
    for (agent, file_cat), group in chunks.groupby(['agent', 'file_category']):
        strategy_counts = group['strategy'].value_counts()
        total = len(group)
        for strategy in strategy_counts.index:
            count = strategy_counts[strategy]
            results.append({
                'agent': agent,
                'file_category': file_cat,
                'strategy': strategy,
                'count': int(count),
                'proportion': count / total,
                'percentage': (count / total) * 100,
            })

    return pd.DataFrame(results)


def analyze_postponed(
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Postponed rate per agent.

    Args:
        chunks: Classified chunks with 'strategy' column

    Returns:
        DataFrame with Postponed counts and rates per agent
    """
    if chunks.empty or 'agent' not in chunks.columns or 'strategy' not in chunks.columns:
        return pd.DataFrame()

    chunks = chunks.dropna(subset=['agent', 'strategy'])

    results = []
    for agent, agent_chunks in chunks.groupby('agent'):
        total = len(agent_chunks)
        postponed = (agent_chunks['strategy'] == 'Postponed').sum()
        rate = postponed / total if total > 0 else 0

        results.append({
            'agent': agent,
            'n_chunks': int(total),
            'n_postponed': int(postponed),
            'postponed_rate': rate,
            'postponed_percentage': rate * 100,
        })

    return pd.DataFrame(results).sort_values('n_chunks', ascending=False)


def analyze_sensitivity_imprecise(
    chunks: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Run sensitivity analysis: reassign Imprecise chunks to V1 or NC.

    Tests whether key findings (pairwise contrasts) survive when Imprecise
    chunks are reassigned to upper bound (V1) or lower bound (NC).

    Args:
        chunks: Classified chunks with 'strategy' column

    Returns:
        Dict with keys 'upper_bound' and 'lower_bound' containing results
    """
    if chunks.empty or 'strategy' not in chunks.columns:
        return {}

    results = {}

    # Upper bound: Imprecise -> V1
    upper_chunks = sensitivity_analysis_imprecise(
        chunks,
        {'Imprecise': 'V1'}
    )
    upper_pairs = compute_pairwise_contrasts(upper_chunks)
    results['upper_bound'] = upper_pairs

    # Lower bound: Imprecise -> NC
    lower_chunks = sensitivity_analysis_imprecise(
        chunks,
        {'Imprecise': 'NC'}
    )
    lower_pairs = compute_pairwise_contrasts(lower_chunks)
    results['lower_bound'] = lower_pairs

    return results


def analyze_rq2_detailed(
    tables: AnalysisTables,
    output_dir: str = None
) -> None:
    """Run comprehensive RQ2 analysis and export results.

    Args:
        tables: AnalysisTables object with all analysis data
        output_dir: Directory to save results (default: data/results)
    """
    if output_dir is None:
        output_dir = 'results'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("RQ2 Detailed Analysis: How Do They Resolve?")

    chunks = build_chunk_frame(tables)
    merges = tables.internal_merges

    # 1. Per-agent strategy distribution
    logging.info("  Computing per-agent strategy distribution...")
    strategy_df = compute_strategy_per_agent(chunks)
    if not strategy_df.empty:
        strategy_df.to_csv(output_dir / 'rq2_strategy_per_agent.csv', index=False)
        logging.info(f"    Exported: rq2_strategy_per_agent.csv ({len(strategy_df)} rows)")

    # 2. Pairwise contrasts
    logging.info("  Computing pairwise agent contrasts...")
    pairs_df = compute_pairwise_contrasts(chunks)
    if not pairs_df.empty:
        pairs_df.to_csv(output_dir / 'rq2_pairwise_contrasts.csv', index=False)
        logging.info(f"    Exported: rq2_pairwise_contrasts.csv ({len(pairs_df)} pairs)")

    # 3. Repository concentration
    logging.info("  Computing repository concentration...")
    repo_conc = compute_repository_concentration(merges)
    if not repo_conc.empty:
        repo_conc.to_csv(output_dir / 'rq2_repository_concentration.csv', index=False)
        logging.info(f"    Exported: rq2_repository_concentration.csv ({len(repo_conc)} agents)")

    # 4. File category analysis
    logging.info("  Analyzing file category stratification...")
    file_cat_df = analyze_file_categories(chunks)
    if not file_cat_df.empty:
        file_cat_df.to_csv(output_dir / 'rq2_file_category_analysis.csv', index=False)
        logging.info(f"    Exported: rq2_file_category_analysis.csv ({len(file_cat_df)} rows)")

    # 5. Postponed analysis
    logging.info("  Analyzing Postponed chunks...")
    postponed_df = analyze_postponed(chunks)
    if not postponed_df.empty:
        postponed_df.to_csv(output_dir / 'rq2_postponed_analysis.csv', index=False)
        logging.info(f"    Exported: rq2_postponed_analysis.csv ({len(postponed_df)} agents)")

    # 6. Sensitivity analysis
    logging.info("  Running sensitivity analysis (Imprecise reassignment)...")
    sensitivity_results = analyze_sensitivity_imprecise(chunks)
    if sensitivity_results:
        if 'upper_bound' in sensitivity_results:
            sensitivity_results['upper_bound'].to_csv(
                output_dir / 'rq2_sensitivity_upper_bound.csv',
                index=False
            )
            logging.info("    Exported: rq2_sensitivity_upper_bound.csv")
        if 'lower_bound' in sensitivity_results:
            sensitivity_results['lower_bound'].to_csv(
                output_dir / 'rq2_sensitivity_lower_bound.csv',
                index=False
            )
            logging.info("    Exported: rq2_sensitivity_lower_bound.csv")

    logging.info("RQ2 Detailed Analysis complete")


__all__ = [
    'compute_strategy_per_agent',
    'compute_pairwise_contrasts',
    'compute_repository_concentration',
    'analyze_file_categories',
    'analyze_postponed',
    'analyze_sensitivity_imprecise',
    'analyze_rq2_detailed',
]
