"""Detailed RQ1 analysis: Who resolves merge conflicts?

Provides comprehensive resolver attribution analysis with:
- Scope per agent (PRs, merges, conflict rates)
- Resolver type distribution (agent, agent-assisted, human)
- Confidence intervals and statistical tests
"""

import logging
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
from .common import AnalysisTables, build_merge_frame
from .statistical_utils import (
    wilson_ci,
    chi2_test_with_cramers,
    compute_ci_per_group,
)


def compute_scope_per_agent(tables: AnalysisTables) -> pd.DataFrame:
    """Compute PR scope and conflict rate per PR-authoring agent.

    Returns:
        DataFrame with columns:
        - agent: PR-authoring agent
        - n_prs: Number of PRs
        - n_merges: Number of internal merge commits
        - n_conflicting_merges: Number of merges with conflicts
        - conflict_rate: Proportion of merges with conflicts
        - conflict_rate_ci_lower: 95% CI lower bound
        - conflict_rate_ci_upper: 95% CI upper bound
    """
    if tables.internal_merges.empty or tables.universe.empty:
        return pd.DataFrame()

    # Group universe by pr_id and agent
    prs = (
        tables.universe[['pr_id', 'agent']]
        .drop_duplicates(subset=['pr_id'])
        .dropna(subset=['agent'])
    )
    n_prs = prs.groupby('agent').size().reset_index(name='n_prs')

    # Count merges per agent
    merges = tables.internal_merges.copy()
    if 'agent' not in merges.columns:
        merges = merges.merge(
            prs[['pr_id', 'agent']],
            on='pr_id',
            how='left'
        )

    merges = merges.dropna(subset=['agent'])

    # All merges per agent
    n_merges = merges.groupby('agent').size().reset_index(name='n_merges')

    # Conflicting merges per agent
    if not tables.classified_chunks.empty:
        conflicts = (
            tables.classified_chunks[['pr_id', 'repo_full_name', 'merge_sha']]
            .drop_duplicates(subset=['repo_full_name', 'merge_sha'])
        )
        conflicts = conflicts.merge(
            prs[['pr_id', 'agent']],
            on='pr_id',
            how='left'
        )
        conflicts = conflicts.dropna(subset=['agent'])
        n_conflicting = conflicts.groupby('agent').size().reset_index(
            name='n_conflicting_merges'
        )
    else:
        n_conflicting = pd.DataFrame(
            {'agent': merges['agent'].unique(), 'n_conflicting_merges': 0}
        )

    # Merge and compute rates with CIs
    result = n_prs.merge(n_merges, on='agent', how='outer')
    result = result.merge(n_conflicting, on='agent', how='outer')
    result = result.fillna(0).astype({'n_prs': int, 'n_merges': int, 'n_conflicting_merges': int})

    # Compute conflict rate with CI
    cis = []
    for _, row in result.iterrows():
        n = row['n_merges']
        successes = row['n_conflicting_merges']
        lower, upper = wilson_ci(int(successes), int(n))
        rate = successes / n if n > 0 else 0
        cis.append({
            'agent': row['agent'],
            'conflict_rate': rate,
            'conflict_rate_ci_lower': lower,
            'conflict_rate_ci_upper': upper,
        })
    ci_df = pd.DataFrame(cis)

    result = result.merge(ci_df, on='agent')
    return result.sort_values('n_merges', ascending=False)


def compute_resolver_by_agent(tables: AnalysisTables) -> Tuple[pd.DataFrame, Dict]:
    """Compute resolver type distribution per agent with statistics.

    Returns:
        Tuple of:
        - DataFrame with resolver counts per agent
        - Dict with chi-squared test results
    """
    if (
        tables.internal_merges.empty
        or tables.resolver_labels.empty
        or tables.universe.empty
    ):
        return pd.DataFrame(), {}

    # Get resolver information
    resolvers = tables.resolver_labels.copy()
    if 'agent' not in resolvers.columns:
        # Join with universe to get agent
        prs = (
            tables.universe[['pr_id', 'agent']]
            .drop_duplicates(subset=['pr_id'])
        )
        resolvers = resolvers.merge(prs, on='pr_id', how='left')

    resolvers = resolvers.dropna(subset=['agent'])

    # Create contingency table
    if 'resolver_type' not in resolvers.columns:
        # Fallback: use binary agent/human classification
        resolvers['resolver_type'] = 'human'
        agent_mask = resolvers['agent'].str.lower().str.contains(
            'agent|bot|claude|copilot|cursor|devin|openai',
            na=False
        )
        resolvers.loc[agent_mask, 'resolver_type'] = 'agent'

    # Crosstab: agent × resolver_type
    crosstab = pd.crosstab(
        resolvers['agent'],
        resolvers['resolver_type'],
        margins=False
    )
    crosstab = crosstab.fillna(0).astype(int)
    crosstab['total'] = crosstab.sum(axis=1)

    # Compute proportions and CIs
    results = []
    for agent in crosstab.index:
        total = crosstab.loc[agent, 'total']
        for resolver_type in crosstab.columns:
            if resolver_type == 'total':
                continue
            count = crosstab.loc[agent, resolver_type]
            lower, upper = wilson_ci(int(count), int(total))
            prop = count / total if total > 0 else 0
            results.append({
                'agent': agent,
                'resolver_type': resolver_type,
                'count': int(count),
                'proportion': prop,
                'ci_lower': lower,
                'ci_upper': upper,
                'total': int(total),
            })
    result_df = pd.DataFrame(results)

    # Chi-squared test: agent × resolver_type
    test_result = {}
    if not crosstab.empty and crosstab.shape[0] > 1:
        # Remove 'total' column for test
        test_data = crosstab.drop('total', axis=1)
        chi2_test = chi2_test_with_cramers(
            resolvers,
            'agent',
            'resolver_type'
        )
        test_result = {
            'chi2': chi2_test['chi2'],
            'p_value': chi2_test['p_value'],
            'cramers_v': chi2_test['cramers_v'],
            'dof': chi2_test['dof'],
        }

    return result_df, test_result


def analyze_rq1_detailed(
    tables: AnalysisTables,
    output_dir: str = None
) -> None:
    """Run comprehensive RQ1 analysis and export results.

    Args:
        tables: AnalysisTables object with all analysis data
        output_dir: Directory to save results (default: data/results)
    """
    import logging
    if output_dir is None:
        output_dir = 'results'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("RQ1 Detailed Analysis: Who Resolves?")

    # 1. Scope per agent
    logging.info("  Computing scope per agent...")
    scope_df = compute_scope_per_agent(tables)
    if not scope_df.empty:
        scope_df.to_csv(output_dir / 'rq1_scope_per_agent.csv', index=False)
        logging.info(f"    Exported: rq1_scope_per_agent.csv ({len(scope_df)} agents)")

    # 2. Resolver by agent
    logging.info("  Computing resolver attribution...")
    resolver_df, test_result = compute_resolver_by_agent(tables)
    if not resolver_df.empty:
        resolver_df.to_csv(output_dir / 'rq1_resolver_by_agent.csv', index=False)
        logging.info(f"    Exported: rq1_resolver_by_agent.csv ({len(resolver_df)} rows)")

        if test_result:
            test_txt = (
                f"Chi-squared Test: Agent vs Resolver Type\n"
                f"Chi-squared: {test_result['chi2']:.4f}\n"
                f"p-value: {test_result['p_value']:.6f}\n"
                f"Cramér's V (effect size): {test_result['cramers_v']:.4f}\n"
                f"Degrees of freedom: {test_result['dof']}\n"
            )
            with open(output_dir / 'rq1_chi_squared_test.txt', 'w') as f:
                f.write(test_txt)
            logging.info(f"    Exported: rq1_chi_squared_test.txt")

    logging.info("RQ1 Detailed Analysis complete")


__all__ = [
    'compute_scope_per_agent',
    'compute_resolver_by_agent',
    'analyze_rq1_detailed',
]
