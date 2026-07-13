"""Statistical utilities for hypothesis testing and effect size computation.

Implements confidence intervals, statistical tests, and effect sizes following
paper methodology (Section 3.5 "Baseline Calibration").
"""

import logging
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint


def wilson_ci(
    successes: int, n: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute 95% Wilson score confidence interval for proportion.

    Args:
        successes: Number of successes
        n: Total count
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n == 0:
        return (0.0, 0.0)
    lower, upper = proportion_confint(successes, n, alpha, method='wilson')
    return (lower, upper)


def cramers_v_bias_corrected(contingency_table: np.ndarray) -> float:
    """Compute bias-corrected Cramér's V effect size from contingency table.

    Cramér's V is a measure of association between two nominal variables.
    Ranges from 0 (no association) to 1 (perfect association).

    Args:
        contingency_table: 2D contingency table (numpy array or pd.crosstab result)

    Returns:
        Cramér's V value
    """
    if isinstance(contingency_table, pd.DataFrame):
        contingency_table = contingency_table.values

    # Chi-squared test
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum()

    if n == 0:
        return 0.0

    # Cramér's V
    min_dim = min(contingency_table.shape) - 1
    if min_dim <= 0:
        return 0.0

    v = np.sqrt(chi2 / (n * min_dim))

    # Bias correction (following paper methodology)
    # Reduced bias Cramér's V for small samples
    k = contingency_table.shape[0]  # number of rows
    r = contingency_table.shape[1]  # number of columns
    phi_2 = chi2 / n
    phi_2_corrected = max(0, phi_2 - ((k - 1) * (r - 1)) / (n - 1))
    v_corrected = np.sqrt(phi_2_corrected / min_dim) if min_dim > 0 else 0.0

    return v_corrected


def chi2_test_with_cramers(
    df: pd.DataFrame, var1: str, var2: str
) -> Dict[str, float]:
    """Perform chi-squared test and compute Cramér's V.

    Args:
        df: DataFrame with variables
        var1: Column name for first variable
        var2: Column name for second variable

    Returns:
        Dict with keys: chi2, p_value, cramers_v, dof
    """
    contingency = pd.crosstab(df[var1], df[var2])
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    v = cramers_v_bias_corrected(contingency.values)

    return {
        'chi2': chi2,
        'p_value': p_val,
        'cramers_v': v,
        'dof': dof,
    }


def cliffs_delta(group1: pd.Series, group2: pd.Series) -> float:
    """Compute Cliff's δ (non-parametric effect size for ordinal data).

    Ranges from -1 (all group1 < group2) to +1 (all group1 > group2).
    Values near 0 indicate no difference.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cliff's δ value
    """
    group1 = group1.dropna().values
    group2 = group2.dropna().values

    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    # Count ordinal comparisons
    greater = sum(1 for g1 in group1 for g2 in group2 if g1 > g2)
    less = sum(1 for g1 in group1 for g2 in group2 if g1 < g2)

    delta = (greater - less) / (n1 * n2)
    return delta


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """Apply Bonferroni correction to multiple p-values.

    Args:
        p_values: List of p-values
        alpha: Significance level (default 0.05)

    Returns:
        Tuple of (corrected_p_values, corrected_alpha)
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    corrected_p_values = np.minimum(p_values * n_tests, 1.0)
    return corrected_p_values, corrected_alpha


def sensitivity_analysis_imprecise(
    chunks_df: pd.DataFrame, strategy_mapping: Dict[str, str]
) -> pd.DataFrame:
    """Reassign Imprecise chunks according to strategy mapping.

    Used to test robustness of findings to Imprecise categorization.
    Maps all Imprecise chunks to alternative strategies (e.g., V1 or NC).

    Args:
        chunks_df: DataFrame with 'strategy' column
        strategy_mapping: Dict mapping 'Imprecise' -> alternative strategy
                         (e.g., {'Imprecise': 'V1'} for upper bound)

    Returns:
        DataFrame with reassigned strategies
    """
    adjusted_df = chunks_df.copy()
    for source_strategy, target_strategy in strategy_mapping.items():
        adjusted_df.loc[adjusted_df['strategy'] == source_strategy, 'strategy'] = target_strategy
    return adjusted_df


def compute_ci_per_group(
    df: pd.DataFrame,
    group_col: str,
    success_col: str,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Compute Wilson CIs for proportions in each group.

    Args:
        df: Input DataFrame
        group_col: Column to group by
        success_col: Boolean/binary column indicating success
        alpha: Significance level

    Returns:
        DataFrame with CI bounds per group
    """
    results = []
    for group_name, group_data in df.groupby(group_col):
        successes = group_data[success_col].sum()
        n = len(group_data)
        lower, upper = wilson_ci(successes, n, alpha)
        results.append({
            'group': group_name,
            'successes': successes,
            'total': n,
            'proportion': successes / n if n > 0 else 0,
            'ci_lower': lower,
            'ci_upper': upper,
        })
    return pd.DataFrame(results)


def compute_pairwise_cramers_v(
    df: pd.DataFrame,
    var: str,
    groups: List[str],
    alpha: float = 0.05
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Compute pairwise Cramér's V comparisons between groups.

    Tests association between categorical variable and group membership.

    Args:
        df: Input DataFrame
        var: Categorical variable column name
        groups: List of group names to compare
        alpha: Significance level for Bonferroni correction

    Returns:
        Dict mapping (group1, group2) -> {cramers_v, p_value, p_corrected}
    """
    results = {}
    n_pairs = len(groups) * (len(groups) - 1) / 2
    bonferroni_alpha = alpha / n_pairs

    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            # Subset data for this pair
            pair_df = df[df['group'].isin([g1, g2])].copy()
            pair_df['is_group1'] = pair_df['group'] == g1

            # Chi-squared test
            test_result = chi2_test_with_cramers(pair_df, 'is_group1', var)

            results[(g1, g2)] = {
                'cramers_v': test_result['cramers_v'],
                'p_value': test_result['p_value'],
                'p_corrected': min(test_result['p_value'] * n_pairs, 1.0),
                'significant': test_result['p_value'] < bonferroni_alpha,
            }

    return results


__all__ = [
    'wilson_ci',
    'cramers_v_bias_corrected',
    'chi2_test_with_cramers',
    'cliffs_delta',
    'bonferroni_correction',
    'sensitivity_analysis_imprecise',
    'compute_ci_per_group',
    'compute_pairwise_cramers_v',
]
