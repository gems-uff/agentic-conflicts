"""Conflict resolution strategy classification utilities.

Classifies merge conflict resolutions into 6 strategy categories:
- V1: Take version 1 (ours)
- V2: Take version 2 (theirs)
- CC: Concatenate both versions
- CB: Concatenate both with manual edits
- NC: New code combining both
- NN: New code, unrelated to conflict
"""

import re
from typing import Optional, Dict, List, Tuple


def classify_strategy(
    version1: str,
    version2: str,
    resolution: str,
    base: Optional[str] = None
) -> str:
    """
    Classify a conflict resolution into one of 6 strategies.

    Args:
        version1: Content from version 1 (ours)
        version2: Content from version 2 (theirs)
        resolution: The resolved content
        base: Original base version (optional, for context)

    Returns:
        Strategy label: 'V1', 'V2', 'CC', 'CB', 'NC', or 'NN'
    """
    # Normalize for comparison
    v1_norm = version1.strip()
    v2_norm = version2.strip()
    res_norm = resolution.strip()

    # Check for exact matches (V1, V2)
    if res_norm == v1_norm:
        return 'V1'
    if res_norm == v2_norm:
        return 'V2'

    # Check for concatenation (CC, CB)
    if v1_norm in res_norm and v2_norm in res_norm:
        # Both versions appear in resolution
        if (res_norm == v1_norm + v2_norm or
            res_norm == v2_norm + v1_norm or
            res_norm == v1_norm + '\n' + v2_norm or
            res_norm == v2_norm + '\n' + v1_norm):
            return 'CC'
        else:
            # Concatenation with modifications
            return 'CB'

    # Check if resolution contains both versions with modifications (NC)
    if v1_norm in res_norm and v2_norm in res_norm:
        return 'NC'

    # Check if only v1 or v2 is present in modified form (NC)
    if v1_norm in res_norm or v2_norm in res_norm:
        return 'NC'

    # Neither version is clearly in resolution (NN)
    return 'NN'


def is_imprecise(
    resolution: str,
    min_lines: int = 1,
    max_conflict_markers: int = 0
) -> bool:
    """
    Determine if a resolution is imprecise (contains unresolved conflict markers).

    Args:
        resolution: The resolved content
        min_lines: Minimum lines to consider valid
        max_conflict_markers: Maximum unresolved markers allowed

    Returns:
        True if resolution appears imprecise
    """
    # Check for conflict markers
    markers = (
        resolution.count('<<<<<<<') +
        resolution.count('=======') +
        resolution.count('>>>>>>>')
    )

    if markers > max_conflict_markers:
        return True

    # Check minimum content
    if len(resolution.strip().split('\n')) < min_lines:
        return True

    return False


__all__ = [
    'classify_strategy',
    'is_imprecise',
]
