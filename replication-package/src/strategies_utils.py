"""Conflict resolution strategy classification utilities.

Classifies merge conflict resolutions using multiset-based analysis to detect
strategy patterns: V1, V2, CC (concatenate clean), CB (concatenate with edits),
NC (new code combining both), NN (new unrelated code), Imprecise, Postponed.
"""

from typing import Optional, List
from multiset import Multiset


def normalize_line(line: str) -> str:
    """Normalize a line by removing whitespace."""
    return line.replace(" ", "").replace("\t", "").replace("\n", "")


def normalize_lines(lines: List[str]) -> List[str]:
    """Normalize a list of lines."""
    return [normalize_line(line) for line in lines]


def remove_empty_lines(lines: List[str]) -> List[str]:
    """Remove empty lines from a list."""
    return [line for line in lines if line.strip()]


def identify_resolution(
    v1: str,
    v2: str,
    resolution: Optional[str],
) -> str:
    """Classify conflict resolution strategy using multiset analysis.

    Args:
        v1: Version 1 (ours) content
        v2: Version 2 (theirs) content
        resolution: Resolved content (None means Imprecise)

    Returns:
        Strategy string: V1, V2, CC, CB, NC, NN, Imprecise, or Postponed
    """
    if resolution is None:
        return "Imprecise"

    v1_lines = normalize_lines(remove_empty_lines(v1.splitlines()))
    v2_lines = normalize_lines(remove_empty_lines(v2.splitlines()))
    resolution_lines = normalize_lines(remove_empty_lines(resolution.splitlines()))

    if not resolution_lines:
        return "None"

    if '<<<<<<<' in resolution or '=======' in resolution or '>>>>>>>' in resolution:
        return "Postponed"

    if normalize_line(v1) == normalize_line(resolution):
        return "V1"
    elif normalize_line(v2) == normalize_line(resolution):
        return "V2"
    elif normalize_line(resolution) == normalize_line(v1) + normalize_line(v2):
        return "CC"
    elif normalize_line(resolution) == normalize_line(v2) + normalize_line(v1):
        return "CC"
    else:
        v1_ms = Multiset(v1_lines)
        v2_ms = Multiset(v2_lines)
        resolution_ms = Multiset(resolution_lines)

        new_code = resolution_ms - (v1_ms + v2_ms)
        if len(new_code) > 0:
            if v1_lines and v2_lines:
                return "NC"
            else:
                return "NN"

        if len(resolution_lines) == 0:
            return "None"

        return "CB"


_RAW_TO_CANONICAL = {
    "V1": "V1",
    "V2": "V2",
    "ConcatV1V2": "CC",
    "ConcatV2V1": "CC",
    "New code": "NC",
    "Combination": "CB",
    "None": "NN",
    "CC": "CC",
    "CB": "CB",
    "NC": "NC",
    "NN": "NN",
    "Imprecise": "Imprecise",
    "Postponed": "Postponed",
}

STRATEGY_ORDER = ["V1", "V2", "CC", "CB", "NC", "NN", "Imprecise"]


def canonicalize_strategy(raw_strategy: str) -> str:
    """Convert raw strategy name to canonical form."""
    return _RAW_TO_CANONICAL.get(raw_strategy, "Imprecise")


__all__ = [
    "normalize_line",
    "normalize_lines",
    "remove_empty_lines",
    "identify_resolution",
    "canonicalize_strategy",
    "STRATEGY_ORDER",
    "_RAW_TO_CANONICAL",
]
