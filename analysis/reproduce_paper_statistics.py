#!/usr/bin/env python
"""
reproduce_paper_statistics.py
=============================
Reproduces ALL statistical analyses described in main.tex with sanity checks.

This script:
  1. Loads data from Parquet tables (produced by the extraction pipeline)
  2. Performs sanity checks on chunk/merge counts against main.tex expectations
  3. Runs RQ1 analyses (resolver attribution) with Wilson CIs and chi-squared tests
  4. Runs RQ2 analyses (resolution strategy heterogeneity) with per-agent contrasts
  5. Outputs results to JSON and human-readable text files

Expected data sizes from main.tex (Section 4.1, Dataset Overview):
  - 50,700 distinct internal merge commits (after merge-level deduplication)
  - 14,960 conflicting merges (29.5% of merges)
  - 121,599 conflict chunks
  - 83,043 chunks with localized resolution (68.3%)
  - 7,291 Postponed chunks (6.0%)
  - 75,752 resolved (classifiable) chunks (62.3%)
  - 45,847 Imprecise chunks (37.7%)

RQ1 findings from main.tex:
  - 14,381 human-resolved (96.1%), 462 agent-resolved (3.1%), 117 agent-assisted (0.8%)
  - Per-agent self-resolution: Devin 29.4%, Cursor 8.9%, Copilot 6.7%,
    Claude Code 5.5%, Codex 0.5%

RQ2 findings from main.tex:
  - Strategy distributions vary by agent (Cramér's V ranges 0.02–0.32)
  - Codex is 99.9% V1 (concentrated in one repository)
  - Claude Code is 57.4% V1, 36.1% V2
  - Devin/Copilot/Cursor lean toward V2 (41–56%)
  - Agents produce Postponed chunks at 0.64% vs humans at 6.56%

Usage (from project root, with venv active):
    python analysis/reproduce_paper_statistics.py

Required packages:
    scipy, statsmodels, scikit-posthocs, pandas, numpy

Output:
    analysis/results/reproduce_statistics.json  -- structured results
    analysis/results/reproduce_statistics.txt   -- human-readable report
"""

from __future__ import annotations

import json
import sys
import warnings
from io import StringIO
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Path setup ─────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
PROJECT_ROOT = _here.parent
for candidate in [_here, *_here.parents]:
    if (candidate / "analysis" / "common.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from analysis.common import (  # noqa: E402
    load_tables,
    build_chunk_frame,
    build_merge_frame,
    STRATEGY_ORDER,
)

RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUT = RESULTS_DIR / "reproduce_statistics.json"
TXT_OUT = RESULTS_DIR / "reproduce_statistics.txt"

STRATEGIES_CLASSIFIABLE = [s for s in STRATEGY_ORDER if s != "Imprecise"]

# Ghiotto et al. (TSE 2020) baseline from Table 13
GHIOTTO_PROPS = {
    "V1": 0.5,
    "V2": 0.25,
    "CC": 0.03,
    "CB": 0.09,
    "NC": 0.13,
    "NN": 0.0,
}

# ── Utility functions ───────────────────────────────────────────────────────

def _sig(p: float) -> str:
    """Significance indicator."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _wilson(k: int, n: int) -> tuple[float, float, float]:
    """Returns (proportion, ci_lo, ci_hi) using Wilson score interval."""
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return k / n, lo, hi


def _cramers_v(chi2_val: float, n: int, r: int, c: int) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    phi2 = chi2_val / n
    phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    cc = c - (c - 1) ** 2 / (n - 1)
    denom = min(rc - 1, cc - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2c / denom))


def _v_label(v: float) -> str:
    """Cramér's V interpretation label."""
    a = abs(v)
    if a < 0.10: return "weak"
    if a < 0.30: return "moderate"
    return "strong"


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Non-parametric effect size: Cliff's delta."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dom = sum(int(xi > yj) - int(xi < yj) for xi in x for yj in y)
    return dom / (len(x) * len(y))


def _delta_label(d: float) -> str:
    """Cliff's delta interpretation label."""
    a = abs(d)
    if a < 0.147: return "negligible"
    if a < 0.330: return "small"
    if a < 0.474: return "medium"
    return "large"


def _safe_chi2(table: np.ndarray) -> dict:
    """Chi-squared test with cell count diagnostics."""
    chi2, p, dof, expected = chi2_contingency(table)
    min_exp = float(expected.min())
    pct_lt5 = float((expected < 5).sum() / expected.size * 100)
    return {
        "chi2": round(float(chi2), 3),
        "p": float(p),
        "dof": int(dof),
        "significant": _sig(p),
        "min_expected_count": round(min_exp, 2),
        "pct_cells_lt5": round(pct_lt5, 1),
        "warning": "small cells (>20%)" if pct_lt5 > 20 else None,
    }


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _section(title: str, buf: StringIO) -> None:
    """Write section header to buffer."""
    buf.write(f"\n{'='*78}\n{title}\n{'='*78}\n")


def _subsection(title: str, buf: StringIO) -> None:
    """Write subsection header to buffer."""
    buf.write(f"\n{'-'*70}\n{title}\n{'-'*70}\n")


# ── SANITY CHECKS ───────────────────────────────────────────────────────────

def sanity_checks(chunks: pd.DataFrame, merges: pd.DataFrame, buf: StringIO) -> dict:
    """
    Verify that data counts match expectations from main.tex § 4.1.

    main.tex expectations (after merge-level deduplication):
      - ~50,700 internal merges
      - ~14,960 conflicting merges (29.5%)
      - ~121,599 conflict chunks
      - ~83,043 localized (68.3%)
      - ~45,847 Imprecise (31.7%)
      - ~75,752 resolved/classifiable (62.3%)
      - ~7,291 Postponed (6.0% of chunks)
    """
    _section("SANITY CHECKS: Data Counts vs. main.tex Expectations", buf)
    results = {}

    # Merge counts
    n_merges_total = len(merges)
    n_merges_conflict = (merges["n_chunks"] > 0).sum()
    n_merges_conflict_rate = n_merges_conflict / n_merges_total if n_merges_total > 0 else 0

    buf.write(f"\n--- Merge Counts ---\n")
    buf.write(f"  Total internal merges:      {n_merges_total:>10,}   "
              f"(main.tex: ~50,700)\n")
    buf.write(f"  Conflicting merges:         {n_merges_conflict:>10,}   "
              f"(main.tex: ~14,960, rate 29.5%)\n")
    buf.write(f"  Conflict rate:              {n_merges_conflict_rate:>10.1%}   "
              f"(main.tex: 29.5%)\n")

    results["merges"] = {
        "n_total": n_merges_total,
        "n_conflicting": n_merges_conflict,
        "conflict_rate": round(n_merges_conflict_rate, 4),
    }

    # Chunk counts
    n_chunks_total = len(chunks)
    n_classifiable = (chunks["strategy"] != "Imprecise").sum()
    n_imprecise = n_chunks_total - n_classifiable

    # Count Postponed (strategy_raw == "Postponed")
    n_postponed = 0
    if "strategy_raw" in chunks.columns:
        n_postponed = (chunks["strategy_raw"] == "Postponed").sum()

    buf.write(f"\n--- Chunk Counts ---\n")
    buf.write(f"  Total chunks:               {n_chunks_total:>10,}   "
              f"(main.tex: 121,599)\n")
    buf.write(f"  Classifiable:               {n_classifiable:>10,}   "
              f"({n_classifiable/n_chunks_total:.1%}, main.tex: 62.3%)\n")
    buf.write(f"  Imprecise:                  {n_imprecise:>10,}   "
              f"({n_imprecise/n_chunks_total:.1%}, main.tex: 37.7%)\n")
    if n_postponed > 0:
        buf.write(f"  Postponed (subset of Imprecise): {n_postponed:>6,}   "
                  f"({n_postponed/n_chunks_total:.1%}, main.tex: 6.0%)\n")

    results["chunks"] = {
        "n_total": n_chunks_total,
        "n_classifiable": n_classifiable,
        "n_imprecise": n_imprecise,
        "n_postponed": n_postponed,
        "classifiable_rate": round(n_classifiable / n_chunks_total, 4),
        "imprecise_rate": round(n_imprecise / n_chunks_total, 4),
    }

    # Resolver type counts (if available in merges) — CONFLICTING MERGES ONLY
    if "resolver_type" in merges.columns:
        buf.write(f"\n--- Resolver Attribution (by merge, CONFLICTING ONLY) ---\n")
        conflicting_only = merges[merges["n_chunks"] > 0]
        resolver_counts = conflicting_only["resolver_type"].value_counts().to_dict()
        for resolver_type in ["human", "agent", "agent-assisted"]:
            cnt = resolver_counts.get(resolver_type, 0)
            pct = cnt / n_merges_conflict if n_merges_conflict > 0 else 0
            buf.write(f"  {resolver_type:<20}: {cnt:>6,} ({pct:>6.1%})\n")
        results["resolver"] = {k: int(v) for k, v in resolver_counts.items()}

    buf.write("\n[End of sanity checks]\n")
    return results


# ── RQ1: RESOLVER ATTRIBUTION ───────────────────────────────────────────────

def run_rq1(chunks: pd.DataFrame, merges: pd.DataFrame, buf: StringIO) -> dict:
    """
    RQ1: Who resolves the merge conflicts (agent vs. human)?

    From main.tex:
      - 14,381 human-resolved (96.1%)
      - 462 agent-resolved (3.1%)
      - 117 agent-assisted (0.8%)
      - Per-agent rates: Devin 29.4%, Cursor 8.9%, Copilot 6.7%,
        Claude Code 5.5%, Codex 0.5%
    """
    _section("RQ1: WHO RESOLVES THE CONFLICTS?", buf)
    results: dict[str, Any] = {}

    # Global resolver distribution (merge level) — CONFLICTING MERGES ONLY
    _subsection("1. Global resolver distribution (by merge, CONFLICTING ONLY)", buf)
    if "resolver_type" in merges.columns:
        conflicting = merges[merges["n_chunks"] > 0]  # conflicting merges only
        resolver_dist = conflicting["resolver_type"].value_counts()
        total = len(conflicting)

        buf.write(f"  Total conflicting merges: {total:,}\n\n")
        for rt in ["human", "agent", "agent-assisted"]:
            cnt = resolver_dist.get(rt, 0)
            rate = cnt / total if total > 0 else 0
            buf.write(f"  {rt:<20}: {cnt:>6,} ({rate:>6.1%})\n")

        results["global_resolver"] = {
            rt: int(resolver_dist.get(rt, 0))
            for rt in ["human", "agent", "agent-assisted"]
        }

    # Per-agent self-resolution rates (merge level)
    _subsection("2. Per-agent self-resolution rates (Wilson CI)", buf)
    if "agent" in merges.columns and "resolver_type" in merges.columns:
        conflicting = merges[merges["n_chunks"] > 0].copy()
        agents = sorted(conflicting["agent"].dropna().unique())

        per_agent_rates = []
        for agent in agents:
            agent_data = conflicting[conflicting["agent"] == agent]
            n_total = len(agent_data)
            n_self = (agent_data["resolver_type"] == "agent").sum()
            rate, lo, hi = _wilson(n_self, n_total)

            per_agent_rates.append({
                "agent": agent,
                "n_total": n_total,
                "n_self_resolved": n_self,
                "rate": round(rate, 4),
                "ci_lo_95": round(lo, 4),
                "ci_hi_95": round(hi, 4),
            })

            buf.write(f"  {agent:<20}  {rate:>6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  "
                      f"({n_self}/{n_total})\n")

        results["per_agent_self_resolution"] = per_agent_rates

    # Per-agent assisted rates
    _subsection("3. Per-agent agent-assisted rates (Wilson CI)", buf)
    if "agent" in merges.columns and "resolver_type" in merges.columns:
        conflicting = merges[merges["n_chunks"] > 0].copy()
        agents = sorted(conflicting["agent"].dropna().unique())

        per_agent_assisted = []
        for agent in agents:
            agent_data = conflicting[conflicting["agent"] == agent]
            n_total = len(agent_data)
            n_assisted = (agent_data["resolver_type"] == "agent-assisted").sum()
            rate, lo, hi = _wilson(n_assisted, n_total)

            per_agent_assisted.append({
                "agent": agent,
                "n_total": n_total,
                "n_assisted": n_assisted,
                "rate": round(rate, 4),
                "ci_lo_95": round(lo, 4),
                "ci_hi_95": round(hi, 4),
            })

            buf.write(f"  {agent:<20}  {rate:>6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  "
                      f"({n_assisted}/{n_total})\n")

        results["per_agent_assisted"] = per_agent_assisted

    # Chi-squared test: resolver type vs. agent
    _subsection("4. Chi-squared test: resolver type × agent", buf)
    if "agent" in merges.columns and "resolver_type" in merges.columns:
        conflicting = merges[merges["n_chunks"] > 0].copy()
        contingency = (
            conflicting.groupby(["agent", "resolver_type"])
            .size()
            .unstack(fill_value=0)
        )
        buf.write(f"\nContingency table:\n{contingency.to_string()}\n\n")

        chi2_result = _safe_chi2(contingency.values)
        r, c = contingency.shape
        V = _cramers_v(chi2_result["chi2"], len(conflicting), r, c)
        chi2_result["cramers_v"] = round(V, 4)
        chi2_result["cramers_v_label"] = _v_label(V)

        buf.write(f"  χ²({chi2_result['dof']}) = {chi2_result['chi2']:.1f}, "
                  f"p = {chi2_result['p']:.2e}  {chi2_result['significant']}\n")
        buf.write(f"  Cramér's V (bias-corrected) = {V:.4f}  ({_v_label(V)})\n")

        results["chi2_resolver_agent"] = chi2_result

    return results


# ── RQ2: RESOLUTION STRATEGIES ──────────────────────────────────────────────

def run_rq2(chunks: pd.DataFrame, buf: StringIO) -> dict:
    """
    RQ2: How do agents resolve their conflicts?

    From main.tex:
      - V1: agents 79.7%, humans 45.1%
      - V2: agents 12.6%, humans 26.0%
      - Agents: V=0.23 vs humans
      - Per-agent: Codex 99.9% V1 (V=0.32), Claude Code 57.4% V1 (V=0.02)
      - Cursor vs Devin: indistinguishable (V=0.08, p>0.05)
    """
    _section("RQ2: RESOLUTION STRATEGIES", buf)
    results: dict[str, Any] = {}

    # Dataset overview
    _subsection("1. Dataset overview", buf)
    n_total = len(chunks)
    classifiable = chunks[chunks["strategy"] != "Imprecise"]
    n_classif = len(classifiable)
    n_imprecise = n_total - n_classif

    buf.write(f"  Total chunks:               {n_total:>10,}\n")
    buf.write(f"  Classifiable:               {n_classif:>10,}  ({n_classif/n_total:.1%})\n")
    buf.write(f"  Imprecise:                  {n_imprecise:>10,}  ({n_imprecise/n_total:.1%})\n")

    results["dataset"] = {
        "n_total_chunks": n_total,
        "n_classifiable": n_classif,
        "n_imprecise": n_imprecise,
        "imprecise_rate": round(n_imprecise / n_total, 4),
    }

    # Global strategy distribution
    _subsection("2. Global strategy distribution (over classifiable chunks)", buf)
    counts = (classifiable["strategy"]
              .value_counts()
              .reindex(STRATEGIES_CLASSIFIABLE, fill_value=0))

    wilson_rows = []
    buf.write(f"  {'Strategy':<12} {'Count':>10} {'Pct':>8} {'95% CI':>25}\n")
    for strat in STRATEGIES_CLASSIFIABLE:
        cnt = int(counts[strat])
        rate, lo, hi = _wilson(cnt, n_classif)
        wilson_rows.append({
            "strategy": strat,
            "count": cnt,
            "proportion": round(rate, 4),
            "ci_lo_95": round(lo, 4),
            "ci_hi_95": round(hi, 4),
        })
        buf.write(f"  {strat:<12} {cnt:>10,} {rate:>8.1%} [{lo:.1%}, {hi:.1%}]\n")

    results["wilson_ci_global"] = wilson_rows

    # Per-agent strategy distributions
    _subsection("3. Per-agent strategy distributions (resolved chunks only)", buf)
    if "agent" in classifiable.columns and "resolver_type" in classifiable.columns:
        agent_resolved = classifiable[classifiable["resolver_type"] == "agent"]
        human_resolved = classifiable[classifiable["resolver_type"] == "human"]

        agents = sorted(agent_resolved["agent"].dropna().unique())

        per_agent_strats = []
        buf.write(f"\n  Agent strategy distributions:\n")
        buf.write(f"  {'Agent':<20} {'n':>8}   V1      V2      CC      CB      NC      "
                  f"NN      V (vs human)\n")

        for agent in agents:
            agent_data = agent_resolved[agent_resolved["agent"] == agent]
            n = len(agent_data)

            strat_dist = agent_data["strategy"].value_counts(normalize=True).reindex(
                STRATEGIES_CLASSIFIABLE, fill_value=0
            )

            # Chi-squared for this agent vs humans
            ct_agent = pd.Series(0, index=STRATEGIES_CLASSIFIABLE)
            ct_human = pd.Series(0, index=STRATEGIES_CLASSIFIABLE)
            for s in STRATEGIES_CLASSIFIABLE:
                ct_agent[s] = (agent_data["strategy"] == s).sum()
                ct_human[s] = (human_resolved["strategy"] == s).sum()

            contingency = pd.DataFrame({
                agent: ct_agent,
                "human": ct_human,
            }).T

            try:
                chi2, p, dof, expected = chi2_contingency(contingency.values)
                r, c = contingency.shape
                V = _cramers_v(chi2, n + len(human_resolved), r, c)
            except ValueError:
                # If chi-squared fails, use V=0
                V = 0.0

            per_agent_strats.append({
                "agent": agent,
                "n": n,
                "v1": round(float(strat_dist["V1"]), 4),
                "v2": round(float(strat_dist["V2"]), 4),
                "cc": round(float(strat_dist["CC"]), 4),
                "cb": round(float(strat_dist["CB"]), 4),
                "nc": round(float(strat_dist["NC"]), 4),
                "nn": round(float(strat_dist["NN"]), 4),
                "cramers_v_vs_human": round(V, 4),
                "cramers_v_label": _v_label(V),
            })

            buf.write(f"  {agent:<20} {n:>8,}   {strat_dist['V1']:>5.1%}  "
                      f"{strat_dist['V2']:>5.1%}  {strat_dist['CC']:>5.1%}  "
                      f"{strat_dist['CB']:>5.1%}  {strat_dist['NC']:>5.1%}  "
                      f"{strat_dist['NN']:>5.1%}   {V:.4f} ({_v_label(V)})\n")

        # Human baseline
        human_strat_dist = human_resolved["strategy"].value_counts(normalize=True).reindex(
            STRATEGIES_CLASSIFIABLE, fill_value=0
        )
        buf.write(f"  {'HUMAN (ref)':<20} {len(human_resolved):>8,}   "
                  f"{human_strat_dist['V1']:>5.1%}  {human_strat_dist['V2']:>5.1%}  "
                  f"{human_strat_dist['CC']:>5.1%}  {human_strat_dist['CB']:>5.1%}  "
                  f"{human_strat_dist['NC']:>5.1%}  {human_strat_dist['NN']:>5.1%}\n")

        results["per_agent_strategies"] = per_agent_strats

    # Pairwise agent contrasts (with both Bonferroni and Holm corrections)
    _subsection("4. Pairwise agent contrasts (Bonferroni & Holm corrections)", buf)
    if "agent" in classifiable.columns and "resolver_type" in classifiable.columns:
        agent_resolved = classifiable[classifiable["resolver_type"] == "agent"]
        agents = sorted(agent_resolved["agent"].dropna().unique())

        pairwise_results = []
        pairs = list(combinations(agents, 2))

        # First pass: compute raw p-values
        buf.write(f"\n  Raw p-values and effect sizes:\n")
        buf.write(f"  {'Agent Pair':<40} {'V':>8}  {'Effect':>12}  "
                  f"{'p (raw)':>12}\n")

        for a1, a2 in pairs:
            d1 = agent_resolved[agent_resolved["agent"] == a1]
            d2 = agent_resolved[agent_resolved["agent"] == a2]

            # Skip pair if either has too few observations
            if len(d1) < 5 or len(d2) < 5:
                buf.write(f"  {a1} vs {a2:<33} [SKIP] too few observations (n1={len(d1)}, n2={len(d2)})\n")
                continue

            ct1 = d1["strategy"].value_counts().reindex(
                STRATEGIES_CLASSIFIABLE, fill_value=0
            )
            ct2 = d2["strategy"].value_counts().reindex(
                STRATEGIES_CLASSIFIABLE, fill_value=0
            )

            contingency = pd.DataFrame({
                a1: ct1,
                a2: ct2,
            }).T

            try:
                chi2, p, dof, expected = chi2_contingency(contingency.values)
                r, c = contingency.shape
                V = _cramers_v(chi2, len(d1) + len(d2), r, c)

                pairwise_results.append({
                    "pair": f"{a1} vs {a2}",
                    "cramers_v": round(V, 4),
                    "cramers_v_label": _v_label(V),
                    "p_raw": float(p),
                    "significant_raw": _sig(p),
                })

                buf.write(f"  {a1} vs {a2:<33} {V:>8.4f}  "
                          f"{_v_label(V):>12}  {p:.2e}\n")
            except ValueError as e:
                # Expected frequencies too small; skip this pair
                buf.write(f"  {a1} vs {a2:<33} [SKIP] expected freq issue\n")
                continue

        # Apply both Bonferroni and Holm corrections
        if pairwise_results and len(pairwise_results) > 0:
            p_vals = [r["p_raw"] for r in pairwise_results if "p_raw" in r]

            if p_vals:
                # Bonferroni
                reject_bonf, p_bonf, _, _ = multipletests(p_vals, method="bonferroni")
                # Holm
                reject_holm, p_holm, _, _ = multipletests(p_vals, method="holm")

                buf.write(f"\n  Corrected p-values (Bonferroni & Holm):\n")
                buf.write(f"  {'Agent Pair':<40} {'Bonferroni':>15}  {'Holm':>15}\n")

                for i, r in enumerate(pairwise_results):
                    if "p_raw" in r:
                        r["p_bonferroni"] = float(p_bonf[i])
                        r["significant_bonferroni"] = _sig(p_bonf[i])
                        r["p_holm"] = float(p_holm[i])
                        r["significant_holm"] = _sig(p_holm[i])

                        pair_name = r["pair"]
                        buf.write(f"  {pair_name:<40} {p_bonf[i]:>15.2e}  {p_holm[i]:>15.2e}\n")
                        buf.write(f"    {'':<40} {_sig(p_bonf[i]):>15}  {_sig(p_holm[i]):>15}\n")
            else:
                buf.write(f"\n  [NO VALID PAIRS] All pairs skipped due to small sample sizes\n")

        results["pairwise_agent_contrasts"] = pairwise_results

    # Imprecise rate by agent
    _subsection("5. Imprecise rate by agent", buf)
    if "agent" in chunks.columns:
        imprecise_by_agent = (
            chunks.groupby("agent")
            .apply(lambda g: (g["strategy"] == "Imprecise").sum() / len(g))
            .sort_values(ascending=False)
        )

        for agent, rate in imprecise_by_agent.items():
            buf.write(f"  {agent:<20}: {rate:>6.1%}\n")

        results["imprecise_rate_by_agent"] = {
            agent: round(rate, 4)
            for agent, rate in imprecise_by_agent.items()
        }

    # Postponed chunks (subset of Imprecise) by resolver
    _subsection("6. Postponed chunks by resolver type", buf)
    if "strategy_raw" in chunks.columns and "resolver_type" in chunks.columns:
        for resolver in ["agent", "human"]:
            resolver_data = chunks[chunks["resolver_type"] == resolver]
            n_total = len(resolver_data)
            n_postponed = (resolver_data["strategy_raw"] == "Postponed").sum()
            rate = n_postponed / n_total if n_total > 0 else 0

            buf.write(f"  {resolver:<15}: {n_postponed:>6,} / {n_total:>10,} "
                      f"({rate:>6.1%})\n")

        results["postponed_by_resolver"] = {
            "agent": round(
                (chunks[(chunks["resolver_type"] == "agent") &
                       (chunks["strategy_raw"] == "Postponed")].shape[0] /
                 chunks[chunks["resolver_type"] == "agent"].shape[0]),
                4
            ),
            "human": round(
                (chunks[(chunks["resolver_type"] == "human") &
                       (chunks["strategy_raw"] == "Postponed")].shape[0] /
                 chunks[chunks["resolver_type"] == "human"].shape[0]),
                4
            ),
        }

    return results


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    """Load data, run all analyses, write outputs."""
    print("=" * 80)
    print("REPRODUCING PAPER STATISTICS (reproduce_paper_statistics.py)")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    tables = load_tables()
    print("Building chunk frame...")
    chunks = build_chunk_frame(tables)
    print("Building merge frame...")
    merges = build_merge_frame(tables)
    print(f"  Total chunks: {len(chunks):,}  |  Total merges: {len(merges):,}")

    # Buffer for text output
    buf = StringIO()
    buf.write("STATISTICAL ANALYSIS RESULTS\n")
    buf.write("Reproducing: How AI Coding Agents Resolve Merge Conflicts\n")
    buf.write("=" * 78 + "\n")

    # Run all analyses
    all_results = {}

    print("\nRunning sanity checks...")
    all_results["sanity_checks"] = sanity_checks(chunks, merges, buf)

    print("Running RQ1 analyses...")
    all_results["rq1"] = run_rq1(chunks, merges, buf)

    print("Running RQ2 analyses...")
    all_results["rq2"] = run_rq2(chunks, buf)

    # Write outputs
    print(f"\nWriting results to {JSON_OUT}...")
    with open(JSON_OUT, "w") as f:
        json.dump(_to_json_serializable(all_results), f, indent=2)

    print(f"Writing results to {TXT_OUT}...")
    text_output = buf.getvalue()
    with open(TXT_OUT, "w") as f:
        f.write(text_output)

    print("\n" + "=" * 80)
    print("SUCCESS! Results written:")
    print(f"  JSON: {JSON_OUT}")
    print(f"  TXT:  {TXT_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)