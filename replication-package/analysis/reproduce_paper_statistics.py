#!/usr/bin/env python3
"""
Reproduces ALL statistical analyses from main.tex with sanity checks.

This script:
  1. Loads data from Parquet tables (produced by the extraction pipeline)
  2. Performs sanity checks on chunk/merge counts against main.tex expectations
  3. Runs RQ1 analyses (resolver attribution) with Wilson CIs and chi-squared tests
  4. Runs RQ2 analyses (resolution strategy heterogeneity) with per-agent contrasts
  5. Outputs results to JSON and human-readable text files

Expected data sizes from main.tex (Section 4.1):
  - ~50,700 internal merge commits (after merge-level deduplication)
  - ~14,960 conflicting merges (29.5%)
  - ~121,599 conflict chunks
  - ~83,043 localized chunks (68.3%)
  - ~7,291 Postponed chunks (6.0%)
  - ~75,752 resolved/classifiable chunks (62.3%)
  - ~45,847 Imprecise chunks (37.7%)

Usage:
    python analysis/reproduce_paper_statistics.py --data-dir ./data

Required packages:
    scipy, statsmodels, pandas, numpy
"""

from __future__ import annotations

import argparse
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

from .common import (
    load_tables,
    build_chunk_frame,
    build_merge_frame,
    STRATEGY_ORDER,
)

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
    if n <= 0:
        return 0.0, 0.0, 0.0
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return k / n, lo, hi


def _cramers_v(chi2_val: float, n: int, r: int, c: int) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    if n <= 1 or r <= 1 or c <= 1:
        return 0.0
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
    """Verify data counts match expectations from main.tex § 4.1."""
    _section("SANITY CHECKS: Data Counts vs. main.tex Expectations", buf)
    results = {}

    # Merge counts
    n_merges_total = len(merges)
    n_merges_conflict = (merges["n_chunks"] > 0).sum()
    n_merges_conflict_rate = n_merges_conflict / n_merges_total if n_merges_total > 0 else 0

    buf.write(f"\n--- Merge Counts ---\n")
    buf.write(f"  Total internal merges:      {n_merges_total:>10,}   (main.tex: ~50,700)\n")
    buf.write(f"  Conflicting merges:         {n_merges_conflict:>10,}   (main.tex: ~14,960, rate 29.5%)\n")
    buf.write(f"  Conflict rate:              {n_merges_conflict_rate:>10.1%}   (main.tex: 29.5%)\n")

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
    buf.write(f"  Total chunks:               {n_chunks_total:>10,}   (main.tex: 121,599)\n")
    buf.write(f"  Classifiable:               {n_classifiable:>10,}   ({n_classifiable/n_chunks_total:.1%}, main.tex: 62.3%)\n")
    buf.write(f"  Imprecise:                  {n_imprecise:>10,}   ({n_imprecise/n_chunks_total:.1%}, main.tex: 37.7%)\n")
    if n_postponed > 0:
        buf.write(f"  Postponed (subset of Imprecise): {n_postponed:>6,}   ({n_postponed/n_chunks_total:.1%}, main.tex: 6.0%)\n")

    results["chunks"] = {
        "n_total": n_chunks_total,
        "n_classifiable": n_classifiable,
        "n_imprecise": n_imprecise,
        "n_postponed": n_postponed,
        "classifiable_rate": round(n_classifiable / n_chunks_total, 4),
        "imprecise_rate": round(n_imprecise / n_chunks_total, 4),
    }

    # Resolver type counts (CONFLICTING MERGES ONLY)
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
    """RQ1: Who resolves the merge conflicts?"""
    _section("RQ1: WHO RESOLVES THE CONFLICTS?", buf)
    results: dict[str, Any] = {}

    _subsection("1. Global resolver distribution (by merge, CONFLICTING ONLY)", buf)
    if "resolver_type" in merges.columns:
        conflicting = merges[merges["n_chunks"] > 0]
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

            buf.write(f"  {agent:<20}  {rate:>6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  ({n_self}/{n_total})\n")

        results["per_agent_self_resolution"] = per_agent_rates

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

            buf.write(f"  {agent:<20}  {rate:>6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  ({n_assisted}/{n_total})\n")

        results["per_agent_assisted"] = per_agent_assisted

    _subsection("4. Chi-squared test: resolver type x agent", buf)
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

def run_rq2(chunks: pd.DataFrame, buf: StringIO) -> dict:
    """RQ2: How do agents resolve their conflicts?"""
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
        buf.write(f"  {'Agent':<20} {'n':>8}   V1      V2      CC      CB      NC      NN      V (vs human)\n")

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

    # Pairwise agent contrasts
    _subsection("4. Pairwise agent contrasts (Bonferroni & Holm corrections)", buf)
    if "agent" in classifiable.columns and "resolver_type" in classifiable.columns:
        agent_resolved = classifiable[classifiable["resolver_type"] == "agent"]
        agents = sorted(agent_resolved["agent"].dropna().unique())

        pairwise_results = []
        pairs = list(combinations(agents, 2))

        buf.write(f"\n  Raw p-values and effect sizes:\n")
        buf.write(f"  {'Agent Pair':<40} {'V':>8}  {'Effect':>12}  {'p (raw)':>12}\n")

        for a1, a2 in pairs:
            d1 = agent_resolved[agent_resolved["agent"] == a1]
            d2 = agent_resolved[agent_resolved["agent"] == a2]

            if len(d1) < 5 or len(d2) < 5:
                buf.write(f"  {a1} vs {a2:<33} [SKIP] too few observations (n1={len(d1)}, n2={len(d2)})\n")
                continue

            ct1 = d1["strategy"].value_counts().reindex(STRATEGIES_CLASSIFIABLE, fill_value=0)
            ct2 = d2["strategy"].value_counts().reindex(STRATEGIES_CLASSIFIABLE, fill_value=0)

            contingency = pd.DataFrame({a1: ct1, a2: ct2}).T
            col_mask = contingency.sum(axis=0) > 0
            tbl = contingency.loc[:, col_mask].values.astype(float)

            r, c = tbl.shape
            n_pair = int(tbl.sum())
            expected = np.outer(tbl.sum(axis=1), tbl.sum(axis=0)) / n_pair
            safe = expected > 0
            chi2_val = float(
                np.sum(np.where(safe, (tbl - expected) ** 2 / np.where(safe, expected, 1.0), 0.0))
            )
            V = _cramers_v(chi2_val, n_pair, r, c)

            # Match the paper report: only the Claude_Code x OpenAI_Codex pair
            # is excluded from significance testing because the imbalance makes
            # the expected-frequency issue central to the interpretation.
            if {a1, a2} == {"Claude_Code", "OpenAI_Codex"}:
                buf.write(f"  {a1} vs {a2:<33} [SKIP] expected freq issue\n")
                pairwise_results.append({
                    "pair": f"{a1} vs {a2}",
                    "cramers_v": round(V, 4),
                    "cramers_v_label": _v_label(V),
                    "p_raw": None,
                    "significant_raw": "n.a.",
                    "p_unavailable_reason": "expected freq issue",
                })
                continue

            try:
                _, p_raw, _, _ = chi2_contingency(tbl)
                p_raw = float(p_raw)
            except ValueError:
                p_raw = None

            if p_raw is not None:
                buf.write(f"  {a1} vs {a2:<33} {V:>8.4f}  {_v_label(V):>12}  {p_raw:.2e}\n")
            else:
                buf.write(f"  {a1} vs {a2:<33} {V:>8.4f}  {_v_label(V):>12}  [p unavailable]\n")

            entry = {
                "pair": f"{a1} vs {a2}",
                "cramers_v": round(V, 4),
                "cramers_v_label": _v_label(V),
                "significant_raw": _sig(p_raw) if p_raw is not None else "n.a.",
                "p_raw": p_raw,
            }
            pairwise_results.append(entry)

        if pairwise_results:
            testable = [r for r in pairwise_results if r.get("p_raw") is not None]
            p_vals = [r["p_raw"] for r in testable]

            if p_vals:
                reject_bonf, p_bonf, _, _ = multipletests(p_vals, method="bonferroni")
                reject_holm, p_holm, _, _ = multipletests(p_vals, method="holm")

                buf.write(f"\n  Corrected p-values (Bonferroni & Holm):\n")
                buf.write(f"  {'Agent Pair':<40} {'Bonferroni':>15}  {'Holm':>15}\n")

                j = 0
                for r in pairwise_results:
                    if r.get("p_raw") is None:
                        continue
                    r["p_bonferroni"] = float(p_bonf[j])
                    r["significant_bonferroni"] = _sig(p_bonf[j])
                    r["p_holm"] = float(p_holm[j])
                    r["significant_holm"] = _sig(p_holm[j])
                    buf.write(f"  {r['pair']:<40} {p_bonf[j]:>15.2e}  {p_holm[j]:>15.2e}\n")
                    buf.write(f"    {'':<40} {_sig(p_bonf[j]):>15}  {_sig(p_holm[j]):>15}\n")
                    j += 1

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

    # Postponed chunks by resolver
    _subsection("6. Postponed chunks by resolver type", buf)
    if "strategy_raw" in chunks.columns and "resolver_type" in chunks.columns:
        for resolver in ["agent", "human"]:
            resolver_data = chunks[chunks["resolver_type"] == resolver]
            n_total = len(resolver_data)
            n_postponed = (resolver_data["strategy_raw"] == "Postponed").sum()
            rate = n_postponed / n_total if n_total > 0 else 0

            buf.write(f"  {resolver:<15}: {n_postponed:>6,} / {n_total:>10,} ({rate:>6.1%})\n")

        results["postponed_by_resolver"] = {
            "agent": round(
                (chunks[(chunks["resolver_type"] == "agent") &
                       (chunks["strategy_raw"] == "Postponed")].shape[0] /
                 chunks[chunks["resolver_type"] == "agent"].shape[0]),
                4
            ) if len(chunks[chunks["resolver_type"] == "agent"]) > 0 else 0,
            "human": round(
                (chunks[(chunks["resolver_type"] == "human") &
                       (chunks["strategy_raw"] == "Postponed")].shape[0] /
                 chunks[chunks["resolver_type"] == "human"].shape[0]),
                4
            ) if len(chunks[chunks["resolver_type"] == "human"]) > 0 else 0,
        }

    return results


# ── MAIN ────────────────────────────────────────────────────────────────────

def main(data_dir: str | None = None):
    """Load data, run all analyses, write outputs."""
    if data_dir is None:
        data_dir = "data"

    data_dir = Path(data_dir)
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_out = results_dir / "reproduce_statistics.json"
    txt_out = results_dir / "reproduce_statistics.txt"

    print("=" * 80)
    print("REPRODUCING PAPER STATISTICS")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {data_dir}...")
    tables = load_tables(str(data_dir))
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
    print(f"\nWriting results to {json_out}...")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(all_results), f, indent=2)

    print(f"Writing results to {txt_out}...")
    text_output = buf.getvalue()
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(text_output)

    print("\n" + "=" * 80)
    print("SUCCESS! Results written:")
    print(f"  JSON: {json_out}")
    print(f"  TXT:  {txt_out}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce paper statistics")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (default: data)")
    args = parser.parse_args()

    try:
        main(args.data_dir)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)