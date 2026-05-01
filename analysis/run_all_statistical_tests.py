"""
run_all_statistical_tests.py
============================
Runs every statistical test for RQ1, RQ2, and RQ3 and writes results to:

    analysis/results/statistical_tests.json   -- structured data (for interpretation)
    analysis/results/statistical_tests.txt    -- human-readable report

Usage (from project root, with the project venv active):
    python analysis/run_all_statistical_tests.py

Required packages (all in the project venv):
    scipy, statsmodels, scikit-posthocs (pip install scikit-posthocs)

The script is intentionally verbose: every helper function prints progress
so you can see where it is when running on a large dataset.
"""

from __future__ import annotations

import json
import sys
import traceback
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

from analysis.common import (       # noqa: E402
    load_tables,
    build_chunk_frame,
    build_merge_frame,
    STRATEGY_ORDER,
)

RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUT = RESULTS_DIR / "statistical_tests.json"
TXT_OUT  = RESULTS_DIR / "statistical_tests.txt"

STRATEGIES_CLASSIFIABLE = [s for s in STRATEGY_ORDER if s != "Imprecise"]

# Ghiotto et al. (TSE 2020) published proportions for the six classifiable
# strategies — normalised to sum to 1 over those six categories.
# Source: Table 13 / text of Ghiotto et al. 2020.
# Adjust these values if you have more precise figures from the paper.
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
    a = abs(d)
    if a < 0.147: return "negligible"
    if a < 0.330: return "small"
    if a < 0.474: return "medium"
    return "large"


def _safe_chi2(table: np.ndarray) -> dict:
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


def _section(title: str, buf: StringIO) -> None:
    buf.write(f"\n{'='*70}\n{title}\n{'='*70}\n")


def _subsection(title: str, buf: StringIO) -> None:
    buf.write(f"\n{'-'*60}\n{title}\n{'-'*60}\n")


# ── Data loading ────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading tables...")
    tables = load_tables()
    print("Building chunk frame...")
    chunks = build_chunk_frame(tables)
    print("Building merge frame...")
    merges = build_merge_frame(tables)
    print(f"  chunks: {len(chunks):,}  |  merges: {len(merges):,}")
    return chunks, merges


# ── RQ1 tests ───────────────────────────────────────────────────────────────

def run_rq1(chunks: pd.DataFrame, merges: pd.DataFrame, buf: StringIO) -> dict:
    _section("RQ1 — Structural Nature", buf)
    results: dict[str, Any] = {}

    # 1. Wilson CIs for per-agent conflict rates
    _subsection("1. Wilson CIs: per-agent conflict rate (conflicting merges / all merges)", buf)
    if "agent" in merges.columns and "has_conflict" in merges.columns:
        ag = (
            merges.groupby("agent")
            .agg(n_total=("has_conflict", "count"),
                 n_conflict=("has_conflict", "sum"))
            .reset_index()
        )
        ci_rows = []
        for _, row in ag.iterrows():
            rate, lo, hi = _wilson(int(row["n_conflict"]), int(row["n_total"]))
            ci_rows.append({
                "agent": row["agent"],
                "n_total": int(row["n_total"]),
                "n_conflicting": int(row["n_conflict"]),
                "rate": round(rate, 4),
                "ci_lo_95": round(lo, 4),
                "ci_hi_95": round(hi, 4),
            })
            buf.write(f"  {row['agent']:<20} {rate:.1%}  95% CI [{lo:.1%}, {hi:.1%}]"
                      f"  ({int(row['n_conflict'])}/{int(row['n_total'])})\n")
        results["wilson_ci_conflict_rates"] = sorted(ci_rows, key=lambda r: -r["rate"])
    else:
        buf.write("  [SKIP] 'agent' or 'has_conflict' column missing from merges frame\n")
        results["wilson_ci_conflict_rates"] = None

    # 2. Kruskal-Wallis: chunks per conflicting merge across agents
    _subsection("2. Kruskal-Wallis: chunks per conflicting merge across agents", buf)
    kw_chunks: dict[str, Any] = {}
    if "agent" in merges.columns and "n_chunks" in merges.columns:
        conflicting = merges[merges["n_chunks"] > 0]
        agent_groups = [
            conflicting.loc[conflicting["agent"] == a, "n_chunks"].values
            for a in sorted(conflicting["agent"].dropna().unique())
        ]
        agent_labels = sorted(conflicting["agent"].dropna().unique().tolist())
        h, p = kruskal(*agent_groups)
        kw_chunks = {
            "H": round(float(h), 3), "p": float(p),
            "significant": _sig(p),
            "n_agents": len(agent_groups),
            "n_conflicting_merges": int(len(conflicting)),
        }
        buf.write(f"  Kruskal-Wallis H = {h:.3f}, p = {p:.2e}  {_sig(p)}\n")
        buf.write(f"  Medians per agent:\n")
        for a in agent_labels:
            vals = conflicting.loc[conflicting["agent"] == a, "n_chunks"]
            buf.write(f"    {a:<20}  n={len(vals):>6,}  median={vals.median():.1f}"
                      f"  p75={vals.quantile(.75):.1f}  p99={vals.quantile(.99):.1f}\n")
        kw_chunks["medians"] = {
            a: {
                "n": int(len(conflicting[conflicting["agent"] == a])),
                "median": float(conflicting[conflicting["agent"] == a]["n_chunks"].median()),
                "p75": float(conflicting[conflicting["agent"] == a]["n_chunks"].quantile(.75)),
                "p90": float(conflicting[conflicting["agent"] == a]["n_chunks"].quantile(.90)),
                "p99": float(conflicting[conflicting["agent"] == a]["n_chunks"].quantile(.99)),
            }
            for a in agent_labels
        }
    else:
        buf.write("  [SKIP] required columns missing\n")
    results["kruskal_wallis_chunks_per_merge"] = kw_chunks

    # 3. Post-hoc Dunn test
    _subsection("3. Post-hoc Dunn test (Bonferroni) for chunks per merge", buf)
    dunn_results: dict[str, Any] = {}
    try:
        import scikit_posthocs as sp
        if "agent" in merges.columns and "n_chunks" in merges.columns:
            conflicting = merges[merges["n_chunks"] > 0].dropna(subset=["agent"])
            dunn_p = sp.posthoc_dunn(conflicting, val_col="n_chunks",
                                     group_col="agent", p_adjust="bonferroni")
            agents_d = dunn_p.index.tolist()
            pairs = {}
            for a1, a2 in combinations(agents_d, 2):
                p_val = float(dunn_p.loc[a1, a2])
                pairs[f"{a1} vs {a2}"] = {"p_bonferroni": round(p_val, 4),
                                           "significant": _sig(p_val)}
            dunn_results["pairs"] = pairs
            buf.write("  (Bonferroni-corrected pairwise p-values)\n")
            for pair_key, info in pairs.items():
                buf.write(f"  {pair_key:<40} p={info['p_bonferroni']:.4f}  {info['significant']}\n")
    except ImportError:
        buf.write("  [SKIP] scikit-posthocs not installed. "
                  "Run: pip install scikit-posthocs\n")
        dunn_results["error"] = "scikit-posthocs not installed"
    except Exception as e:
        buf.write(f"  [ERROR] {e}\n")
        dunn_results["error"] = str(e)
    results["dunn_pairwise_chunks"] = dunn_results

    # 4. Cliff's delta for chunks per merge (pairwise)
    _subsection("4. Cliff's delta effect sizes for chunks per merge (pairwise)", buf)
    cliffs: dict[str, Any] = {}
    if "agent" in merges.columns and "n_chunks" in merges.columns:
        conflicting = merges[merges["n_chunks"] > 0]
        agent_labels = sorted(conflicting["agent"].dropna().unique().tolist())
        buf.write(f"  {'Agent 1':<20} {'Agent 2':<20} {'delta':>8} {'effect':>12}\n")
        for a1, a2 in combinations(agent_labels, 2):
            x = conflicting.loc[conflicting["agent"] == a1, "n_chunks"].values
            y = conflicting.loc[conflicting["agent"] == a2, "n_chunks"].values
            if len(x) < 5 or len(y) < 5:
                continue
            d = _cliffs_delta(x[:5000], y[:5000])   # cap for speed
            label = _delta_label(d)
            cliffs[f"{a1} vs {a2}"] = {"delta": round(d, 3), "label": label}
            buf.write(f"  {a1:<20} {a2:<20} {d:>8.3f} {label:>12}\n")
    results["cliffs_delta_chunks"] = cliffs

    # 5. Kruskal-Wallis for LOC distributions
    _subsection("5. Kruskal-Wallis: LOC distributions across agents", buf)
    kw_loc: dict[str, Any] = {}
    if "agent" in chunks.columns:
        for col in ["v1_loc", "v2_loc", "base_loc", "resolution_loc"]:
            if col not in chunks.columns:
                buf.write(f"  {col}: [SKIP] column not found\n")
                continue
            grps = [
                chunks.loc[chunks["agent"] == a, col].dropna().values
                for a in sorted(chunks["agent"].dropna().unique())
            ]
            grps = [g for g in grps if len(g) > 0]
            h, p = kruskal(*grps)
            kw_loc[col] = {"H": round(float(h), 3), "p": float(p),
                           "significant": _sig(p)}
            buf.write(f"  {col:<20}  H={h:.2f}  p={p:.2e}  {_sig(p)}\n")
    results["kruskal_wallis_loc_by_agent"] = kw_loc

    return results


# ── RQ2 tests ───────────────────────────────────────────────────────────────

def run_rq2(chunks: pd.DataFrame, buf: StringIO) -> dict:
    _section("RQ2 — Resolution Strategies", buf)
    results: dict[str, Any] = {}

    classifiable = chunks[chunks["strategy"] != "Imprecise"].copy()
    n_total   = len(chunks)
    n_classif = len(classifiable)
    n_imp     = n_total - n_classif

    buf.write(f"  Total chunks:    {n_total:,}\n")
    buf.write(f"  Classifiable:    {n_classif:,}  ({n_classif/n_total:.1%})\n")
    buf.write(f"  Imprecise:       {n_imp:,}  ({n_imp/n_total:.1%})\n")

    results["dataset"] = {
        "n_total_chunks": n_total,
        "n_classifiable": n_classif,
        "n_imprecise": n_imp,
        "imprecise_rate": round(n_imp / n_total, 4),
    }

    counts = (classifiable["strategy"]
              .value_counts()
              .reindex(STRATEGIES_CLASSIFIABLE, fill_value=0))

    # 1. Wilson CIs for global proportions
    _subsection("1. Wilson CIs: global strategy proportions", buf)
    wilson_rows = []
    buf.write(f"  {'Strategy':<12} {'Count':>8} {'Pct':>8} {'95% CI':>22}\n")
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
        buf.write(f"  {strat:<12} {cnt:>8,} {rate:>8.1%} [{lo:.1%}, {hi:.1%}]\n")
    results["wilson_ci_global_strategies"] = wilson_rows

    # 2. Descriptive comparison with Ghiotto et al. (no GoF test — see note)
    _subsection("2. Descriptive comparison with Ghiotto et al. baseline", buf)
    # NOTE: A chi-squared GoF test is NOT appropriate here for three reasons:
    #   (a) Non-independence: chunks within the same merge/repo are correlated,
    #       so the effective n is much smaller than the nominal 60k+.
    #   (b) Different populations: Ghiotto studied human-authored Java merges;
    #       our data is multi-language, AI-authored. The null hypothesis
    #       "our data was drawn from Ghiotto's distribution" is ill-posed.
    #   (c) Power inflation: at n~60k any deviation >0.1pp would be
    #       "significant", making the p-value uninformative.
    # Instead we report the absolute difference and whether Ghiotto's published
    # point estimate falls outside our 95% Wilson CI (a descriptive check only).
    ghiotto_props_arr = np.array([GHIOTTO_PROPS[s] for s in STRATEGIES_CLASSIFIABLE])
    ghiotto_props_arr /= ghiotto_props_arr.sum()
    observed_arr = counts.values.astype(float)
    comparison_rows = []
    buf.write(f"  {'Strategy':<10} {'Ours':>8} {'CI_lo':>8} {'CI_hi':>8}"
              f" {'Ghiotto':>8} {'Diff':>8} {'Ghiotto in CI?':>16}\n")
    for i, s in enumerate(STRATEGIES_CLASSIFIABLE):
        ours = float(observed_arr[i]) / n_classif
        gh   = float(ghiotto_props_arr[i])
        _, lo, hi = _wilson(int(observed_arr[i]), n_classif)
        in_ci = lo <= gh <= hi
        diff  = ours - gh
        comparison_rows.append({
            "strategy": s,
            "ours": round(ours, 4),
            "ci_lo_95": round(lo, 4),
            "ci_hi_95": round(hi, 4),
            "ghiotto": round(gh, 4),
            "abs_diff": round(diff, 4),
            "ghiotto_within_our_ci": bool(in_ci),
        })
        flag = "YES" if in_ci else "NO (differs)"
        buf.write(f"  {s:<10} {ours:>8.1%} {lo:>8.1%} {hi:>8.1%}"
                  f" {gh:>8.1%} {diff:>+8.1%} {flag:>16}\n")
    buf.write("\n  NOTE: 'Ghiotto in CI?' is a descriptive check, not a formal test.\n"
              "  The populations differ (Java/human vs multi-lang/AI); no GoF test\n"
              "  was run. Interpret differences substantively, not via p-values.\n")
    results["descriptive_comparison_ghiotto"] = {
        "note": ("GoF test omitted: different populations (Java/human vs multi-language/AI),"
                 " within-cluster dependence, and n~60k power inflation all invalidate it."
                 " Use absolute differences and Wilson CIs for substantive interpretation."),
        "ghiotto_source": "Ghiotto et al. TSE 2020, Table 13 (proportions normalised over 6 classifiable strategies)",
        "comparison": comparison_rows,
    }

    # 3. Chi-squared independence: strategy × agent
    _subsection("3. Chi-squared independence: strategy × agent", buf)
    chisq_strat_agent: dict[str, Any] = {}
    if "agent" in classifiable.columns:
        ct = (classifiable.groupby(["agent", "strategy"])
              .size().unstack(fill_value=0)
              .reindex(columns=STRATEGIES_CLASSIFIABLE, fill_value=0))
        buf.write(f"  Contingency table ({ct.shape[0]} agents × {ct.shape[1]} strategies):\n")
        buf.write(ct.to_string(max_cols=20) + "\n")
        chisq_strat_agent = _safe_chi2(ct.values)
        r, c = ct.shape
        V = _cramers_v(chisq_strat_agent["chi2"], n_classif, r, c)
        chisq_strat_agent["cramers_v"] = round(V, 4)
        chisq_strat_agent["cramers_v_label"] = _v_label(V)
        chisq_strat_agent["row_percentages"] = {
            agent: {
                s: round(float(ct.loc[agent, s]) / ct.loc[agent].sum(), 4)
                for s in STRATEGIES_CLASSIFIABLE
            }
            for agent in ct.index
        }
        buf.write(f"  chi2({chisq_strat_agent['dof']}) = {chisq_strat_agent['chi2']:.1f},"
                  f"  p = {chisq_strat_agent['p']:.2e}  {chisq_strat_agent['significant']}\n")
        buf.write(f"  Cramér's V = {V:.3f}  ({_v_label(V)})\n")
        buf.write("\n  Row % (classifiable only):\n")
        pct = ct.div(ct.sum(axis=1), axis=0)
        buf.write(pct.map(lambda x: f"{x:.1%}").to_string() + "\n")
    results["chisq_strategy_agent"] = chisq_strat_agent

    # 4. Imprecise rate × agent
    _subsection("4. Chi-squared: Imprecise rate × agent", buf)
    buf.write("  NOTE: p-value is anti-conservative due to within-merge/within-repo clustering.\n"
              "  Cramér's V (bias-corrected) is the primary evidence of effect magnitude;\n"
              "  it shares the independence assumption but is far less N-sensitive than p.\n")
    chisq_imp_agent: dict[str, Any] = {}
    if "agent" in chunks.columns:
        ct_imp = (
            chunks.assign(is_imp=(chunks["strategy"] == "Imprecise").astype(int))
            .groupby(["agent", "is_imp"]).size().unstack(fill_value=0)
            .rename(columns={0: "classifiable", 1: "imprecise"})
        )
        ct_imp["imprecise_rate"] = (
            ct_imp["imprecise"] / (ct_imp["classifiable"] + ct_imp["imprecise"])
        )
        ct_imp_vals = ct_imp[["classifiable", "imprecise"]].values
        chisq_imp_agent = _safe_chi2(ct_imp_vals)
        n_imp_total = int(ct_imp_vals.sum())
        r_imp, c_imp = ct_imp_vals.shape
        V_imp = _cramers_v(chisq_imp_agent["chi2"], n_imp_total, r_imp, c_imp)
        chisq_imp_agent["cramers_v"] = round(V_imp, 4)
        chisq_imp_agent["cramers_v_label"] = _v_label(V_imp)
        chisq_imp_agent["rates"] = {
            a: round(float(ct_imp.loc[a, "imprecise_rate"]), 4)
            for a in ct_imp.index
        }
        buf.write(f"  chi2({chisq_imp_agent['dof']}) = {chisq_imp_agent['chi2']:.1f},"
                  f"  p = {chisq_imp_agent['p']:.2e}  {chisq_imp_agent['significant']}\n")
        buf.write(f"  Cramér's V = {V_imp:.3f}  ({_v_label(V_imp)})  [primary effect-size metric]\n")
        buf.write("  Imprecise rates by agent:\n")
        for a, r in chisq_imp_agent["rates"].items():
            buf.write(f"    {a:<22}  {r:.1%}\n")
    results["chisq_imprecise_rate_by_agent"] = chisq_imp_agent

    # 5. Pairwise Holm-corrected chi-squared between agents
    _subsection("5. Pairwise Holm-corrected chi-squared (strategy × agent pairs)", buf)
    buf.write("  NOTE: At n~60k virtually all pairs will be 'significant' — p-values are\n"
              "  uninformative here. Primary output is the absolute pp difference per\n"
              "  strategy and Cramér's V for each pair. Use those for interpretation.\n"
              "  Clustering also inflates p (and V slightly), so treat V as an upper bound.\n")
    pairwise: dict[str, Any] = {}
    if "agent" in classifiable.columns:
        ct = (classifiable.groupby(["agent", "strategy"])
              .size().unstack(fill_value=0)
              .reindex(columns=STRATEGIES_CLASSIFIABLE, fill_value=0))
        # Proportion table (row %)
        prop = ct.div(ct.sum(axis=1), axis=0)
        agents_list = ct.index.tolist()
        pairs_list  = list(combinations(agents_list, 2))
        raw_ps = []
        cramers_vs = []
        for a1, a2 in pairs_list:
            sub = ct.loc[[a1, a2]].values
            n_pair = int(sub.sum())
            if n_pair < 10:
                raw_ps.append(1.0)
                cramers_vs.append(0.0)
            else:
                _, p_pair, _, _ = chi2_contingency(sub)
                raw_ps.append(p_pair)
                V_pair = _cramers_v(chi2_contingency(sub)[0], n_pair, 2, len(STRATEGIES_CLASSIFIABLE))
                cramers_vs.append(V_pair)
        _, corr_ps, _, _ = multipletests(raw_ps, method="holm")
        pairwise = {}
        buf.write(f"\n  {'Pair':<40} {'V':>6} {'effect':>10} {'p_holm':>10}\n")
        for i, (a1, a2) in enumerate(pairs_list):
            # Absolute pp differences per strategy
            pp_diffs = {s: round(float(prop.loc[a1, s] - prop.loc[a2, s]), 4)
                        for s in STRATEGIES_CLASSIFIABLE}
            V_pair = cramers_vs[i]
            pairwise[f"{a1} vs {a2}"] = {
                "p_raw": round(float(raw_ps[i]), 6),
                "p_holm": round(float(corr_ps[i]), 6),
                "significant": _sig(float(corr_ps[i])),
                "cramers_v": round(V_pair, 4),
                "cramers_v_label": _v_label(V_pair),
                "abs_pp_diff_by_strategy": pp_diffs,
            }
            buf.write(f"  {a1} vs {a2:<{40 - len(a1) - 4}} {V_pair:>6.3f} {_v_label(V_pair):>10}"
                      f" {float(corr_ps[i]):>10.4f}\n")
            buf.write(f"    pp diffs: " +
                      "  ".join(f"{s}:{pp_diffs[s]:+.1%}" for s in STRATEGIES_CLASSIFIABLE) + "\n")
    results["pairwise_holm_strategy_agent"] = pairwise

    return results


# ── RQ3 tests ───────────────────────────────────────────────────────────────

def run_rq3(chunks: pd.DataFrame, merges: pd.DataFrame, buf: StringIO) -> dict:
    _section("RQ3 — Resolver", buf)
    results: dict[str, Any] = {}

    RCOL = "resolver_type"
    if RCOL not in merges.columns:
        buf.write(f"  [SKIP] '{RCOL}' column not found in merges frame.\n")
        return {"error": f"column '{RCOL}' not in merges"}

    # 1. Wilson CIs for agent self-resolution rate per coding agent
    _subsection("1. Wilson CIs: agent self-resolution rate per coding agent", buf)
    ci_rows = []
    if "agent" in merges.columns:
        ag = (
            merges.assign(is_agent=(merges[RCOL] == "agent").astype(int))
            .groupby("agent")
            .agg(n_total=("is_agent", "count"), n_agent=("is_agent", "sum"))
            .reset_index()
        )
        buf.write(f"  {'Agent':<22} {'Rate':>8} {'95% CI':>22} {'n_agent/n_total':>18}\n")
        for _, row in ag.sort_values("n_agent", ascending=False).iterrows():
            rate, lo, hi = _wilson(int(row["n_agent"]), int(row["n_total"]))
            ci_rows.append({
                "agent": row["agent"],
                "n_total": int(row["n_total"]),
                "n_agent_resolved": int(row["n_agent"]),
                "rate": round(rate, 4),
                "ci_lo_95": round(lo, 4),
                "ci_hi_95": round(hi, 4),
            })
            buf.write(f"  {row['agent']:<22} {rate:>8.1%} [{lo:.1%}, {hi:.1%}]"
                      f"  {int(row['n_agent'])}/{int(row['n_total'])}\n")
    results["wilson_ci_self_resolution_rates"] = sorted(ci_rows, key=lambda r: -r["rate"])

    # 2. Chi-squared: resolver × agent
    _subsection("2. Chi-squared independence: resolver × agent", buf)
    buf.write("  NOTE: This tests at merge level (each merge one observation), so clustering\n"
              "  is less severe than chunk-level tests. Still, merges within the same repo\n"
              "  are correlated. Cramér's V (bias-corrected) is the primary effect measure.\n")
    chisq_res_agent: dict[str, Any] = {}
    if "agent" in merges.columns:
        ct = (merges.groupby(["agent", RCOL]).size()
              .unstack(fill_value=0))
        buf.write(ct.to_string() + "\n")
        chisq_res_agent = _safe_chi2(ct.values)
        n_merges_total = int(ct.values.sum())
        r_ra, c_ra = ct.shape
        V_ra = _cramers_v(chisq_res_agent["chi2"], n_merges_total, r_ra, c_ra)
        chisq_res_agent["cramers_v"] = round(V_ra, 4)
        chisq_res_agent["cramers_v_label"] = _v_label(V_ra)
        buf.write(f"  chi2({chisq_res_agent['dof']}) = {chisq_res_agent['chi2']:.1f},"
                  f"  p = {chisq_res_agent['p']:.2e}  {chisq_res_agent['significant']}\n")
        buf.write(f"  Cramér's V = {V_ra:.3f}  ({_v_label(V_ra)})  [primary effect-size metric]\n")
        chisq_res_agent["contingency_table"] = {
            str(idx): {str(col): int(ct.loc[idx, col]) for col in ct.columns}
            for idx in ct.index
        }
    results["chisq_resolver_agent"] = chisq_res_agent

    # 3. Wilson CIs for agent-assisted rate per coding agent
    # NOTE: Fisher's exact test was originally used here but was removed because the
    # "each agent vs all others" 2×2 framing conflates a between-agent comparison with
    # a within-agent rate estimate. The counts are too small (n_assisted ≈ 54 total)
    # to support an omnibus chi-squared AND the asymmetry of the 2×2 cells varies by
    # agent size, making p-values non-comparable across rows.
    # Wilson CIs are the correct summary: they directly quantify the uncertainty
    # around each agent's assisted rate without requiring a distributional assumption.
    _subsection("3. Wilson CIs: agent-assisted rate per coding agent (replaces Fisher exact)", buf)
    buf.write("  NOTE: Fisher exact was dropped — 'each agent vs all others' framing\n"
              "  is asymmetric (cell sizes differ by agent), making p-values\n"
              "  non-comparable. Wilson CIs give a cleaner per-agent summary.\n")
    fisher_rows = []
    if "agent" in merges.columns:
        ag_asst = (
            merges.assign(is_asst=(merges[RCOL] == "agent-assisted").astype(int))
            .groupby("agent")
            .agg(n_total=("is_asst", "count"), n_asst=("is_asst", "sum"))
            .reset_index()
        )
        buf.write(f"  {'Agent':<22} {'Rate':>8} {'95% CI':>22} {'n_asst/n_total':>16}\n")
        for _, row in ag_asst.iterrows():
            n_a  = int(row["n_asst"])
            rate, lo, hi = _wilson(n_a, int(row["n_total"]))
            fisher_rows.append({
                "agent": row["agent"],
                "n_assisted": n_a,
                "n_total": int(row["n_total"]),
                "rate": round(rate, 4),
                "ci_lo_95": round(lo, 4),
                "ci_hi_95": round(hi, 4),
            })
            buf.write(f"  {row['agent']:<22} {rate:>8.1%} [{lo:.1%}, {hi:.1%}]"
                      f"  {n_a}/{int(row['n_total'])}\n")
    else:
        buf.write("  'agent' column missing from merges frame.\n")
    results["wilson_ci_agent_assisted"] = fisher_rows

    # 4. Chi-squared: strategy × resolver  (the main test)
    _subsection("4. Chi-squared independence: strategy × resolver (MAIN TEST)", buf)
    chisq_strat_res: dict[str, Any] = {}
    if RCOL in chunks.columns and "strategy" in chunks.columns:
        classif = chunks[chunks["strategy"] != "Imprecise"]
        classif_ah = classif[classif[RCOL].isin(["agent", "human"])]
        ct = (classif_ah.groupby([RCOL, "strategy"]).size()
              .unstack(fill_value=0)
              .reindex(columns=STRATEGIES_CLASSIFIABLE, fill_value=0))
        buf.write(ct.to_string() + "\n")
        buf.write("\nRow percentages:\n")
        pct = ct.div(ct.sum(axis=1), axis=0)
        buf.write(pct.map(lambda x: f"{x:.1%}").to_string() + "\n")

        chisq_strat_res = _safe_chi2(ct.values)
        r, c = ct.shape
        V = _cramers_v(chisq_strat_res["chi2"], int(classif_ah.shape[0]), r, c)
        chisq_strat_res["cramers_v"] = round(V, 4)
        chisq_strat_res["cramers_v_label"] = _v_label(V)
        chisq_strat_res["row_percentages"] = {
            resolver: {
                s: round(float(ct.loc[resolver, s]) / ct.loc[resolver].sum(), 4)
                for s in STRATEGIES_CLASSIFIABLE
            }
            for resolver in ct.index
        }
        chisq_strat_res["n_agent_classifiable"]  = int((classif_ah[RCOL] == "agent").sum())
        chisq_strat_res["n_human_classifiable"]  = int((classif_ah[RCOL] == "human").sum())

        buf.write(f"\n  chi2({chisq_strat_res['dof']}) = {chisq_strat_res['chi2']:.1f},"
                  f"  p = {chisq_strat_res['p']:.2e}  {chisq_strat_res['significant']}\n")
        buf.write(f"  Cramér's V = {V:.3f}  ({_v_label(V)})\n")
    else:
        buf.write(f"  [SKIP] '{RCOL}' not found in chunks frame — need to join with merges.\n")
    results["chisq_strategy_resolver"] = chisq_strat_res

    # 5. Wilson CIs per strategy within each resolver type
    _subsection("5. Wilson CIs: per-strategy proportions within each resolver", buf)
    ci_per_resolver: dict[str, list] = {}
    if RCOL in chunks.columns:
        classif = chunks[chunks["strategy"] != "Imprecise"]
        for resolver in ["agent", "human"]:
            sub = classif[classif[RCOL] == resolver]
            n_sub = len(sub)
            if n_sub == 0:
                continue
            rows = []
            buf.write(f"\n  {resolver.upper()} (n={n_sub:,}):\n")
            buf.write(f"  {'Strategy':<10} {'Count':>8} {'Pct':>8} {'95% CI':>22}\n")
            for strat in STRATEGIES_CLASSIFIABLE:
                cnt = int((sub["strategy"] == strat).sum())
                rate, lo, hi = _wilson(cnt, n_sub)
                rows.append({
                    "strategy": strat, "count": cnt,
                    "proportion": round(rate, 4),
                    "ci_lo_95": round(lo, 4), "ci_hi_95": round(hi, 4),
                })
                buf.write(f"  {strat:<10} {cnt:>8,} {rate:>8.1%} [{lo:.1%}, {hi:.1%}]\n")
            ci_per_resolver[resolver] = rows
    results["wilson_ci_per_resolver"] = ci_per_resolver

    # 6. Imprecise rate by resolver
    _subsection("6. Imprecise rate by resolver type", buf)
    imp_by_resolver: dict[str, Any] = {}
    if RCOL in chunks.columns:
        imp_ct = (
            chunks.assign(is_imp=(chunks["strategy"] == "Imprecise").astype(int))
            .groupby([RCOL, "is_imp"]).size().unstack(fill_value=0)
            .rename(columns={0: "classifiable", 1: "imprecise"})
        )
        imp_ct["imprecise_rate"] = (
            imp_ct["imprecise"] / (imp_ct["classifiable"] + imp_ct["imprecise"])
        )
        imp_by_resolver = {
            resolver: {
                "n_classifiable": int(imp_ct.loc[resolver, "classifiable"]),
                "n_imprecise":    int(imp_ct.loc[resolver, "imprecise"]),
                "imprecise_rate": round(float(imp_ct.loc[resolver, "imprecise_rate"]), 4),
            }
            for resolver in imp_ct.index
            if resolver in imp_ct.index
        }
        for resolver, info in imp_by_resolver.items():
            buf.write(f"  {resolver:<20}  imprecise_rate={info['imprecise_rate']:.1%}"
                      f"  ({info['n_imprecise']:,}/{info['n_imprecise']+info['n_classifiable']:,})\n")
    results["imprecise_rate_by_resolver"] = imp_by_resolver

    # 7. Sensitivity analysis: Imprecise bias
    _subsection("7. Sensitivity analysis: Imprecise bias (V1 rate robustness check)", buf)
    sensitivity: dict[str, Any] = {}
    if RCOL in chunks.columns:
        ah = chunks[chunks[RCOL].isin(["agent", "human"])].copy()
        scenarios = [
            ("classifiable_only",       None),         # baseline
            ("upper_imprecise_as_V1",   "V1"),
            ("lower_imprecise_as_NC",   "NC"),
            ("neutral_imprecise_as_NN", "NN"),
        ]
        buf.write(f"  {'Scenario':<30} {'Agent V1%':>12} {'Human V1%':>12} {'Ratio':>8}\n")
        buf.write(f"  {'-'*66}\n")
        sensitivity_rows = []
        for label, imp_label in scenarios:
            if imp_label is None:
                sub = ah[ah["strategy"] != "Imprecise"]
            else:
                sub = ah.copy()
                sub["strategy"] = sub["strategy"].cat.add_categories([imp_label]) \
                    if imp_label not in sub["strategy"].cat.categories else sub["strategy"]
                sub.loc[sub["strategy"] == "Imprecise", "strategy"] = imp_label
            a_v1 = (sub[sub[RCOL] == "agent"]["strategy"] == "V1").mean()
            h_v1 = (sub[sub[RCOL] == "human"]["strategy"] == "V1").mean()
            ratio = a_v1 / h_v1 if h_v1 > 0 else float("inf")
            sensitivity_rows.append({
                "scenario": label,
                "agent_v1_rate": round(float(a_v1), 4),
                "human_v1_rate": round(float(h_v1), 4),
                "ratio": round(float(ratio), 3),
            })
            buf.write(f"  {label:<30} {a_v1:>11.1%} {h_v1:>11.1%} {ratio:>7.2f}x\n")
        sensitivity["scenarios"] = sensitivity_rows
        sensitivity["interpretation"] = (
            "If agent V1 rate remains substantially above human V1 rate across all scenarios,"
            " the finding is robust to the Imprecise localization bias."
        )
    results["sensitivity_imprecise_bias"] = sensitivity

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    buf = StringIO()
    buf.write("STATISTICAL TEST RESULTS\n")
    buf.write("Generated by analysis/run_all_statistical_tests.py\n")
    buf.write(f"Dataset: {PROJECT_ROOT}\n")

    try:
        chunks, merges = load_data()
    except Exception as e:
        print(f"ERROR loading data: {e}")
        traceback.print_exc()
        sys.exit(1)

    all_results: dict[str, Any] = {}

    print("\n── RQ1 tests...")
    try:
        all_results["rq1"] = run_rq1(chunks, merges, buf)
    except Exception as e:
        print(f"  ERROR in RQ1: {e}")
        traceback.print_exc()
        all_results["rq1"] = {"error": str(e)}

    print("\n── RQ2 tests...")
    try:
        all_results["rq2"] = run_rq2(chunks, buf)
    except Exception as e:
        print(f"  ERROR in RQ2: {e}")
        traceback.print_exc()
        all_results["rq2"] = {"error": str(e)}

    print("\n── RQ3 tests...")
    try:
        all_results["rq3"] = run_rq3(chunks, merges, buf)
    except Exception as e:
        print(f"  ERROR in RQ3: {e}")
        traceback.print_exc()
        all_results["rq3"] = {"error": str(e)}

    # Save JSON
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON saved to: {JSON_OUT}")

    # Save text report
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Text report saved to: {TXT_OUT}")

    # Also print the text report to stdout
    print("\n" + "="*70)
    print(buf.getvalue())


if __name__ == "__main__":
    main()
