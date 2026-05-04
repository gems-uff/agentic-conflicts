#!/usr/bin/env python3
"""
main3_codex_deepdive.py - Analise detalhada do efeito Codex e robustez do
contraste humano-vs-agente para o paper main3.

Saida (na raiz do projeto, em analysis/results/):
  results_main3_codex.json          (numerico, mande de volta)
  results_main3_codex_samples.jsonl (chunks amostrados para inspecao manual)

Uso (a partir da raiz do projeto):
  python3 analysis/main3_codex_deepdive.py

Requer as mesmas dependencias do projeto: pandas, numpy, scipy, pyarrow.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Bootstrap: localiza a raiz do projeto (que contem analysis/common.py).
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent, HERE):
    if (candidate / "analysis" / "common.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        PROJECT_ROOT = candidate
        break
else:
    sys.exit(
        "ERRO: nao encontrei analysis/common.py. Coloque este script em "
        "analysis/ ou ajuste o PYTHONPATH."
    )

from analysis.common import (  # noqa: E402
    load_tables,
    build_chunk_frame,
    build_merge_frame,
    STRATEGY_ORDER,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "analysis" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "results_main3_codex.json"
SAMPLES_PATH = OUT_DIR / "results_main3_codex_samples.jsonl"

CLASSIFIABLE = ["V1", "V2", "CC", "CB", "NC", "NN"]

LOCK_BASENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "npm-shrinkwrap.json",
    "pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "gemfile.lock",
    "composer.lock",
    "cargo.lock",
    "go.sum",
    "mix.lock",
    "flake.lock",
    "pubspec.lock",
    "podfile.lock",
    "bun.lockb",
}

GENERATED_PATH_RE = re.compile(
    r"(^|/)(dist|build|out|node_modules|vendor|target|coverage|"
    r"__pycache__|\.next|\.nuxt|\.cache|\.parcel-cache)/",
    re.IGNORECASE,
)

GENERATED_EXT_SUFFIXES = (".min.js", ".min.css", ".bundle.js", ".map", ".snap")


def classify_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        return "other"
    p = path.lower()
    base = p.rsplit("/", 1)[-1]
    if base in LOCK_BASENAMES:
        return "lock"
    if any(p.endswith(s) for s in GENERATED_EXT_SUFFIXES):
        return "generated"
    if GENERATED_PATH_RE.search("/" + p):
        return "generated"
    if p.endswith((".md", ".rst", ".txt", ".adoc")):
        return "doc"
    if p.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env")):
        return "config"
    if p.endswith(".sql") or "/migrations/" in p or "/migration/" in p:
        return "migration"
    return "source"


def file_extension(path: str) -> str:
    if not isinstance(path, str) or not path:
        return "(empty)"
    base = path.rsplit("/", 1)[-1]
    if "." not in base:
        return "(no-ext)"
    return "." + base.rsplit(".", 1)[-1].lower()


# ---------------------------------------------------------------------------
# Helpers estatisticos
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    halfwidth = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def cramers_v_table(table: pd.DataFrame) -> tuple[float, float, int, float]:
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.values.sum())
    r, k = table.shape
    denom = max(1, min(r, k) - 1)
    V = float(np.sqrt(chi2 / (n * denom))) if n > 0 else 0.0
    return float(chi2), float(p), int(dof), V


def strategy_distribution(chunks_subset: pd.DataFrame) -> dict:
    n_total = int(len(chunks_subset))
    counts = chunks_subset["strategy"].value_counts()
    classifiable = {s: int(counts.get(s, 0)) for s in CLASSIFIABLE}
    n_classifiable = sum(classifiable.values())
    out = {}
    for s, k in classifiable.items():
        pct = (k / n_classifiable) if n_classifiable else 0.0
        ci_low, ci_high = wilson(k, n_classifiable)
        out[s] = {"n": k, "pct": pct, "ci_low": ci_low, "ci_high": ci_high}
    return {
        "n_total": n_total,
        "n_classifiable": n_classifiable,
        "imprecise_share": (n_total - n_classifiable) / n_total if n_total else 0.0,
        "distribution": out,
    }


def contrast_human_vs_agent(chunks_subset: pd.DataFrame) -> dict | None:
    """Cramer's V e distribuicao para human vs (qualquer) agent dentro do subset."""
    h = chunks_subset[chunks_subset["resolver_type"] == "human"]
    a = chunks_subset[chunks_subset["resolver_type"] == "agent"]
    h_clf = h[h["strategy"].isin(CLASSIFIABLE)]
    a_clf = a[a["strategy"].isin(CLASSIFIABLE)]
    if len(h_clf) == 0 or len(a_clf) == 0:
        return None
    table = pd.crosstab(
        pd.Categorical(
            ["human"] * len(h_clf) + ["agent"] * len(a_clf),
            categories=["human", "agent"],
        ),
        pd.Categorical(
            list(h_clf["strategy"]) + list(a_clf["strategy"]),
            categories=CLASSIFIABLE,
        ),
    )
    chi2, p, dof, V = cramers_v_table(table)
    return {
        "n_human_total": int(len(h)),
        "n_agent_total": int(len(a)),
        "n_human_classifiable": int(len(h_clf)),
        "n_agent_classifiable": int(len(a_clf)),
        "chi2": chi2,
        "dof": dof,
        "p": p,
        "cramers_v": V,
        "human_distribution": strategy_distribution(h)["distribution"],
        "agent_distribution": strategy_distribution(a)["distribution"],
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def truncate(s, n=300):
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", "").replace("\n", "\\n").replace("\t", "    ")
    return s[:n] + ("..." if len(s) > n else "")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
print(">> Loading tables ...", flush=True)
tables = load_tables()
print(">> Building frames ...", flush=True)
chunks = build_chunk_frame(tables)
merges = build_merge_frame(tables)
imprecise = ['Imprecise', 'Postponed']
# Filter by only resolved chunks
chunks=chunks[~chunks['strategy'].isin(imprecise)]

# Filtra para chunks com resolver conhecido e merge_sha valido
chunks = chunks[chunks["resolver_type"].notna() & chunks["merge_sha"].notna()].copy()

# Anota categoria de path e extensao
chunks["path_category"] = chunks["file_path"].fillna("").map(classify_path)
chunks["file_ext"] = chunks["file_path"].fillna("").map(file_extension)
chunks["chunk_one_line"] = (
    (chunks["v1_loc"].fillna(0) == 1) & (chunks["v2_loc"].fillna(0) == 1)
)

results = {
    "_meta": {
        "n_chunks_total": int(len(chunks)),
        "n_merges_total": int(len(merges)),
        "resolver_levels": sorted(map(str, chunks["resolver_type"].dropna().unique())),
        "agent_levels": sorted(map(str, chunks["agent"].dropna().unique())),
        "path_category_levels": sorted(map(str, chunks["path_category"].dropna().unique())),
    }
}

# ============================================================
# G. Per-merge anomaly profile (self-resolved by each agent)
# ============================================================
print(">> [G] per-merge anomaly profile ...", flush=True)
self_resolved = chunks[chunks["resolver_type"] == "agent"].copy()

G = {}
for agent, sub in self_resolved.groupby("agent"):
    if pd.isna(agent):
        continue
    by_merge = by_merge = sub.groupby(["repo_full_name", "merge_sha"]).agg(
        n_chunks=("chunk_index", "count"),
        v1_pct=("strategy", lambda s: float((s == "V1").mean())),
        mean_v1_loc=("v1_loc", "mean"),
        mean_v2_loc=("v2_loc", "mean"),
        mean_res_loc=("resolution_loc", "mean"),
        frac_one_line=("chunk_one_line", "mean"),
        frac_lock=("path_category", lambda s: float((s == "lock").mean())),
        frac_generated=("path_category", lambda s: float((s == "generated").mean())),
    )
    G[str(agent)] = {
        "n_merges": int(len(by_merge)),
        "n_chunks": int(len(sub)),
        "merge_quantiles": {
            col: {f"p{int(q*100)}": float(by_merge[col].quantile(q))
                  for q in (0.5, 0.75, 0.9, 0.99)}
            for col in ["n_chunks", "v1_pct", "mean_v1_loc", "mean_v2_loc",
                       "frac_one_line", "frac_lock", "frac_generated"]
        },
        "merge_means": {
            col: float(by_merge[col].mean())
            for col in ["n_chunks", "v1_pct", "mean_v1_loc", "mean_v2_loc",
                       "frac_one_line", "frac_lock", "frac_generated"]
        },
    }
results["G_per_merge_anomaly_profile"] = G

# ============================================================
# H. File-path / extension breakdown
# ============================================================
print(">> [H] path & extension breakdown ...", flush=True)


def path_breakdown(sub: pd.DataFrame) -> dict | None:
    if len(sub) == 0:
        return None
    cat_counts = sub["path_category"].value_counts()
    cat_pct = (cat_counts / len(sub) * 100).round(3).to_dict()
    top_ext = sub["file_ext"].value_counts().head(15).to_dict()
    top_basename = (
        sub["file_path"]
        .fillna("")
        .map(lambda p: p.rsplit("/", 1)[-1] if isinstance(p, str) else "")
        .value_counts()
        .head(15)
        .to_dict()
    )
    sxc = pd.crosstab(sub["strategy"], sub["path_category"])
    return {
        "n_chunks": int(len(sub)),
        "category_counts": {str(k): int(v) for k, v in cat_counts.to_dict().items()},
        "category_pct": {str(k): float(v) for k, v in cat_pct.items()},
        "top_extensions": {str(k): int(v) for k, v in top_ext.items()},
        "top_basenames": {str(k): int(v) for k, v in top_basename.items()},
        "strategy_by_category": {
            str(strat): {str(cat): int(sxc.loc[strat, cat])
                         for cat in sxc.columns}
            for strat in sxc.index
        },
    }


H = {"by_agent_self_resolved": {}}
for agent, sub in self_resolved.groupby("agent"):
    if pd.isna(agent):
        continue
    H["by_agent_self_resolved"][str(agent)] = path_breakdown(sub)
H["humans"] = path_breakdown(chunks[chunks["resolver_type"] == "human"])
results["H_path_breakdown"] = H

# ============================================================
# I. Robustness: o efeito human-vs-agent persiste sob exclusoes?
# ============================================================
print(">> [I] robustness contrasts ...", flush=True)
I_out = {"full": contrast_human_vs_agent(chunks)}

# 1) sem Codex no lado agent
mask_no_codex = ~(
    (chunks["resolver_type"] == "agent") & (chunks["agent"] == "OpenAI_Codex")
)
I_out["exclude_codex_self_resolved"] = contrast_human_vs_agent(chunks[mask_no_codex])

# 2) sem merges "bulk" (>95 chunks ~ p99 humano)
merge_sizes = chunks.groupby("merge_sha").size()
big_merges = set(merge_sizes[merge_sizes > 95].index)
I_out["exclude_bulk_merges_gt95"] = contrast_human_vs_agent(
    chunks[~chunks["merge_sha"].isin(big_merges)]
)

# 3) sem lock + generated
mask_real = ~chunks["path_category"].isin(["lock", "generated"])
I_out["exclude_lock_and_generated"] = contrast_human_vs_agent(chunks[mask_real])

# 4) sem chunks de 1 linha em ambos os lados (v1_loc=1 e v2_loc=1)
I_out["exclude_one_line_chunks"] = contrast_human_vs_agent(
    chunks[~chunks["chunk_one_line"]]
)

# 5) combinacoes
I_out["exclude_codex_and_lock_generated"] = contrast_human_vs_agent(
    chunks[mask_no_codex & mask_real]
)
I_out["exclude_codex_bulk_lock_generated"] = contrast_human_vs_agent(
    chunks[mask_no_codex & mask_real & ~chunks["merge_sha"].isin(big_merges)]
)

# 6) cada agente isolado contra todos os humanos
for agent in sorted(self_resolved["agent"].dropna().unique()):
    mask = (chunks["resolver_type"] == "human") | (
        (chunks["resolver_type"] == "agent") & (chunks["agent"] == agent)
    )
    I_out[f"per_agent_vs_human::{agent}"] = contrast_human_vs_agent(chunks[mask])

results["I_robustness_contrasts"] = I_out

# ============================================================
# J. Pairwise agent x agent strategy contrasts (Bonferroni)
# ============================================================
print(">> [J] pairwise agent comparisons ...", flush=True)
agents = sorted(self_resolved["agent"].dropna().unique())
J = {}
pair_keys = []
for i in range(len(agents)):
    for j in range(i + 1, len(agents)):
        a1, a2 = agents[i], agents[j]
        s1 = self_resolved[
            (self_resolved["agent"] == a1) & self_resolved["strategy"].isin(CLASSIFIABLE)
        ]
        s2 = self_resolved[
            (self_resolved["agent"] == a2) & self_resolved["strategy"].isin(CLASSIFIABLE)
        ]
        key = f"{a1}_vs_{a2}"
        if len(s1) < 10 or len(s2) < 10:
            J[key] = {"skipped": "low_n", "n_a1": int(len(s1)), "n_a2": int(len(s2))}
            continue
        table = pd.crosstab(
            pd.Categorical([a1] * len(s1) + [a2] * len(s2), categories=[a1, a2]),
            pd.Categorical(
                list(s1["strategy"]) + list(s2["strategy"]),
                categories=CLASSIFIABLE,
            ),
        )
        chi2, p, dof, V = cramers_v_table(table)
        J[key] = {
            "n_a1": int(len(s1)),
            "n_a2": int(len(s2)),
            "chi2": chi2,
            "dof": dof,
            "p_raw": p,
            "cramers_v": V,
        }
        pair_keys.append((key, p))

n_pairs = max(1, len(pair_keys))
for key, p in pair_keys:
    J[key]["p_bonferroni"] = float(min(1.0, p * n_pairs))
    J[key]["significant_bonferroni_05"] = bool(J[key]["p_bonferroni"] < 0.05)

results["J_pairwise_agent_strategy_contrasts"] = J

# ============================================================
# K. Codex merge enumeration
# ============================================================
print(">> [K] enumerating Codex self-resolved merges ...", flush=True)
codex = self_resolved[self_resolved["agent"] == "OpenAI_Codex"]


def _safe_mode(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return ""
    m = s.mode()
    if m.empty:
        return ""
    return str(m.iloc[0])


if len(codex) > 0:
    codex_merges = (
        codex.groupby(["repo_full_name", "merge_sha"])
        .agg(
            n_chunks=("chunk_index", "count"),
            v1_pct=("strategy", lambda s: float((s == "V1").mean())),
            mean_v1_loc=("v1_loc", "mean"),
            mean_v2_loc=("v2_loc", "mean"),
            frac_one_line=("chunk_one_line", "mean"),
            frac_lock=("path_category", lambda s: float((s == "lock").mean())),
            frac_generated=("path_category", lambda s: float((s == "generated").mean())),
            dominant_path=("file_path", _safe_mode),
            dominant_category=("path_category", _safe_mode),
        )
        .reset_index()
    )
    results["K_codex_merge_table"] = codex_merges.to_dict(orient="records")
else:
    results["K_codex_merge_table"] = []

# ============================================================
# K2. Repo concentration per self-resolving agent
# ============================================================
print(">> [K2] repo concentration per agent ...", flush=True)
K2 = {}
for agent, sub in self_resolved.groupby("agent"):
    if pd.isna(agent):
        continue
    # By chunk
    chunks_by_repo = sub["repo_full_name"].value_counts()
    # By merge (one row per (repo, merge_sha))
    merges_by_repo = (
    sub.drop_duplicates(subset=["repo_full_name", "merge_sha"])["repo_full_name"].value_counts()
    )
    n_chunks = int(chunks_by_repo.sum())
    n_merges = int(merges_by_repo.sum())
    K2[str(agent)] = {
        "n_chunks": n_chunks,
        "n_merges": n_merges,
        "n_unique_repos": int(len(merges_by_repo)),
        "top1_repo": str(merges_by_repo.index[0]) if len(merges_by_repo) else "",
        "top1_share_chunks": float(chunks_by_repo.iloc[0] / n_chunks) if n_chunks else 0.0,
        "top1_share_merges": float(merges_by_repo.iloc[0] / n_merges) if n_merges else 0.0,
        "top3_share_merges": float(merges_by_repo.iloc[:3].sum() / n_merges) if n_merges else 0.0,
        "top5_repos_by_merges": {
            str(r): int(c) for r, c in merges_by_repo.head(5).items()
        },
    }
results["K2_repo_concentration_per_agent"] = K2

# ============================================================
# L. Humans resolving by host agent (e o conflito-agent-original?)
# ============================================================
print(">> [L] humans resolving by host agent ...", flush=True)
human_chunks = chunks[chunks["resolver_type"] == "human"]
L = {}
for host_agent, sub in human_chunks.groupby("agent"):
    if pd.isna(host_agent):
        continue
    L[str(host_agent)] = strategy_distribution(sub)
results["L_humans_resolving_by_host_agent"] = L

# ============================================================
# M. Sample chunks for qualitative inspection (separate file)
# ============================================================
print(">> [M] sampling chunks for qualitative inspection ...", flush=True)
sample_rows = []
for (agent, strat), grp in self_resolved.groupby(["agent", "strategy"]):
    if pd.isna(agent):
        continue
    take = grp.sample(min(5, len(grp)), random_state=7)
    for _, row in take.iterrows():
        sample_rows.append({
            "agent": str(agent),
            "strategy": str(strat),
            "repo_full_name": str(row.get("repo_full_name", "")),
            "merge_sha": str(row.get("merge_sha", "")),
            "file_path": str(row.get("file_path", "")),
            "path_category": str(row.get("path_category", "")),
            "chunk_index": int(row.get("chunk_index", -1))
                if pd.notna(row.get("chunk_index", -1)) else -1,
            "v1_loc": int(row.get("v1_loc", 0))
                if pd.notna(row.get("v1_loc", 0)) else 0,
            "v2_loc": int(row.get("v2_loc", 0))
                if pd.notna(row.get("v2_loc", 0)) else 0,
            "resolution_loc": int(row.get("resolution_loc", 0))
                if pd.notna(row.get("resolution_loc", 0)) else 0,
            "v1": truncate(row.get("v1", "")),
            "base": truncate(row.get("base", "")),
            "v2": truncate(row.get("v2", "")),
            "resolution": truncate(row.get("resolution", "")),
        })

with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
    for r in sample_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ============================================================
# Save & summary
# ============================================================
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(to_jsonable(results), f, indent=2, ensure_ascii=False)

print(f"\n>> Wrote: {OUT_PATH}")
print(f">> Wrote: {SAMPLES_PATH}")

# Quick stdout summary
print("\n=== ROBUSTNESS: human-vs-agent Cramér's V under exclusions ===")
for key, r in results["I_robustness_contrasts"].items():
    if not isinstance(r, dict) or "cramers_v" not in r:
        continue
    print(f"  {key:48s}  V={r['cramers_v']:.3f}  "
          f"n_h={r['n_human_classifiable']:>7}  n_a={r['n_agent_classifiable']:>7}")

print("\n=== CODEX PROFILE (per-merge medians) ===")
codex_g = G.get("OpenAI_Codex", {})
if codex_g:
    q = codex_g.get("merge_quantiles", {})
    print(f"  n_merges        = {codex_g['n_merges']}")
    print(f"  n_chunks        = {codex_g['n_chunks']}")
    print(f"  n_chunks p50    = {q.get('n_chunks', {}).get('p50')}")
    print(f"  v1_pct  p50     = {q.get('v1_pct', {}).get('p50')}")
    print(f"  frac_one_line p50 = {q.get('frac_one_line', {}).get('p50')}")
    print(f"  frac_lock p50   = {q.get('frac_lock', {}).get('p50')}")
    print(f"  frac_generated p50 = {q.get('frac_generated', {}).get('p50')}")

print("\n=== PAIRWISE AGENT V (Bonferroni) ===")
for key, r in results["J_pairwise_agent_strategy_contrasts"].items():
    if "skipped" in r:
        print(f"  {key:35s}  SKIPPED ({r['skipped']})")
        continue
    flag = "*" if r.get("significant_bonferroni_05") else " "
    print(f"  {key:35s}  V={r['cramers_v']:.3f}  p_bonf={r['p_bonferroni']:.2e}  {flag}")

print("\nDone.")
