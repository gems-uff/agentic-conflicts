#!/usr/bin/env python3
"""
main3_extra_stats.py - Computa as estatisticas que faltam para a versao
restruturada do paper (main3.tex). Usa diretamente os helpers do projeto
(analysis/common.py) para nao depender de edicao manual de schema.

Saida:
  results_main3.json   (envie este arquivo de volta)
  + resumo legivel impresso na tela

Uso:
  Coloque este arquivo na raiz do projeto agentic_conflicts
  (ao lado de analysis/, data/, paper/) e rode:

      python3 paper/main3_extra_stats.py

  ou, se preferir rodar de outro lugar:

      PYTHONPATH=/caminho/para/agentic_conflicts python3 main3_extra_stats.py

Requer: pandas, numpy, scipy, pyarrow (mesmas deps das notebooks de analise).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Bootstrap: garante que o pacote `analysis` deste projeto esteja importavel.
# O script tenta primeiro o diretorio que o contem; se nao achar, sobe um
# nivel (para o caso de ele ter sido copiado para paper/).
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent):
    if (candidate / "analysis" / "common.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        PROJECT_ROOT = candidate
        break
else:
    sys.exit(
        "ERRO: nao encontrei analysis/common.py. Coloque este script "
        "na raiz do projeto (ao lado de analysis/) ou em paper/."
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
OUT_PATH = PROJECT_ROOT / "results_main3.json"

CLASSIFIABLE = ["V1", "V2", "CC", "CB", "NC", "NN"]

# Ghiotto et al. 2020, exatamente como reportado em paper/main2.tex (Tabela 4).
GHIOTTO = {"V1": 0.500, "V2": 0.250, "CC": 0.030, "CB": 0.090, "NC": 0.130, "NN": 0.000}

# Os cinco agentes que nos interessam (alinha com o paper).
AGENTS_OF_INTEREST = ["claude_code", "codex", "copilot", "cursor", "devin"]


# ---------------------------------------------------------------------------
# Helpers estatisticos
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def cliffs_delta(x, y, max_n: int = 4000, seed: int = 0) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = rng.choice(x, max_n, replace=False)
    if len(y) > max_n:
        y = rng.choice(y, max_n, replace=False)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float(np.sign(x[:, None] - y[None, :]).mean())


def cramers_v(ct: pd.DataFrame):
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    v = float(np.sqrt((chi2 / n) / max(1, min(k - 1, r - 1))))
    return v, float(chi2), int(dof), float(p)


def quantiles(s: pd.Series, qs=(0.5, 0.75, 0.9, 0.99)) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {f"p{int(q * 100)}": None for q in qs}
    return {f"p{int(q * 100)}": float(s.quantile(q)) for q in qs}


def strategy_dist(sub_chunks: pd.DataFrame) -> tuple[int, dict]:
    """Distribui cada chunk em V1/V2/CC/CB/NC/NN com Wilson CI."""
    classifiable = sub_chunks[sub_chunks["strategy"].astype(str).isin(CLASSIFIABLE)]
    n = len(classifiable)
    out = {}
    for s in CLASSIFIABLE:
        k = int((classifiable["strategy"].astype(str) == s).sum())
        lo, hi = wilson(k, n)
        out[s] = {"n": k, "pct": (k / n if n else 0.0), "ci_low": lo, "ci_high": hi}
    return n, out


def safe_mannwhitney(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"U": float(u), "p": float(p), "cliffs_delta": cliffs_delta(a, b)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def postponed_share(sub_chunks: pd.DataFrame) -> float | None:
    """Postponed eh um label *raw* que common.py dobra em Imprecise.
    Le da coluna strategy_raw (criada por _canonicalize_strategy)."""
    if "strategy_raw" not in sub_chunks.columns or len(sub_chunks) == 0:
        return None
    return float((sub_chunks["strategy_raw"].astype(str) == "Postponed").mean())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Project root:  {PROJECT_ROOT}")
    print("Loading tables ...")
    tables = load_tables()
    print("Building chunk frame ...")
    chunks = build_chunk_frame(tables)
    print("Building merge frame ...")
    merges = build_merge_frame(tables)
    print(f"  chunks: {len(chunks):,}    merges: {len(merges):,}")

    if "has_conflict" in merges.columns:
        conflicting = merges[merges["has_conflict"].astype(bool)]
    else:
        conflicting = merges[merges["n_chunks"] > 0]
    print(f"  conflicting merges: {len(conflicting):,}")

    if "resolver_type" not in chunks.columns:
        sys.exit("ERRO: chunks sem coluna resolver_type apos o join. Reveja internal_merges.")
    if "agent" not in chunks.columns:
        sys.exit("ERRO: chunks sem coluna agent apos o join. Reveja universe.")

    ag_c = chunks[chunks["resolver_type"] == "agent"]
    hu_c = chunks[chunks["resolver_type"] == "human"]
    ag_m = conflicting[conflicting["resolver_type"] == "agent"]
    hu_m = conflicting[conflicting["resolver_type"] == "human"]

    results: dict = {
        "_meta": {
            "n_chunks_total":         int(len(chunks)),
            "n_merges_total":         int(len(merges)),
            "n_conflicting_merges":   int(len(conflicting)),
            "resolver_levels":        sorted(map(str, chunks["resolver_type"].dropna().unique())),
            "agent_levels":           sorted(map(str, chunks["agent"].dropna().unique())),
            "strategy_levels":        sorted(map(str, chunks["strategy"].astype(str).dropna().unique())),
            "strategy_raw_levels":    (sorted(map(str, chunks["strategy_raw"].dropna().unique()))
                                       if "strategy_raw" in chunks.columns else None),
        }
    }

    # ===== [A] RQ2: estrutura por resolver (agregado) ======================
    print("\n[A] structural by resolver ...")
    A: dict = {}
    for r in conflicting["resolver_type"].dropna().unique():
        m_sub = conflicting[conflicting["resolver_type"] == r]
        c_sub = chunks[chunks["resolver_type"] == r]
        A[str(r)] = {
            "n_merges":              int(len(m_sub)),
            "n_chunks":              int(len(c_sub)),
            "chunks_per_merge":      quantiles(m_sub["n_chunks"]),
            "chunks_per_merge_mean": float(m_sub["n_chunks"].mean()) if len(m_sub) else None,
            "chunks_per_merge_max":  float(m_sub["n_chunks"].max()) if len(m_sub) else None,
            "v1_loc":                quantiles(c_sub["v1_loc"]),
            "v2_loc":                quantiles(c_sub["v2_loc"]),
            "res_loc":               quantiles(c_sub["resolution_loc"]),
        }
    A["_mannwhitney_chunks_per_merge_agent_vs_human"] = safe_mannwhitney(
        ag_m["n_chunks"], hu_m["n_chunks"]
    )
    for col, lab in [("v1_loc", "v1_loc"), ("v2_loc", "v2_loc"), ("resolution_loc", "res_loc")]:
        A[f"_mannwhitney_{lab}_agent_vs_human"] = safe_mannwhitney(
            ag_c[col].dropna(), hu_c[col].dropna()
        )
    results["A_rq2_structural_by_resolver"] = A

    # ===== [B] RQ2: estrutura por agente (subset agent-resolved) ===========
    print("[B] structural per-agent self-resolved ...")
    B: dict = {}
    for ag in ag_m["agent"].dropna().unique():
        m_sub = ag_m[ag_m["agent"] == ag]
        c_sub = ag_c[ag_c["agent"] == ag]
        B[str(ag)] = {
            "n_self_resolved_merges": int(len(m_sub)),
            "n_self_resolved_chunks": int(len(c_sub)),
            "chunks_per_merge":       quantiles(m_sub["n_chunks"]),
            "v1_loc":                 quantiles(c_sub["v1_loc"]),
            "v2_loc":                 quantiles(c_sub["v2_loc"]),
            "res_loc":                quantiles(c_sub["resolution_loc"]),
        }
    results["B_rq2_structural_per_agent_self_resolved"] = B

    # ===== [C] RQ3: distribuicao de estrategia por resolver ================
    print("[C] strategy by resolver ...")
    C: dict = {}
    for r in chunks["resolver_type"].dropna().unique():
        sub = chunks[chunks["resolver_type"] == r]
        n_class, dist = strategy_dist(sub)
        C[str(r)] = {
            "n_total":         int(len(sub)),
            "n_classifiable":  int(n_class),
            "imprecise_share": float((sub["strategy"].astype(str) == "Imprecise").mean()),
            "postponed_share": postponed_share(sub),
            "distribution":    dist,
        }
    classif = chunks[chunks["strategy"].astype(str).isin(CLASSIFIABLE)]
    ct = pd.crosstab(classif["resolver_type"], classif["strategy"].astype(str))
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        v, chi2, dof, p = cramers_v(ct)
        C["_chi2_strategy_x_resolver"] = {"chi2": chi2, "dof": dof, "p": p, "cramers_v": v}
    results["C_rq3_strategy_by_resolver"] = C

    # ===== [D] RQ3: estrategia per-agente RESTRITA ao agent-resolved =======
    print("[D] strategy per-agent within self-resolved ...")
    D: dict = {}
    for ag in ag_c["agent"].dropna().unique():
        sub = ag_c[ag_c["agent"] == ag]
        n_class, dist = strategy_dist(sub)
        D[str(ag)] = {
            "n_total":         int(len(sub)),
            "n_classifiable":  int(n_class),
            "imprecise_share": float((sub["strategy"].astype(str) == "Imprecise").mean()),
            "postponed_share": postponed_share(sub),
            "distribution":    dist,
        }
    self_classif = ag_c[ag_c["strategy"].astype(str).isin(CLASSIFIABLE)]
    if self_classif["agent"].nunique() >= 2:
        ct = pd.crosstab(self_classif["agent"], self_classif["strategy"].astype(str))
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            v, chi2, dof, p = cramers_v(ct)
            D["_chi2_strategy_x_agent_within_self_resolved"] = {
                "chi2": chi2, "dof": dof, "p": p, "cramers_v": v
            }
    results["D_rq3_strategy_per_agent_self_resolved"] = D

    # ===== [E] Validacao: humanos do corpus vs Ghiotto =====================
    print("[E] validation human-in-corpus vs Ghiotto ...")
    E: dict = {"ghiotto_reference": GHIOTTO}
    hu_classif = hu_c[hu_c["strategy"].astype(str).isin(CLASSIFIABLE)]
    n = len(hu_classif)
    E["n_human_classifiable"] = int(n)
    dist_check = {}
    for s in CLASSIFIABLE:
        k = int((hu_classif["strategy"].astype(str) == s).sum())
        pct = k / n if n else 0.0
        lo, hi = wilson(k, n)
        g = GHIOTTO.get(s)
        in_ci = (g is not None) and (lo <= g <= hi)
        dist_check[s] = {
            "k":                  k,
            "pct":                pct,
            "ci_low":             lo,
            "ci_high":            hi,
            "ghiotto":            g,
            "ghiotto_in_our_ci":  in_ci,
            "delta_pp":           (pct - g) * 100 if g is not None else None,
        }
    E["dist_humans_in_corpus_vs_ghiotto"] = dist_check
    results["E_validation_humans_vs_ghiotto"] = E

    # ===== [F] Robustez: estratificacao por complexidade ===================
    print("[F] stratified agent-vs-human by complexity ...")
    F: dict = {}
    nconf_lookup = (conflicting.set_index(["repo_full_name", "merge_sha"])["n_chunks"]).to_dict()
    keys = list(zip(chunks["repo_full_name"], chunks["merge_sha"]))
    chunks_aug = chunks.assign(_nconf=[nconf_lookup.get(k) for k in keys])
    classif_with_nconf = chunks_aug[
        chunks_aug["strategy"].astype(str).isin(CLASSIFIABLE) & chunks_aug["_nconf"].notna()
    ]
    bins = [(1, 1), (2, 2), (3, 4), (5, 10), (11, 10**9)]
    for lo, hi in bins:
        label = f"{lo}-{'inf' if hi == 10**9 else hi}_chunks"
        bucket = classif_with_nconf[
            (classif_with_nconf["_nconf"] >= lo) & (classif_with_nconf["_nconf"] <= hi)
        ]
        h = bucket[bucket["resolver_type"] == "human"]
        a = bucket[bucket["resolver_type"] == "agent"]

        def pct(df, s):
            return float((df["strategy"].astype(str) == s).mean()) if len(df) else None

        F[label] = {
            "n_human":      int(len(h)),
            "n_agent":      int(len(a)),
            "human_v1_pct": pct(h, "V1"), "agent_v1_pct": pct(a, "V1"),
            "human_v2_pct": pct(h, "V2"), "agent_v2_pct": pct(a, "V2"),
            "human_nc_pct": pct(h, "NC"), "agent_nc_pct": pct(a, "NC"),
            "human_cb_pct": pct(h, "CB"), "agent_cb_pct": pct(a, "CB"),
            "human_cc_pct": pct(h, "CC"), "agent_cc_pct": pct(a, "CC"),
        }
    results["F_stratified_agent_vs_human_by_complexity"] = F

    # ===== Salvar e resumir ================================================
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"OK. Resultados salvos em {OUT_PATH}")
    print("=" * 72)

    print("\n[E] Humanos do corpus vs Ghiotto:")
    print("    (CI = Wilson 95%; FORA = Ghiotto fora do nosso CI)")
    for s, d in results["E_validation_humans_vs_ghiotto"]["dist_humans_in_corpus_vs_ghiotto"].items():
        flag = "OK  " if d["ghiotto_in_our_ci"] else "FORA"
        g = (d["ghiotto"] or 0) * 100
        print(f"  {s:3s}  ours={d['pct']*100:6.2f}%  CI=[{d['ci_low']*100:5.2f},{d['ci_high']*100:5.2f}]"
              f"  ghiotto={g:5.2f}%  delta={d['delta_pp']:+6.2f}pp  [{flag}]")

    print("\n[A] Chunks-per-merge por resolver:")
    for r, d in results["A_rq2_structural_by_resolver"].items():
        if r.startswith("_"):
            continue
        m = d["chunks_per_merge"]
        print(f"  {r:16s}  n_merges={d['n_merges']:6d}  med={m['p50']}"
              f"  p75={m['p75']}  p90={m['p90']}  p99={m['p99']}")

    print("\n[D] Estrategias per-agente restritas a agent-resolved:")
    for ag, d in results["D_rq3_strategy_per_agent_self_resolved"].items():
        if ag.startswith("_"):
            continue
        v1 = d["distribution"]["V1"]
        nc = d["distribution"]["NC"]
        pp = d["postponed_share"]
        pp_str = f"{pp*100:5.2f}%" if pp is not None else "  n/a"
        print(f"  {ag:14s}  n_class={d['n_classifiable']:6d}"
              f"  V1={v1['pct']*100:5.2f}% [{v1['ci_low']*100:5.2f},{v1['ci_high']*100:5.2f}]"
              f"  NC={nc['pct']*100:5.2f}%  Imprecise={d['imprecise_share']*100:5.2f}%"
              f"  Postponed={pp_str}")

    chi = results["D_rq3_strategy_per_agent_self_resolved"].get(
        "_chi2_strategy_x_agent_within_self_resolved")
    if chi:
        print(f"  --> chi2={chi['chi2']:.1f}  dof={chi['dof']}  p={chi['p']:.3g}"
              f"  Cramer's V={chi['cramers_v']:.3f}")

    print("\n[F] V1% agente vs humano por faixa de complexidade (chunks-per-merge):")
    for label, d in results["F_stratified_agent_vs_human_by_complexity"].items():
        if d["human_v1_pct"] is None or d["agent_v1_pct"] is None:
            continue
        delta = d["agent_v1_pct"] * 100 - d["human_v1_pct"] * 100
        print(f"  {label:22s}  n_h={d['n_human']:6d}  n_a={d['n_agent']:5d}"
              f"  V1: human={d['human_v1_pct']*100:5.2f}%  agent={d['agent_v1_pct']*100:5.2f}%"
              f"  delta={delta:+6.2f}pp")


    # ============================================================
    # Agent-assisted bucket: strategy distribution and host-agent breakdown
    # ============================================================
    print("\n" + "=" * 60)
    print("AGENT-ASSISTED BUCKET")
    print("=" * 60)
    
    # Whatever resolver_type isn't "agent" or "human" (likely "agent_assisted")
    assisted = chunks[~chunks["resolver_type"].isin(["agent", "human"])].copy()
    classifiable_strats = ["V1", "V2", "CC", "CB", "NC", "NN"]
    
    n_total = len(assisted)
    n_class = int(assisted["strategy"].isin(classifiable_strats).sum())
    
    print(f"resolver_type values in this bucket: "
          f"{sorted(assisted['resolver_type'].dropna().unique())}")
    print(f"Total chunks:                {n_total}")
    print(f"Classifiable (non-Imprecise): {n_class}")
    print(f"Imprecise share:             "
          f"{(n_total - n_class) / n_total * 100:.2f}%" if n_total else "n/a")
    
    # Strategy distribution over classifiable chunks
    clf = assisted[assisted["strategy"].isin(classifiable_strats)]
    counts = clf["strategy"].value_counts().reindex(classifiable_strats, fill_value=0)
    pcts = (counts / max(len(clf), 1) * 100).round(2)
    
    print("\nStrategy distribution (classifiable, percent of n_class):")
    for s in classifiable_strats:
        print(f"  {s:>2}: n={int(counts[s]):>5d}  pct={pcts[s]:>6.2f}%")
    
    # Per host agent (which agent's PRs were these assisted merges on)
    print("\nPer host agent (classifiable):")
    for ag, sub in clf.groupby("agent"):
        if pd.isna(ag):
            continue
        sub_counts = sub["strategy"].value_counts().reindex(classifiable_strats, fill_value=0)
        sub_pcts = (sub_counts / len(sub) * 100).round(1)
        dist = " ".join(f"{s}={sub_pcts[s]:.1f}" for s in classifiable_strats)
        print(f"  {ag:<14}  n={len(sub):>4d}  {dist}")
    
    # Sanity check: aggregate + humans + assisted should reconstruct total classifiable
    n_agent = int((chunks["resolver_type"] == "agent")
                  & chunks["strategy"].isin(classifiable_strats)).sum() \
        if False else \
        int(chunks[(chunks["resolver_type"] == "agent")
                   & chunks["strategy"].isin(classifiable_strats)].shape[0])
    n_human = int(chunks[(chunks["resolver_type"] == "human")
                         & chunks["strategy"].isin(classifiable_strats)].shape[0])
    print(f"\nReconciliation (classifiable):")
    print(f"  agent (autonomous) : {n_agent}")
    print(f"  agent-assisted     : {n_class}")
    print(f"  human              : {n_human}")
    print(f"  TOTAL              : {n_agent + n_class + n_human}")
    print(f"  expected (Tab. 1)  : 75752")

    print("\nEnvie de volta o arquivo:", OUT_PATH)


if __name__ == "__main__":
    main()
