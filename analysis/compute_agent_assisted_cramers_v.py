#!/usr/bin/env python3
"""
Compute Cramér's V for agent-assisted vs humans and agent vs agent-assisted comparisons.
Uses the analysis.common infrastructure and saves results to analysis/results/.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from scipy.stats import chi2_contingency
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from analysis.common import load_tables, build_chunk_frame

def cramers_v_bc(ct):
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = ct.sum()
    r, k = ct.shape
    phi2 = chi2 / n
    phi2_tilde = max(0, phi2 - (k-1)*(r-1)/(n-1))
    k_tilde = k - (k-1)**2/(n-1)
    r_tilde = r - (r-1)**2/(n-1)
    return np.sqrt(phi2_tilde / (min(k_tilde, r_tilde) - 1))

def main():
    logger.info("Loading analysis data...")
    tables = load_tables()
    chunks = build_chunk_frame(tables)

    # Filter imprecise and ensure valid data
    imprecise = ['Imprecise', 'Postponed']
    chunks = chunks[~chunks['strategy'].isin(imprecise)]
    chunks = chunks[chunks["resolver_type"].notna() & chunks["merge_sha"].notna()].copy()

    # Get strategy distributions
    def get_dist(df, rtype):
        sub = df[df["resolver_type"] == rtype]
        # Use only the 5 main strategies (exclude NN which has zero in assisted)
        strategies = ['V1', 'V2', 'CC', 'CB', 'NC']
        return sub["strategy"].value_counts().reindex(strategies, fill_value=0)

    human    = get_dist(chunks, "human")
    agent    = get_dist(chunks, "agent")
    assisted = get_dist(chunks, "agent-assisted")

    logger.info(f"human:    {human.sum():,} chunks")
    logger.info(f"agent:    {agent.sum():,} chunks")
    logger.info(f"assisted: {assisted.sum():,} chunks")

    # Compute pairwise Cramér's V
    strategies = ['V1', 'V2', 'CC', 'CB', 'NC']

    pairs = [
        ("agent", "human", agent, human),
        ("assisted", "human", assisted, human),
        ("agent", "assisted", agent, assisted),
    ]

    results = {
        "metadata": {
            "n_human": int(human.sum()),
            "n_agent": int(agent.sum()),
            "n_assisted": int(assisted.sum()),
            "strategies_used": strategies,
            "method": "Bias-corrected Cramér's V (Bergsma 2013)",
        },
        "agent_assisted_distribution": {
            s: {
                "n": int(assisted[s]),
                "pct": float(assisted[s] / assisted.sum() * 100)
            }
            for s in strategies
        },
        "cramers_v": {}
    }

    for a_name, b_name, a_dist, b_dist in pairs:
        ct = np.array([
            [a_dist[s] for s in strategies],
            [b_dist[s] for s in strategies]
        ])
        v = cramers_v_bc(ct)
        key = f"{a_name}_vs_{b_name}"
        results["cramers_v"][key] = float(v)
        logger.info(f"Cramér's V ({a_name} vs {b_name}): {v:.4f}")

    # Save to JSON
    output_dir = Path("./analysis/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "agent_assisted_cramers_v.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("AGENT-ASSISTED ANALYSIS: Cramér's V Results")
    print("="*80)
    print(f"\nAgent-assisted distribution (n={assisted.sum():,}):")
    for s in strategies:
        pct = assisted[s] / assisted.sum() * 100
        print(f"  {s:3s}: {assisted[s]:5,}  ({pct:5.1f}%)")

    print(f"\nPairwise Cramér's V comparisons:")
    for key, v in results["cramers_v"].items():
        print(f"  {key:30s}: V = {v:.4f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()