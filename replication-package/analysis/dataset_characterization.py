"""Dataset characterization analysis for the replication package."""

import logging
from pathlib import Path
import pandas as pd
from .common import AnalysisTables, build_chunk_frame, build_merge_frame


def _export_results(results: dict, output_dir: str, prefix: str = "dataset_overview"):
    """Export analysis results to CSV and TXT formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export as TXT (human-readable)
    txt_file = output_dir / f"{prefix}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("Dataset Characterization\n")
        f.write("=" * 50 + "\n\n")
        for key, value in results.items():
            if not isinstance(value, dict):
                f.write(f"{key}: {value}\n")
        f.write("\n")

        # Strategy distribution as table
        if 'strategy_distribution' in results and isinstance(results['strategy_distribution'], dict):
            f.write("Strategy Distribution:\n")
            for strategy, count in results['strategy_distribution'].items():
                f.write(f"  {strategy}: {count}\n")

    logging.info(f"Exported results to {txt_file}")

    # Export key metrics as CSV
    csv_file = output_dir / f"{prefix}_summary.csv"
    summary_df = pd.DataFrame([results])
    summary_df.to_csv(csv_file, index=False)
    logging.info(f"Exported summary to {csv_file}")


def analyze_dataset(tables: AnalysisTables, output_dir: str = './results') -> dict:
    """Analyze and characterize the dataset."""
    logging.info("\nDataset Characterization")
    logging.info("=" * 50)

    results = {}

    universe = tables.universe
    if universe is None:
        logging.error("ERROR: universe table not found")
        return results

    universe_count = len(universe)
    results['universe_pairs'] = universe_count
    logging.info(f"\nUniverse: {universe_count:,} (PR, SHA) pairs")

    repo_count = universe['full_name'].nunique() if 'full_name' in universe.columns else 0
    results['unique_repos'] = repo_count
    logging.info(f"Repositories: {repo_count:,}")

    pr_count = universe['pr_id'].nunique()
    results['unique_prs'] = pr_count
    logging.info(f"AI PRs: {pr_count:,}")

    internal_merges = build_merge_frame(tables)
    num_merges = len(internal_merges) if internal_merges is not None else 0
    results['internal_merges'] = num_merges
    logging.info(f"Internal Merge Commits: {num_merges:,}")

    classified = build_chunk_frame(tables)
    num_conflicts = len(classified) if classified is not None else 0
    results['conflict_chunks'] = num_conflicts
    logging.info(f"Conflict Chunks: {num_conflicts:,}")

    if internal_merges is not None and 'resolver_type' in internal_merges.columns:
        logging.info("\nResolver Attribution:")
        resolver_dist = internal_merges['resolver_type'].value_counts()
        for resolver_type, count in resolver_dist.items():
            pct = count / num_merges * 100
            logging.info(f"  {resolver_type}: {count:,} ({pct:.1f}%)")
            results[f'resolver_{resolver_type}'] = count

    if classified is not None and 'strategy' in classified.columns:
        logging.info("\nStrategy Distribution:")
        strategy_dist = classified['strategy'].value_counts()
        results['strategy_distribution'] = strategy_dist.to_dict()
        for strategy, count in strategy_dist.items():
            pct = count / num_conflicts * 100
            logging.info(f"  {strategy}: {count:,} ({pct:.1f}%)")

    logging.info("=" * 50)

    # Export results
    _export_results(results, output_dir, prefix="dataset_overview")

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './data'
    tables = AnalysisTables(data_dir)
    analyze_dataset(tables, output_dir=f'{data_dir}/results')
