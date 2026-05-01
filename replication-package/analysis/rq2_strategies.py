"""RQ2 Analysis: How Do Agents Resolve Conflicts?"""

import logging
from pathlib import Path
import pandas as pd
from .common import AnalysisTables, STRATEGY_ORDER


def _export_results(results: dict, output_dir: str):
    """Export RQ2 results to CSV and TXT formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export as TXT (human-readable)
    txt_file = output_dir / "rq2_strategy_distribution.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("RQ2: How Do Agents Resolve Merge Conflicts?\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Chunks: {results.get('total_chunks', 0)}\n\n")
        f.write("Strategy Distribution:\n")
        for strategy in STRATEGY_ORDER:
            key = f'strategy_{strategy}'
            count = results.get(key, 0)
            pct = results.get(f'{key}_pct', 0)
            if count > 0:
                f.write(f"  {strategy}: {count} ({pct:.1f}%)\n")

    logging.info(f"Exported results to {txt_file}")

    # Export as CSV
    csv_file = output_dir / "rq2_strategy_distribution.csv"
    rows = []
    for strategy in STRATEGY_ORDER:
        key = f'strategy_{strategy}'
        count = results.get(key, 0)
        if count > 0:
            rows.append({
                'strategy': strategy,
                'count': count,
                'percentage': results.get(f'{key}_pct', 0)
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logging.info(f"Exported summary to {csv_file}")


def analyze_rq2(tables: AnalysisTables, output_dir: str = './results') -> dict:
    """Analyze RQ2: How do agents resolve merge conflicts (strategies)?"""
    logging.info("\nRQ2: How Do Agents Resolve Merge Conflicts?")
    logging.info("=" * 50)

    results = {}
    classified = tables.classified_chunks
    if classified is None or classified.empty:
        logging.warning("WARNING: No classified chunks available for RQ2 analysis")
        logging.info("=" * 50)
        # Export empty results
        _export_results(results, output_dir)
        return results

    if 'strategy' not in classified.columns:
        logging.warning("WARNING: strategy column not found in classified_chunks")
        logging.info("=" * 50)
        _export_results(results, output_dir)
        return results

    logging.info(f"\nOverall Strategy Distribution:")
    strategy_dist = classified['strategy'].value_counts()
    total = len(classified)
    results['total_chunks'] = total

    for strategy in STRATEGY_ORDER:
        if strategy in strategy_dist.index:
            count = strategy_dist[strategy]
            pct = count / total * 100
            logging.info(f"  {strategy}: {count:,} ({pct:.1f}%)")
            results[f'strategy_{strategy}'] = count
            results[f'strategy_{strategy}_pct'] = pct

    logging.info("=" * 50)

    # Export results
    _export_results(results, output_dir)

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './data'
    tables = AnalysisTables(data_dir)
    analyze_rq2(tables, output_dir=f'{data_dir}/results')
