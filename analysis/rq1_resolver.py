"""RQ1 Analysis: Who Resolves Merge Conflicts?"""

import logging
from pathlib import Path
import pandas as pd
from .common import AnalysisTables, build_merge_frame


def _export_results(results: dict, output_dir: str):
    """Export RQ1 results to CSV and TXT formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export as TXT (human-readable)
    txt_file = output_dir / "rq1_resolver_attribution.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("RQ1: Who Resolves Merge Conflicts?\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Merges: {results.get('total_merges', 0)}\n\n")
        f.write("Resolver Attribution:\n")
        for key, value in results.items():
            if key.startswith('resolver_') and key.endswith('_pct'):
                resolver_type = key.replace('resolver_', '').replace('_pct', '')
                count = results.get(f'resolver_{resolver_type}', 0)
                f.write(f"  {resolver_type}: {count} ({value:.1f}%)\n")

    logging.info(f"Exported results to {txt_file}")

    # Export as CSV
    csv_file = output_dir / "rq1_resolver_attribution.csv"
    rows = []
    for key, value in results.items():
        if key.startswith('resolver_') and not key.endswith('_pct'):
            resolver_type = key.replace('resolver_', '')
            rows.append({
                'resolver_type': resolver_type,
                'count': value,
                'percentage': results.get(f'{key}_pct', 0)
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logging.info(f"Exported summary to {csv_file}")


def analyze_rq1(tables: AnalysisTables, output_dir: str = './results') -> dict:
    """Analyze RQ1: Who resolves merge conflicts (agent vs. human)?"""
    logging.info("\nRQ1: Who Resolves Merge Conflicts?")
    logging.info("=" * 50)

    results = {}
    merges = build_merge_frame(tables)
    if merges is None or merges.empty:
        logging.warning("WARNING: No internal merges available for RQ1 analysis")
        logging.info("=" * 50)
        # Export empty results
        _export_results(results, output_dir)
        return results

    if 'n_chunks' in merges.columns:
        merges = merges[merges['n_chunks'] > 0]

    total = len(merges)
    results['total_merges'] = total

    if 'resolver_type' in merges.columns:
        logging.info(f"\nResolver Attribution:")
        resolver_dist = merges['resolver_type'].value_counts()
        for resolver_type, count in resolver_dist.items():
            pct = count / total * 100
            logging.info(f"  {resolver_type}: {count:,} ({pct:.1f}%)")
            results[f'resolver_{resolver_type}'] = count
            results[f'resolver_{resolver_type}_pct'] = pct
    else:
        logging.warning("WARNING: resolver_type column not found in internal_merges")

    logging.info("=" * 50)

    # Export results
    _export_results(results, output_dir)

    return results


if __name__ == '__main__':
    import sys
    from .common import load_tables
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './data'
    tables = load_tables(data_dir)
    analyze_rq1(tables, output_dir=f'{data_dir}/results')
