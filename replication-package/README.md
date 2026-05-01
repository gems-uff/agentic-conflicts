# Replication Package: How AI Coding Agents Resolve Merge Conflicts

An empirical study analyzing merge conflict resolution strategies in pull requests authored by AI coding agents.

## Study Summary

This replication package contains the extraction and analysis pipeline for an empirical study of how AI coding agents (Claude Code, Codex, GitHub Copilot, Cursor, Devin) resolve merge conflicts. The study examines 14,960 conflicting internal merge commits across 57,582 repositories, finding that 96.1% of conflicts are resolved by humans despite agent authorship, with a 60× spread in per-agent self-resolution rates.

## Repository Structure

```
replication-package/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (pinned versions)
├── launch_pipeline.py                 # Single entry point: full pipeline or analysis-only
├── src/
│   ├── mining_utils.py               # Git/conflict extraction utilities
│   ├── strategies_utils.py           # Conflict resolution strategy classification
│   └── analysis_utils.py             # Shared analysis helpers
├── analysis/
│   ├── common.py                     # AnalysisTables, shared utilities
│   ├── dataset_characterization.py   # Dataset Overview section (scope, counts)
│   ├── rq1_resolver.py              # RQ1 analysis: who resolves (agent vs. human)
│   ├── rq2_strategies.py            # RQ2 analysis: how agents resolve (strategy distribution)
│   └── plotting.py                   # Shared figure generation
├── figures/
│   └── pipelinev2.pdf               # Pipeline architecture diagram
├── data/
│   └── README.md                    # Data download instructions (Zenodo/FigShare)
└── supplementary/
    ├── README.md                    # Overview of supplementary analyses
    ├── by_language.md              # Language-level stratification results
    └── by_task_type.md             # Task-type stratification results
```

## Requirements

- **Python:** 3.10 or later
- **OS:** Linux or macOS (signal-based timeouts; Windows adaptation possible)
- **Git:** ≥ 2.38 (for `git merge-tree`)
- **Disk space:** ~50 GB for bare repository clones (mining stage)

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/gems-uff/agentic-conflicts.git
cd replication-package

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

### Option A: Full Pipeline (Server, 5-7 days)

Requires the AIDev dataset (private, available by request) and sufficient compute resources.

```bash
python launch_pipeline.py \
  --aidev-dir /path/to/AIDev \
  --data-dir ./data \
  --workers 32
```

**Stages executed:**
1. Build universe DataFrame from AIDev parquet files
2. Mine internal merge commits (per-repository, parallel)
3. Aggregate incremental JSONL intermediates → final parquet files
4. Dataset characterization (counts, scope table)
5. RQ1 analysis: resolver attribution (agent vs. human)
6. RQ2 analysis: resolution strategies (global, per-agent, heterogeneity)
7. Generate all figures

### Option B: Pilot Run (Local, 30-60 minutes)

Quick validation on a subset of repositories:

```bash
python launch_pipeline.py \
  --aidev-dir /path/to/AIDev \
  --data-dir ./data \
  --pilot 10 \
  --workers 4
```

### Option C: Analysis Only (Local, 5-10 minutes)

Requires pre-downloaded parquet files from Zenodo/FigShare:

```bash
# First, download data (see data/README.md for links)
python launch_pipeline.py --analyze-only --data-dir ./data
```

## Expected Outputs

After running the pipeline, outputs will be in `./data/`:

**Parquet files (data stage):**
- `universe.parquet` — Agent PR metadata (57,582 repos)
- `internal_merges.parquet` — All internal merge commits (14,960 conflicts)
- `conflict_chunks.parquet` — Individual conflict chunks
- `classified_chunks.parquet` — Chunks with assigned resolution strategies
- `resolver_labels.parquet` — Resolver attribution (agent vs. human)

**CSV & figures (analysis stage):**
- `results/dataset_overview/` — Scope table, conflict counts
- `results/rq1/` — RQ1 tables: resolver attribution
- `results/rq2/` — RQ2 tables: strategy distributions
- `results_summary.json` — All headline numbers for the paper

## Supplementary Material

Additional analyses not shown in the main paper:

- **`supplementary/by_language.md`** — Conflict patterns across programming languages
- **`supplementary/by_task_type.md`** — Patterns by PR task type (feature, bug fix, refactor, etc.)

These are generated automatically during the analysis stage.

## Citation

If you use this replication package, please cite:

```bibtex
@inproceedings{camposjunior2026merge,
  title={How AI Coding Agents Resolve Merge Conflicts: An Empirical Study},
  author={Campos Junior, Heleno de Souza and Murta, Leonardo Gresta Paulino},
  booktitle={Proceedings of the 40th Brazilian Symposium on Software Engineering (SBES)},
  year={2026}
}
```

## License & Attribution

- **Code:** MIT License
- **Data:** CC-BY-4.0 (available from Zenodo/FigShare upon publication)

## Contact & Questions

For questions about this replication package or the study, please open an issue on GitHub.
