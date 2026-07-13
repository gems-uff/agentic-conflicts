# Replication Package: How AI Coding Agents Resolve Merge Conflicts

An empirical study analyzing merge conflict resolution strategies in pull requests authored by AI coding agents.

**Paper:** *How AI Coding Agents Resolve Merge Conflicts: An Empirical Study* — accepted at the 40th Brazilian Symposium on Software Engineering (SBES 2026).
<!-- TODO: replace with the link to the published article / preprint PDF once available. -->

## Study Summary

This replication package contains the extraction and analysis pipeline for an empirical study of how AI coding agents (Claude Code, Codex, GitHub Copilot, Cursor, Devin) resolve merge conflicts. The study examines 14,960 conflicting internal merge commits across 57,582 repositories, finding that 96.1% of conflicts are resolved by humans despite agent authorship, with a 60× spread in per-agent self-resolution rates.

## Repository Structure

```
.
├── README.md              # This file
├── LICENSE                # MIT (code); data is CC BY 4.0 (see License below)
├── requirements.txt       # Python dependencies (minimum versions)
├── launch_pipeline.py     # Single entry point: full pipeline or analysis-only
├── clean_pipeline.py/.sh  # Utilities to reset intermediate outputs
├── src/                   # Mining and shared utilities
├── analysis/              # RQ1/RQ2 analyses, statistics, plotting
└── data/
    └── README.md          # Data download instructions (figshare DOI)
```

## Requirements

- **Python:** 3.10 or later
- **OS:** Linux or macOS for the full mining pipeline (uses POSIX signal-based timeouts); the analysis-only path additionally runs on Windows.
- **Git:** ≥ 2.38 (for `git merge-tree`)
- **Disk space:** ~50 GB for bare repository clones (full mining stage); ~5 GB for the analysis-only path (downloaded data).
- **RAM:** ≥ 32 GB recommended for the analysis stage — the `resolved_chunks` and `classified_chunks` tables expand to > 5 GB each in memory when loaded in full.
- **Dependencies:** see [`requirements.txt`](requirements.txt).

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/gems-uff/agentic-conflicts.git
cd agentic-conflicts

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Verifying the Installation

The quickest end-to-end check runs the analysis stage against the published data and confirms the paper's headline numbers.

```bash
# 1. Download the data (~3.9 GB; see data/README.md for details)
#    figshare DOI: 10.6084/m9.figshare.32159415
# 2. Run the analysis-only pipeline
python launch_pipeline.py --analyze-only --data-dir ./data
```

**Expected output** (paper headline results):

- **14,960** conflicting internal merges · **121,599** conflict chunks
- **RQ1** (who resolves): **96.1%** human · **3.1%** agent · **0.8%** agent-assisted
- Per-agent self-resolution rate: **0.5%** (Codex) → **29.4%** (Devin)
- Postponed chunks: **0.76%** (agent) vs **9.85%** (human), over classifiable chunks

Results are written to `data/results/`. On machines with less than ~32 GB of RAM, read the parquet tables with column projection (only the columns you need) to avoid loading the large conflict-text columns.

## Running the Pipeline

### Option A: Full Pipeline

Requires the [AIDev dataset](https://huggingface.co/datasets/hao-li/AIDev) and sufficient compute resources.

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

### Option B: Pilot Run

Quick validation on a subset of repositories:

```bash
python launch_pipeline.py \
  --aidev-dir /path/to/AIDev \
  --data-dir ./data \
  --pilot 10 \
  --workers 4
```

### Option C: Analysis Only

Requires the pre-computed parquet files from figshare (see [`data/README.md`](data/README.md)):

```bash
# First, download the data (see data/README.md for links)
python launch_pipeline.py --analyze-only --data-dir ./data
```

## Expected Outputs

After running the pipeline, outputs will be in `./data/`:

**Parquet files (data stage):**
- `universe.parquet` — Agent PR metadata
- `internal_merges.parquet` — All internal merge commits
- `conflict_chunks.parquet` — Individual conflict chunks
- `classified_chunks.parquet` — Chunks with assigned resolution strategies
- `resolver_labels.parquet` — Resolver attribution (agent vs. human)

**CSV & figures (analysis stage):**
- `results/dataset_overview/` — Scope table, conflict counts
- `results/rq1/` — RQ1 tables: resolver attribution
- `results/rq2/` — RQ2 tables: strategy distributions
- `results_summary.json` — All headline numbers for the paper

## Ethical and Legal Considerations

All data derives from **public** pull requests and repositories in the AIDev catalogue (public GitHub activity). No private or personal data beyond public commit and author metadata is collected or redistributed. The derived datasets are shared for research purposes under CC BY 4.0; each analyzed repository retains its own upstream license, and only the conflict/resolution fragments required to reproduce the study are included.

## Availability & Archival

- **Source code:** archived on Zenodo with a version DOI.
  <!-- TODO: add the Zenodo DOI minted from a tagged GitHub release (see note below). -->
  Development repository: <https://github.com/gems-uff/agentic-conflicts>.
- **Data:** archived on figshare under CC BY 4.0 — DOI [10.6084/m9.figshare.32159415](https://doi.org/10.6084/m9.figshare.32159415).

> **Creating the code DOI (one-time):** enable the GitHub–Zenodo integration for the repository, then publish a GitHub *release*; Zenodo mints a DOI for that specific version. Cite that **version** DOI (not the general GitHub URL) in the paper's *Artifact Availability* section.

## License

- **Code:** [MIT License](LICENSE).
- **Data:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), archived on figshare ([doi.org/10.6084/m9.figshare.32159415](https://doi.org/10.6084/m9.figshare.32159415); see [`data/README.md`](data/README.md)).

## Citation

If you use this replication package or its data, please cite:

```bibtex
@inproceedings{agenticconflicts2026,
  title     = {How AI Coding Agents Resolve Merge Conflicts: An Empirical Study},
  author    = {Campos Junior, Heleno de Souza and Murta, Leonardo Gresta Paulino},
  booktitle = {Proceedings of the 40th Brazilian Symposium on Software Engineering (SBES)},
  year      = {2026},
  note      = {To appear. Update with final page numbers and DOI once available.}
}
```

## Contact & Questions

For questions about this replication package or the study, please open an issue on GitHub.
