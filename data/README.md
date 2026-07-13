# Data

The intermediate datasets needed to reproduce the analysis are **not stored in
this repository** because of their size (~3.9 GB compressed). They are archived
on figshare under the Creative Commons Attribution 4.0 (CC BY 4.0) license:

- **DOI:** [10.6084/m9.figshare.32159415](https://doi.org/10.6084/m9.figshare.32159415)
- **Title:** *Folder with the dataset used for the paper "How AI Coding Agents Resolve Merge Conflicts: An Empirical Study"*

## Download and extract

```bash
# From the repository root, download the archive into this data/ directory
curl -L -o data/dataset.zip https://ndownloader.figshare.com/files/64217220

# Verify integrity (optional)
#   expected MD5: 9471803480abf70523b7376ac0d1f4d9

# Extract the parquet files directly into data/
unzip data/dataset.zip -d data/
rm data/dataset.zip
```

After extraction, `data/` should contain the parquet tables consumed by the
analysis stage:

| File | Contents |
|---|---|
| `universe.parquet` | `(pr_id, sha)` rows joined with PR and repository metadata |
| `internal_merges.parquet` | One row per two-parent internal merge commit |
| `conflict_chunks.parquet` | One row per conflicting chunk produced by replay |
| `resolved_chunks.parquet` | Conflict chunks plus the localized resolution region |
| `classified_chunks.parquet` | Resolved chunks plus the assigned resolution strategy |
| `resolver_labels.parquet` | Per-merge resolver attribution (`agent` / `agent-assisted` / `human`) |
| `extraction_errors.parquet` | Audit table (clone failures, missing SHAs, octopus/integration merges) |

> If the archive extracts into a nested folder, move the `*.parquet` files so
> they sit directly under `data/`, or point the pipeline at the folder that
> contains them with `--data-dir`.

## Reproduce the analysis

```bash
python launch_pipeline.py --analyze-only --data-dir ./data
```

See the top-level [`README.md`](../README.md) for the full pipeline (mining +
analysis) and other run modes.
