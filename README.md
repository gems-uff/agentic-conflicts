# How AI Coding Agents Resolve Merge Conflicts: An Empirical Study

An empirical study of who resolves merge conflicts inside AI-authored pull requests and how each agent's resolution strategies compare to one another and to humans. The study covers five coding agents — Anthropic's **Claude Code**, OpenAI's **Codex**, GitHub's **Copilot** (agentic mode), **Cursor**, and Cognition's **Devin** — drawn from the AIDev catalogue.

**Submitted to:** SBES 2026 (September 8–12, 2026, São Paulo, SP, Brazil).

**Replication package:** see [`replication-package/`](replication-package/README.md).

## Study Overview

Starting from ~932,791 AI-authored PRs in AIDev, the pipeline:

1. Enumerates per-PR commit histories via `refs/pull/*/head` (reaching 57,582 repos / 210,902 in-scope PRs).
2. Identifies internal merge commits (two parents, excluding GitHub's integration merges) and deduplicates on `(repository, merge SHA)`, yielding 50,700 unique merge commits.
3. Replays each merge with `git merge-file --diff3`, localizes the resolution using LOCALIZERESREGION, and classifies each chunk as V1 / V2 / CC / CB / NC / NN / Imprecise.
4. Attributes each resolution to the AI agent, an agent-assisted human, or a human, using a login substring/suffix heuristic.
5. Conditions strategy distributions on six file-path categories (source, lock, config, generated, doc, migration) and surfaces per-agent repository-concentration metrics.

**Corpus after deduplication:** 14,960 conflicting merges · 121,599 conflict chunks · 75,752 resolved chunks in the analysis set.

## Research Questions

**RQ1 — Who resolves the conflicts?** Who commits the resolution on conflicting internal merges in AI-authored PRs, and how does the per-agent self-resolution rate vary?

**RQ2 — How do agents resolve their own merges?** What resolution strategies do agents apply autonomously, how do per-agent distributions compare to the human reference, and how do agents compare to one another?

## About the Pipeline

The pipeline runs in two passes, implemented across several scripts.

### Pass A — `extract_pr_chronology.py` (one-off, per repository)

Fetches `refs/pull/*/head` from each remote and, for every PR, walks the commits between the fork-point (`git merge-base refs/pull/<N>/head HEAD`) and the PR tip in chronological order. Writes `data/pr_chronology/pr_commits.parquet` with schema `(pr_id, repo_full_name, pr_number, sha, author_date, commit_index, commit_count)`. This supersedes AIDev's `pr_commits.parquet`, which is only published at the AIDev-pop tier and would otherwise bound the study to ~2,807 repositories.

### Pass B — `extract_aidev_nature.py` (analysis, resumable)

Consumes the Pass A chronology and drives six sequential stages:

- **Stage 0 (Universe):** Joins `all_pull_request.parquet` × `all_repository.parquet` with the chronology; every row is one `(pr_id, sha)` pair enriched with `agent`, `language`, `state`, and `merged_at`. Persisted as `data/nature_of_agent_conflicts/universe.parquet`.
- **Stage 1 (Bare clone):** `git clone --bare --filter=blob:none` per repository into a temporary scratch directory; removed at the end of the repository's processing to bound disk usage.
- **Stage 2 (Internal-merge enumeration):** `git cat-file -p <sha>` per candidate SHA to retrieve the parent list. Two-parent commits are kept; GitHub's `"Merge pull request #N"` integration merges are filtered by message pattern; octopus merges (≥3 parents) and unreachable SHAs are recorded in the audit table.
- **Stage 3 & 4 (Conflict replay & resolution):** `git merge-file --diff3` replays each internal merge and extracts conflicting chunks; LOCALIZERESREGION (`collect.py :: find_resolution`) isolates the corresponding resolution region from the merge tree.
- **Stage 5 (Strategy classification):** Each chunk is labeled V1 / V2 / CC / CB / NC / NN / Imprecise via `identify_resolution` (`extract_resolution_strategies.py`).
- **Stage 6 (Resolver attribution):** Author login substring/suffix heuristic matches against known agent-account patterns (`[bot]`, `copilot`, `devin`, `claude`, `cursor`, `openai`) to classify each merge as `agent` / `agent-assisted` / `human`.

### Additional analysis scripts

- **`extract_agent_file_categories.py`** — Applies the six-category file-path heuristic (source, lock, config, generated, doc, migration) to `classified_chunks.parquet` and emits `file_categories.parquet`.
- **`generate_file_category_table.py`** — Produces Table 6 (file-category distribution of self-resolved chunks per agent) from the classified and categorized data.
- **`extract_file_category_data.py`** — Prepares the per-category strategy distributions used in the RQ2 conditioning analysis.
- **`run_file_category_analysis.sh`** — Convenience wrapper that runs the three file-category scripts in order.
- **`extract_candidates_similarities.py`** — Computes pairwise cosine similarities between candidate resolution strategies; used in the sensitivity analysis.
- **`recover_pr_ref_shas.py`** — Attempts to recover merge SHAs for PRs whose `refs/pull/*/head` was no longer reachable at Pass A time, using alternative ref patterns.
- **`retry_clone_failures.py`** — Serial retry pass for repositories whose clone/fetch failed transiently during the parallel main passes (see below).
- **`sbcr_ga_evaluate.py`** — Evaluates SBCR (search-based conflict resolution) and genetic-algorithm baselines against the corpus; used for the automated-resolution discussion in Section 5.

## Getting Started

To reproduce the data on your own:

1. Clone this repository with the embedded `AIDev` dataset submodule:
    ```bash
    git clone --recursive https://github.com/gems-uff/agentic-conflicts.git
    cd agentic-conflicts
    ```
    *(If you already cloned without `--recursive`, run `git submodule update --init` to fetch the dataset.)*

    > **⚠️ Important:** The `AIDev` submodule uses **Git LFS** (Large File Storage) for its parquet files. Make sure `git-lfs` is installed on your system. If parquet files fail to load, navigate to `AIDev/` and run:
    > ```bash
    > git lfs install
    > git lfs pull
    > ```

2. Enable the virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate    # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Run Pass A (chronology) once per scope:
    ```bash
    # Full AIDev catalog (~116k repos, network-bound — takes days)
    python -m src.extract_pr_chronology --full-aidev

    # Or restricted to AIDev-pop (~2,807 repos) for faster iteration
    python -m src.extract_pr_chronology
    ```

4. Run Pass B (merge extraction and classification):
    ```bash
    # Smoke test on the first 5 repositories
    python -m src.extract_aidev_nature --pilot 5

    # Full run over whatever Pass A covered
    python -m src.extract_aidev_nature

    # Restrict to AIDev-pop even if Pass A covered full AIDev
    python -m src.extract_aidev_nature --pop-only

    # Point at a chronology output elsewhere
    python -m src.extract_aidev_nature --pr-commits /path/to/pr_commits.parquet
    ```

5. Run file-category analysis:
    ```bash
    bash run_file_category_analysis.sh
    ```

6. Open `analysis.ipynb` to explore results and reproduce tables and figures.

### Scope and incremental runs

By default Pass B targets the **full AIDev** tier. Pass `--pop-only` to narrow to AIDev-pop (stars > 100), which is useful for quicker iteration and for joining with enriched AIDev-pop tables (`pr_task_type`, `pr_timeline`, `pr_comments`, etc.). The coverage of Pass B is bounded by whatever Pass A has processed: PRs absent from `pr_commits.parquet` are recorded in `extraction_errors.parquet` as `no_sha_for_pr` and skipped.

Both passes are **incremental**: each successfully processed repository is appended to `processed_repos.txt`. Re-running skips repositories already listed there; partial runs can be resumed or shared across machines. To re-run from scratch, delete `processed_repos.txt` and the `*.jsonl` files in the respective output directory.

### Retrying clone/fetch failures — `retry_clone_failures.py`

Both passes clone in parallel, so transient GitHub errors (DNS blips, 5xx responses, rate-limit spikes, connection resets) surface as `clone_failed`/`fetch_failed` rows in the audit, and the repository is marked as processed and skipped on re-runs. This helper performs a **serial** second pass to distinguish transient from permanent failures.

```bash
# Retry both passes (3 attempts, 30s between attempts)
python -m src.retry_clone_failures --pass both

# Only chronology, with more patience
python -m src.retry_clone_failures --pass chronology --max-retries 5 --retry-delay 120

# AIDev-pop scope
python -m src.retry_clone_failures --pop-only
```

Pass B's retry depends on Pass A: if a repository failed during chronology, its PRs are absent from `pr_commits.parquet` and Pass B has nothing to process until the chronology is rescued first. `--pass both` respects that ordering.

## Outputs

Pass A artifacts land in `data/pr_chronology/`:

| File | Contents |
|---|---|
| `pr_commits.parquet` | Full PR chronology: `(pr_id, repo_full_name, pr_number, sha, author_date, commit_index, commit_count)` |
| `errors.parquet` | Per-repo audit: fetch failures, PR refs no longer reachable |
| `processed_repos.txt` | Incremental tracker |

Pass B artifacts land in `data/nature_of_agent_conflicts/`:

| File | Contents |
|---|---|
| `universe.parquet` | `(pr_id, sha)` rows joined with PR and repo metadata — Stage 0 frame |
| `internal_merges.parquet` | One row per two-parent merge commit found inside a PR |
| `conflict_chunks.parquet` | One row per conflicting chunk produced by replay |
| `resolved_chunks.parquet` | Same, plus LOCALIZERESREGION output (`resolution`, `localized_ok`) |
| `classified_chunks.parquet` | Same, plus resolution `strategy` label |
| `resolver_labels.parquet` | Per-merge `resolver_type` (`agent` / `agent-assisted` / `human`) |
| `file_categories.parquet` | Per-chunk file-path category (source / lock / config / generated / doc / migration) |
| `extraction_errors.parquet` | Audit table: clone failures, missing SHAs, octopus merges, filtered integration merges, timeouts |

Chunk-level tables are de-duplicated on `(repo_full_name, merge_sha, file_path, chunk_index)` at aggregation time. Use `analysis.ipynb` to join these outputs and reproduce the paper's results.

## Key Results

- **96.1%** of conflicting internal merges in AI-authored PRs are resolved by a human; only **3.1%** by the agent itself.
- The per-agent self-resolution rate spans **~60×**: Codex 0.5% → Devin 29.4%.
- The five agents **do not share a common resolution strategy**: pairwise Cramér's *V* ranges from near-indistinguishable (Cursor vs. Devin, *V*=0.06) to nearly disjoint (Copilot vs. Codex, *V*=0.90).
- **Two agents are single-repository dominated**: 90.2% of Codex's and 83.3% of Claude Code's self-resolved merges originate from one repository each; per-agent profiles for these two reflect the dominant repository's workflow, not general agent behaviour.
- Agents commit **Postponed** chunks (unresolved markers left in the file) at **0.76%** vs. **9.85%** for humans — a 13× asymmetry consistent with scripted pre-commit verification.
