# Research Plan: How AI Coding Agents Resolve Merge Conflicts: An Empirical Study

**Plan author:** Heleno
**Date:** 2026-04-20 (revised 2026-04-21: full AIDev via `extract_pr_chronology.py`; revised 2026-05-04: updated to reflect submitted paper)
**Base dataset:** AIDev (Li et al., MSR 2026 Mining Challenge) — `AIDev/`
**Status:** Paper submitted to SBES 2026 (September 8–12, 2026, São Paulo, SP, Brazil)
**Reference papers in the plan folder:**
- `AIDev_paper_1.pdf` — AIDev catalog paper (findings and research directions)
- `AIDev_paper_2.pdf` — MSR'26 Mining Challenge description
- `related_papers/On_the_Nature_of_Merge_Conflicts_A_Study_of_2731_Open_Source_Java_Projects_Hosted_by_GitHub.pdf` — Ghiotto et al., TSE 2020 (methodological anchor for resolution strategy taxonomy)

---

## 1. Motivation

AI coding agents (Claude Code, OpenAI Codex, GitHub Copilot in agentic mode, Cursor, and Devin) now open pull requests at scale across public GitHub. The AIDev catalogue documents over 932,000 AI-authored PRs across 116,000 repositories. When these agents pull upstream changes into their feature branches, textual merge conflicts arise; two questions are entirely unanswered: *who* performs the integration step on these branches, and *how* the agents that do their own merges behave relative to one another and to humans.

Prior work on merge conflict resolution (Ghiotto et al. TSE 2020; Vale et al. TSE 2021) has focused exclusively on human developers. The one dataset that targets agentic PRs — AgenticFlict (Ogenrwot & Businge, 2026) — analyzes simulated merges between PR head and base branches but does not distinguish AI from human actors in the integration step and does not study resolution strategies. No prior study has separated AI-performed from human-performed resolutions on real merge commits, nor characterized the strategies applied by each individual agent.

This study fills that gap by conducting an empirical study on internal merge commits within AI-authored pull requests from the AIDev catalogue, attributing each conflict resolution to either the agent or a human, and characterizing the strategy distribution per agent using the Ghiotto et al. taxonomy.

---

## 2. Main Objective

To empirically characterize (i) who resolves the merge conflicts that arise inside AI-authored pull requests — the agent itself or a human collaborator — and how this varies across the five prominent coding agents, and (ii) what resolution strategies each agent applies to its own conflicting merges, both individually and relative to one another and to a human reference baseline drawn from the same corpus. The results provide the empirical foundation for per-agent tool calibration and pipeline design in the era of autonomous coding agents.

---

## 3. Research Questions

**RQ1 — Who resolves the conflicts?** Who resolves the merge conflicts introduced in AI-authored PRs (agent autonomously, agent-assisted by a human co-committer, or human), and how does the self-resolution rate vary across the five coding agents (Claude Code, OpenAI Codex, GitHub Copilot, Cursor, Devin)?

**RQ2 — How do agents resolve their own merges?** How are resolution strategies distributed across conflicting chunks that agents resolve themselves (V1, V2, CC, CB, NC, NN, Imprecise), how does each agent's distribution compare to the human reference, and how do the agents compare to one another — both in aggregate and conditioned on the file-path category of the conflicting file?

**Cross-cutting concerns:**
- Results for both RQs are stratified by PR-authoring agent.
- RQ2 additionally conditions on six file-path categories (source, lock, config, generated, doc, migration) and surfaces per-agent repository-concentration metrics as an external-validity check.

*Dropped relative to earlier plan:* The structural RQ (distributions of chunk count, v1/v2/resolution LOC) was folded into the dataset overview section rather than treated as a primary research question.

---

## 4. Dataset and Scope

### 4.1 Source

The full AIDev dataset (`all_pull_request.parquet` + `all_repository.parquet`), approximately 932,791 PRs from five agents (Claude Code, Codex, Copilot, Cursor, Devin) across 116,211 repositories and 72,631 developers. PRs in all states (merged, closed, open) are included; restricting to merged PRs would bias toward rebase/squash-merge histories that drop the intermediate merge commits we study.

### 4.2 Actual universe reached

Per-PR commit histories are produced by `extract_pr_chronology.py` (Pass A), which fetches `refs/pull/*/head` for every in-scope repository. SHA lists were not recoverable for 25,571 PRs (12.1%): 20,439 due to disconnected histories and 5,132 due to PR head references no longer reachable. The reachable universe comprises **210,902 in-scope PRs** across **57,582 repositories**.

After merge-level deduplication (keyed on `(repository, merge SHA)` to avoid inflation when the same commit is reachable from multiple PRs due to force-pushes, re-openings, or cross-branch merges), the pipeline identifies **50,700 distinct internal merge commits**. Integration merges (subject matches `^Merge pull request #\d+`, case-insensitive) and octopus merges (≥3 parents) are excluded; 70,042 integration-merge SHAs and 93 octopus-merge SHAs were filtered at this step.

### 4.3 Conflict dataset

| Metric | Count | Share |
|---|---|---|
| Repositories | 4,806 | — |
| PRs with ≥1 internal merge | 12,299 | — |
| PRs with ≥1 conflicting merge | 7,268 | 59.1% of PRs |
| Internal merge commits (dedup.) | 50,700 | — |
| Conflicting merge commits | 14,960 | 29.5% of merges |
| Conflict chunks | 121,599 | — |
| Chunks with localized resolution | 83,043 | 68.3% of chunks |
| — of which: Postponed | 7,291 | 6.0% of chunks |
| Resolved chunks (analysis set) | 75,752 | 62.3% of chunks |

The merge-level conflict incidence (fraction of internal merges that produce at least one textual conflict) ranges from 10.5% (Copilot) to 47.8% (Cursor).

### 4.4 Unit of analysis

- **Internal merge commit**: a commit referenced by the PR with exactly two parents, excluding the GitHub integration merge and octopus merges.
- **Conflicting chunk**: the unit used by Ghiotto et al., bounded by `<<<<<<<`, `=======`, `>>>>>>>` markers. Analysis granularity follows Ghiotto et al. [20].

---

## 5. Methodology

### 5.1 Identifying internal merge commits (Pass A + Pass B Stage 2)

**Pass A — `extract_pr_chronology.py`.** For every repository, fetches `refs/pull/*/head` and enumerates, per PR, commits between the fork-point (`git merge-base`) and the PR tip. Output: `data/pr_chronology/pr_commits.parquet` with schema `(pr_id, repo_full_name, pr_number, sha, author_date, commit_index, commit_count)`.

**Pass B Stage 2 — `extract_aidev_nature.py`.** For each SHA in the chronology, `git cat-file -p <sha>` retrieves the parent list. Two-parent commits are candidate internal merges; GitHub integration merges are filtered by message pattern; octopus (≥3 parent) and unreachable SHAs are recorded in the audit table.

### 5.2 Conflict reconstruction

For each internal merge `M` with parents `P1`, `P2`:
1. Compute `B = git merge-base P1 P2`.
2. For each file modified on both sides, materialize the three blobs and invoke `git merge-file -p --diff3 p1_file base_file p2_file`.
3. Parse output with `MERGE_TREE_CONFLICT_REGEX` / `DIFF3_CHUNK_REGEX` to extract each `(v1, base, v2)` triple.

### 5.3 Resolution localization

The LOCALIZERESREGION algorithm (Dinella et al. [16], implemented in `collect.py :: find_resolution`) anchors on the unique textual context (prefix and suffix) surrounding each conflict region in the three-way diff output to isolate the exact resolution text from the merge commit's tree. Chunks where unique anchors cannot be found are marked *Imprecise* and reported as a separate bucket. Chunks whose committed file still contains conflict markers are marked *Postponed*.

### 5.4 Resolution strategy classification

`identify_resolution(v1, v2, resolution)` from `extract_resolution_strategies.py` returns one of seven labels:
- **V1 / V2**: resolution equals one side verbatim.
- **CC** (Concatenation): resolution equals `v1 + v2` or `v2 + v1` (both orderings collapsed to CC for comparability with Ghiotto et al.).
- **CB** (Combination): resolution interleaves lines from both sides without introducing new lines.
- **NC** (New Code): resolution introduces content absent from both sides.
- **NN** (None): resolution is empty.
- **Imprecise**: localization failed; kept as a separate bucket to avoid silent misclassification.

*Imprecise* and *Postponed* are folded for the statistical analyses but reported separately throughout for transparency.

### 5.5 Resolver attribution

The author of the internal merge commit `M` is matched against known agent-account signatures via a substring/suffix heuristic over the author login:
- A login that ends in `[bot]` or contains any of `copilot`, `devin`, `claude`, `cursor`, `openai` → **Agent**.
- A `Co-authored-by:` trailer in the commit message matching the same rule → **Agent-assisted**.
- Otherwise → **Human**.

The PR-authoring agent is recorded separately from the merge committer, taken from AIDev's bot-account mapping. Per-agent strategy distributions (RQ2) include only chunks where the resolver is classified as *Agent*.

### 5.6 File-path categorization

Each conflicting file is assigned to one of six mutually exclusive categories using a deterministic, priority-ordered heuristic applied to the repository-relative file path (first matching rule wins):

1. **lock** — basename exactly matches one of 16 known dependency lock-file names (e.g., `package-lock.json`, `yarn.lock`, `Cargo.lock`, …).
2. **generated** — path suffix in `{.min.js, .min.css, .bundle.js, .map, .snap}` or path traverses a known build/dependency directory (`dist/`, `build/`, `node_modules/`, `vendor/`, `target/`, `coverage/`, `__pycache__/`, `.next/`, `.nuxt/`, `.cache/`, `.parcel-cache/`).
3. **doc** — extension in `{.md, .rst, .txt, .adoc}`.
4. **config** — extension in `{.json, .yaml, .yml, .toml, .ini, .cfg, .conf, .env}`.
5. **migration** — extension `.sql` or path contains `/migrations/` or `/migration/`.
6. **source** — all remaining files.

### 5.7 Statistical analyses

- Proportions reported with 95% Wilson confidence intervals.
- Association between categorical variables tested with bias-corrected Cramér's *V* as the primary effect-size metric (weak < 0.10, moderate 0.10–0.30, strong ≥ 0.30).
- Per-merge distributions (e.g., chunks per merge) compared with Kruskal–Wallis + post-hoc Dunn tests, Holm-adjusted *p*-values.
- Pairwise agent-vs-agent strategy contrasts: chi-squared tests for the nine pairs where distributional assumptions hold (expected counts ≥ 5), with Holm correction across ten testable pairs.
- Repository-concentration analysis: top-1 and top-3 repo share among self-resolved merges per agent, reported as an external-validity check.

### 5.8 Sensitivity analysis

To bound the effect of differential Imprecise rates between agent-resolved and human-resolved chunks, a sensitivity analysis reassigns all Imprecise chunks to *V1* (upper bound for *V1* dominance) or to *NC* (lower bound), and checks whether per-agent contrasts survive both extremes.

---

## 6. Extraction Pipeline

### 6.1 Stages

**Stage 0 — Universe preparation.** Load `all_pull_request.parquet` + `all_repository.parquet` from `AIDev/` (or `pull_request.parquet` + `repository.parquet` under `--pop-only`), join with `data/pr_chronology/pr_commits.parquet`, attach `agent`, `language`, `state`, `merged_at`. Persist as `data/nature_of_agent_conflicts/universe.parquet`.

**Stage 1 — Bare cloning, per-shard.** `git clone --bare --filter=blob:none` per repository into a temporary scratch directory; removed at end of shard to bound disk usage.

**Stage 2 — Merge-commit enumeration.** `git cat-file -p <sha>` per candidate SHA; classify as `regular` / `internal-merge` / `octopus`; emit `internal_merges.parquet` with schema `(pr_id, merge_sha, parent1_sha, parent2_sha, author, committer, committed_at)`.

**Stage 3 — Conflict replay.** `git merge-base` + `git merge-file --diff3`; parse chunks; emit `conflict_chunks.parquet` with schema `(pr_id, merge_sha, file_path, chunk_index, v1, base, v2, v1_loc, v2_loc, base_loc)`.

**Stage 4 — Resolution localization.** LOCALIZERESREGION against merge tree; emit `resolved_chunks.parquet` adding `(resolution, resolution_loc, localized_ok)`.

**Stage 5 — Strategy classification.** `identify_resolution` per chunk; emit `classified_chunks.parquet` adding `(strategy)`.

**Stage 6 — Resolver attribution.** Agent-account signature heuristic against `internal_merges.author`; emit `resolver_labels.parquet` with `(pr_id, merge_sha, resolver_type)` ∈ `{agent, agent-assisted, human}`.

**Stage 7 — File-path categorization.** Priority-ordered heuristic on `file_path`; emit `file_categories.parquet` with `(merge_sha, file_path, category)`.

**Stage 8 — Analysis.** `analysis.ipynb` joins the parquet outputs and produces per-RQ tables, plots, and statistical tests.

### 6.2 Deduplication

Chunk-level tables are de-duplicated on natural keys `(repo_full_name, merge_sha, file_path, chunk_index)` at JSONL→Parquet aggregation so that PRs sharing a merge commit (force-pushed, re-opened, cherry-picked) do not inflate downstream counts. Merge-level deduplication on `(repo_full_name, merge_sha)` reduces 452,114 raw PR-level merge references to 50,700 distinct merge commits.

### 6.3 Output directory layout

```
data/pr_chronology/                     # Pass A output (extract_pr_chronology.py)
  pr_commits.parquet                    # (pr_id, repo_full_name, pr_number, sha,
                                        #  author_date, commit_index, commit_count)
  errors.parquet                        # fetch / ref-not-found audit
  processed_repos.txt                   # incremental tracker
  logs/

data/nature_of_agent_conflicts/         # Pass B output (extract_aidev_nature.py)
  universe.parquet
  internal_merges.parquet
  conflict_chunks.parquet
  resolved_chunks.parquet
  classified_chunks.parquet
  resolver_labels.parquet
  file_categories.parquet
  extraction_errors.parquet             # clone failures, missing SHAs,
                                        # octopus merges, integration merges, timeouts
  processed_repos.txt                   # incremental tracker
  logs/
```

---

## 7. Computational Budget

The pipeline reached 57,582 repositories out of ~116,000 in AIDev. The gap is primarily due to private/deleted repos or PR head references no longer exposed by GitHub at fetch time. Within the reached repositories, 108 (0.2%) were further lost to cloning failures, and 236,449 merge SHAs across 88,339 PRs were unrecoverable. No evidence of systematic directional bias from these losses.

---

## 8. Work Plan and Status

All four phases are **complete** as of paper submission.

**Phase 1 — Pipeline implementation and pilot.** ✓ `extract_aidev_nature.py` implemented and validated on a pilot shard.

**Phase 2 — Full extraction.** ✓ Pass A and Pass B completed. Final corpus: 50,700 unique internal merge commits, 14,960 conflicting, 75,752 resolved chunks in the analysis set.

**Phase 3 — Analysis and paper writing.** ✓ RQ1 and RQ2 analyses complete; paper drafted and submitted to SBES 2026.

**Phase 4 — Internal review and revision.** ✓ Sensitivity analysis completed; threats to validity discussed; paper submitted anonymously.

---

## 9. Threats to Validity

**Construct validity — Replay vs. historical merge.** The pipeline replays three-way merges using `git merge-file` with the default diff3 algorithm; projects with custom merge drivers may yield different conflict counts (same limitation as Ghiotto et al. [20] and AgenticFlict [29]).

**Construct validity — Localization errors.** LOCALIZERESREGION can fail when context lines around a conflict chunk were edited in the merge commit itself. Every failure is kept as *Imprecise* rather than silently misclassified. The differential *Imprecise* rate between agent-resolved (16.86%) and human-resolved (43.25%) chunks has two components: *localization failure* (16.1% vs. 33.4%) and *Postponed* cases (0.76% vs. 9.85%). The sensitivity analysis (§5.8) bounds the agent-to-human *V1* ratio at 1.24×–2.45×, showing contrasts are not artefacts of differential localization.

**Internal validity — Resolver attribution.** The substring/suffix heuristic over author login introduces false positives (generic automation bots such as `kodiakhq[bot]` and `mergify[bot]`) and false negatives (agent accounts committing under non-matching logins). The dominant findings are unlikely to be reversed under reasonable misclassification levels.

**Internal validity — PR-authoring agent as proxy.** We use the PR-authoring agent (from AIDev) as a proxy for the merge-committing agent; this is exact under the canonical workflow but inexact if a developer invokes a different agent inside an already-open PR.

**External validity — Repository concentration.** Two of the five agents are dominated by a single repository: 90.2% of Codex self-resolved merges come from `BlazingS​u/nocobase` and 83.3% of Claude Code self-resolved merges come from `itdojp/ITDO_ERP2`. Per-agent strategy distributions for these two agents reflect the dominant repository's workflow rather than the agent itself; per-agent conclusions (especially "Claude Code is indistinguishable from humans") should be read with this in mind and are generalized with caution.

**External validity — Agent mix and time scope.** The corpus is heavily skewed toward Codex (~76% of in-scope PRs). The five agents are the most prominent as of mid-2025; baselines should be revisited as the ecosystem matures.

**External validity — Public-GitHub scope.** Results should not be extrapolated to private repositories without care.

**Conclusion validity.** Conflict chunks within the same merge and repository are not fully independent, so *p*-values are anti-conservative (Type I error may be inflated). We therefore emphasize effect sizes (Cramér's *V* and Cliff's δ) and do not treat *p* < 0.05 alone as evidence of practical importance.

---

## 10. Key Findings

**Finding 1.** Conflict resolution in agent-authored PRs is overwhelmingly human-driven: of the 14,960 conflicting internal merges, 96.1% are committed by a human and only 3.1% by the agent itself. The per-agent self-resolution rate spans nearly 60× (Codex 0.5% to Devin 29.4%).

**Finding 2.** The five agents do not share a common resolution approach. Per-agent Cramér's *V* against the human reference ranges from indistinguishable (Claude Code, *V*=0.02) to strongly divergent (Codex, *V*=0.32); pairwise inter-agent contrasts span from indistinguishable (Cursor vs. Devin, *V*=0.06, the only Holm-non-significant pair) to nearly disjoint (Copilot vs. Codex, *V*=0.90).

**Finding 3.** Two of the five agents are concentrated in a single repository: Codex (90.2% from `BlazingS​u/nocobase`) and Claude Code (83.3% from `itdojp/ITDO_ERP2`). For these two agents, the per-agent distribution reflects the dominant repository's workflow rather than the agent itself.

**Finding 4.** Agent resolvers commit *Postponed* chunks (files still containing conflict markers) at 0.76%, versus 9.85% for humans — a 13× asymmetry. Agents typically execute within scripted workflows that programmatically verify absence of markers before committing; a CI-level grep would catch the 6.56% human rate far more cheaply than post-hoc tooling.
