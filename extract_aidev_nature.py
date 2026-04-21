#!/usr/bin/env python3
"""
Pipeline for Phase 1: Nature of Agent Conflicts.

The unit of analysis is the internal merge commit of a PR (PLAN.md §4.3). Per-PR
commit SHAs are read from the chronology produced by
``extract_pr_chronology.py`` (PLAN.md §5.1), which walks ``refs/pull/*/head``
for every repository and enumerates commits from the fork-point with the base
branch to the PR tip. That source works for the **full AIDev** catalog, not
only AIDev-pop -- it removes the restriction that the AIDev-provided
``pr_commits.parquet`` is published only for repositories with more than 100
stars.

Expected run order:

    python extract_pr_chronology.py [--full-aidev]   # Step 1 (one-off per repo)
    python extract_aidev_nature.py                    # Step 2 (this script)

Scope defaults to the full AIDev tier (``all_pull_request.parquet`` x
``all_repository.parquet``). Pass ``--pop-only`` to narrow to AIDev-pop.
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from collect import (
    clone_repo_bare,
    run_git_command,
    run_merge_file,
    parse_diff3_chunks,
    find_resolution,
    MERGE_TREE_CONFLICT_REGEX,
    time_limit,
    FunctionTimeoutError,
)
from extract_resolution_strategies import identify_resolution, remove_empty_lines

# --- Configuration ---
AIDEV_DIR = Path("AIDev")
DATA_DIR = Path("data/nature_of_agent_conflicts")
SCRATCH_DIR = DATA_DIR / "scratch"
LOGS_DIR = DATA_DIR / "logs"

# Default location of the per-PR commit inventory. This file is produced by
# ``extract_pr_chronology.py`` -- see PLAN.md §5.1. Override with
# ``--pr-commits PATH`` on the command line if the chronology output lives
# somewhere else (e.g. a sibling clone or a shared drive).
DEFAULT_PR_COMMITS = Path("data/pr_chronology/pr_commits.parquet")

# Hard cap, in seconds, for LOCALIZERESREGION on a single chunk. The algorithm
# is O(n^2) on the resolved-file size in the worst case; without this guard a
# pathologically large file can stall a worker indefinitely.
LOCALIZE_TIMEOUT_S = 60

# Mensagem do merge de integracao que o GitHub cria quando o botao "Merge
# pull request" e usado. Esses commits tem 2 parents mas NAO sao merges
# internos -- devem ser filtrados (PLAN.md §4.3).
_PR_INTEGRATION_MERGE_RE = re.compile(r"merge pull request\s+#\d+", re.IGNORECASE)


for d in [DATA_DIR, SCRATCH_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configure custom logger to prevent hijack
logger = logging.getLogger("extract_aidev")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"pipeline_{timestamp}.log"
fh = logging.FileHandler(LOGS_DIR / log_filename, mode='w', encoding='utf-8')
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)
logger.propagate = False


def is_bot_signature(text: str) -> bool:
    """Classify if a string matches typical agent/bot signatures."""
    t = text.lower()
    bot_markers = ["[bot]", "-bot", "copilot", "devin", "claude", "cursor", "openai"]
    if t.endswith("bot"):
        return True
    return any(marker in t for marker in bot_markers)


def classify_resolver(author: str, message: str) -> str:
    """Classify resolver into agent, agent-assisted, human based on author and commit message."""
    if is_bot_signature(author):
        return "agent"

    lines = message.splitlines()
    for line in lines:
        if line.lower().startswith("co-authored-by:"):
            if is_bot_signature(line):
                return "agent-assisted"

    return "human"


def _normalize_repo_url(repo_url: str) -> str:
    """Coerce any AIDev-style repo URL to ``https://github.com/<owner>/<repo>.git``."""
    if not isinstance(repo_url, str):
        return repo_url
    url = repo_url.strip()
    if url.startswith("https://api.github.com/repos/"):
        url = url.replace("https://api.github.com/repos/", "https://github.com/")
    if not url.endswith(".git"):
        url += ".git"
    return url


def build_universe(
    is_pilot: bool,
    pilot_count: int,
    use_pop_only: bool,
    pr_commits_path: Path,
) -> pd.DataFrame:
    """Stage 0 -- build the analysis universe (PLAN.md §4.1, §6.1).

    Joins AIDev's PR and repository metadata with the per-PR commit inventory
    produced by ``extract_pr_chronology.py``. Every row carries exactly one
    candidate SHA belonging to a PR (one row per ``(pr_id, sha)`` pair).

    Scope is controlled by ``use_pop_only``:

    * ``False`` (default)  -> full AIDev tier, using ``all_pull_request.parquet``
      and ``all_repository.parquet``.
    * ``True``             -> AIDev-pop, using ``pull_request.parquet`` and
      ``repository.parquet`` (stars > 100).

    PRs with no rows in the chronology output are dropped -- either the
    chronology has not yet processed that repository, or the PR has zero
    own commits past the fork-point. In the first case, re-running
    ``extract_pr_chronology.py`` extends the universe on the next run.
    All PR states are kept (PLAN.md §4.2).
    """
    if use_pop_only:
        logger.info("Stage 0: Building AIDev-pop universe (stars > 100)...")
        pr_df = pd.read_parquet(AIDEV_DIR / "pull_request.parquet")
        repo_df = pd.read_parquet(AIDEV_DIR / "repository.parquet")
    else:
        logger.info("Stage 0: Building full AIDev universe...")
        pr_df = pd.read_parquet(AIDEV_DIR / "all_pull_request.parquet")
        repo_df = pd.read_parquet(AIDEV_DIR / "all_repository.parquet")

    try:
        task_df = pd.read_parquet(AIDEV_DIR / "pr_task_type.parquet")
    except Exception:
        task_df = pd.DataFrame(columns=['id', 'type'])

    pr_repo_df = pr_df.merge(
        repo_df[['id', 'full_name', 'language']],
        left_on='repo_id', right_on='id', suffixes=('', '_repo'),
    )

    if not task_df.empty:
        pr_repo_task_df = pr_repo_df.merge(task_df[['id', 'type']], on='id', how='left')
        pr_repo_task_df.rename(columns={'type': 'pr_task_type'}, inplace=True)
    else:
        pr_repo_task_df = pr_repo_df.copy()
        pr_repo_task_df['pr_task_type'] = None

    cols_to_keep = ['id', 'number', 'repo_url', 'full_name', 'language',
                    'agent', 'pr_task_type', 'state']
    if 'merged_at' in pr_repo_task_df.columns:
        cols_to_keep.append('merged_at')
    cols_to_keep = [c for c in cols_to_keep if c in pr_repo_task_df.columns]

    if not pr_commits_path.exists():
        raise FileNotFoundError(
            f"PR commit inventory not found at {pr_commits_path}. "
            f"Run extract_pr_chronology.py first (see PLAN.md §5.1 / README)."
        )
    commits_df = pd.read_parquet(pr_commits_path)
    # The chronology adds context columns (commit_index, author_date, pr_number,
    # commit_count, repo_full_name). Keep what's available; fall back cleanly if
    # a user points this at AIDev's official pr_commits.parquet, whose schema is
    # ``(pr_id, sha, author, committer, message)``.
    extras = [c for c in ('commit_index', 'author_date', 'commit_count')
              if c in commits_df.columns]
    commits_df = commits_df[['pr_id', 'sha'] + extras]

    universe_df = commits_df.merge(
        pr_repo_task_df[cols_to_keep],
        left_on='pr_id', right_on='id',
    )

    if is_pilot:
        unique_repos = universe_df['full_name'].dropna().unique()[:pilot_count]
        universe_df = universe_df[universe_df['full_name'].isin(unique_repos)]
        logger.info(f"Pilot mode: restricted to {len(unique_repos)} repos.")

    out_path = DATA_DIR / "universe.parquet"
    universe_df.to_parquet(out_path)
    logger.info(
        f"Stage 0 complete. Universe has {len(universe_df)} (pr_id, sha) rows "
        f"across {universe_df['full_name'].nunique()} repositories and "
        f"{universe_df['pr_id'].nunique()} PRs."
    )
    return universe_df


def _parse_commit_object(blob: bytes):
    """Parse the output of ``git cat-file -p <sha>`` for a commit object."""
    text = blob.decode("utf-8", "ignore")
    parts = text.split("\n\n", 1)
    metadata = parts[0]
    message = parts[1] if len(parts) > 1 else ""

    parents = []
    author = "Unknown"
    committer = "Unknown"
    for line in metadata.split("\n"):
        if line.startswith("parent "):
            parents.append(line.split(" ", 1)[1].strip())
        elif line.startswith("author "):
            author = line.split("author ", 1)[1].split("<")[0].strip()
        elif line.startswith("committer "):
            committer = line.split("committer ", 1)[1].split("<")[0].strip()

    return {
        "parents": parents,
        "author": author,
        "committer": committer,
        "message": message,
    }


def _process_one_merge(repo_path, repo_full_name, pr_id, sha, parsed):
    """Replay the three-way merge for a single internal merge commit."""
    p1, p2 = parsed["parents"]
    resolver_type = classify_resolver(parsed["author"], parsed["message"])

    internal_merge = {
        "pr_id": pr_id,
        "merge_sha": sha,
        "parent1_sha": p1,
        "parent2_sha": p2,
        "author": parsed["author"],
        "committer": parsed["committer"],
        "repo_full_name": repo_full_name,
        "resolver_type": resolver_type,
    }

    conflict_chunks = []
    resolved_chunks = []
    classified_chunks = []
    errors = []

    base_bytes = run_git_command(repo_path, "merge-base", p1, p2, check=False)
    if not base_bytes:
        return internal_merge, [], [], [], []
    base = base_bytes.decode("utf-8", "ignore").strip()
    if not base:
        return internal_merge, [], [], [], []

    tree_bytes = run_git_command(repo_path, "merge-tree", base, p1, p2, check=False)
    if not tree_bytes:
        return internal_merge, [], [], [], []
    tree_output = tree_bytes.decode("utf-8", "ignore")

    for match in MERGE_TREE_CONFLICT_REGEX.finditer(tree_output):
        base_blob, path, p1_blob, p2_blob = match.groups()

        try:
            base_content = run_git_command(repo_path, "show", base_blob, check=False)
            p1_content = run_git_command(repo_path, "show", p1_blob, check=False)
            p2_content = run_git_command(repo_path, "show", p2_blob, check=False)
            resolved_bytes = run_git_command(
                repo_path, "show", f"{sha}:{path}", check=False
            )
            resolved_content = resolved_bytes.decode("utf-8", "ignore") if resolved_bytes else ""

            conflict_content, has_conflict = run_merge_file(p1_content, base_content, p2_content)
            if not has_conflict:
                continue

            try:
                with time_limit(LOCALIZE_TIMEOUT_S):
                    chunks = parse_diff3_chunks(conflict_content)
                    if not chunks:
                        continue

                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_data = {
                            "repo_full_name": repo_full_name,
                            "pr_id": pr_id,
                            "merge_sha": sha,
                            "file_path": path,
                            "chunk_index": chunk_idx,
                            "v1": chunk['v1'],
                            "base": chunk['base'],
                            "v2": chunk['v2'],
                            "v1_loc": len(remove_empty_lines(chunk['v1'].splitlines())),
                            "v2_loc": len(remove_empty_lines(chunk['v2'].splitlines())),
                            "base_loc": len(remove_empty_lines(chunk['base'].splitlines())),
                        }
                        conflict_chunks.append(chunk_data.copy())

                        resolution, status = find_resolution(
                            chunk["pre_context"], chunk["post_context"], resolved_content
                        )
                        chunk_data["resolution"] = resolution
                        chunk_data["localized_ok"] = (status == "found")
                        chunk_data["resolution_loc"] = (
                            len(remove_empty_lines(resolution.splitlines())) if resolution else 0
                        )
                        resolved_chunks.append(chunk_data.copy())

                        strategy = (
                            identify_resolution(chunk['v1'], chunk['v2'], resolution)
                            if resolution is not None else "Imprecise"
                        )
                        chunk_data["strategy"] = strategy
                        classified_chunks.append(chunk_data.copy())
            except FunctionTimeoutError:
                errors.append({
                    "repo_full_name": repo_full_name,
                    "pr_id": pr_id,
                    "merge_sha": sha,
                    "file_path": path,
                    "error_type": "localize_timeout",
                    "error_message": f"LOCALIZERESREGION exceeded {LOCALIZE_TIMEOUT_S}s",
                })
                continue
        except Exception as e:
            errors.append({
                "repo_full_name": repo_full_name,
                "pr_id": pr_id,
                "merge_sha": sha,
                "file_path": path,
                "error_type": "file_processing_exception",
                "error_message": str(e),
            })
            continue

    return internal_merge, conflict_chunks, resolved_chunks, classified_chunks, errors


def process_repository(repo_info: tuple):
    repo_full_name, repo_df = repo_info

    repo_url = _normalize_repo_url(repo_df.iloc[0]['repo_url'])

    repo_path = clone_repo_bare(repo_url, SCRATCH_DIR)
    if not repo_path:
        logger.error(f"Nao foi possivel clonar/atualizar o repositorio {repo_full_name}.")
        return repo_full_name, [], [], [], [], [{
            "repo_full_name": repo_full_name,
            "pr_id": None,
            "merge_sha": None,
            "error_type": "clone_failed",
            "error_message": "Nao foi possivel clonar ou atualizar o repositorio",
        }]

    internal_merges = []
    conflict_chunks = []
    resolved_chunks = []
    classified_chunks = []
    extraction_errors = []

    try:
        prs = repo_df.groupby('pr_id')
        for pr_id, pr_data in prs:
            try:
                shas = (
                    pr_data['sha'].dropna().drop_duplicates().tolist()
                    if 'sha' in pr_data.columns else []
                )
                if not shas:
                    extraction_errors.append({
                        "repo_full_name": repo_full_name,
                        "pr_id": pr_id,
                        "merge_sha": None,
                        "error_type": "no_sha_for_pr",
                        "error_message": "PR has no entries in the chronology output (repo not yet processed by extract_pr_chronology.py, or PR has zero own commits past the fork-point).",
                    })
                    continue

                for sha in shas:
                    try:
                        blob = run_git_command(repo_path, "cat-file", "-p", sha, check=False)
                        if not blob:
                            extraction_errors.append({
                                "repo_full_name": repo_full_name,
                                "pr_id": pr_id,
                                "merge_sha": sha,
                                "error_type": "sha_not_in_repo",
                                "error_message": "git cat-file returned empty (force-push or rebased away?).",
                            })
                            continue

                        parsed = _parse_commit_object(blob)
                        n_parents = len(parsed["parents"])

                        if n_parents < 2:
                            continue

                        if n_parents >= 3:
                            extraction_errors.append({
                                "repo_full_name": repo_full_name,
                                "pr_id": pr_id,
                                "merge_sha": sha,
                                "error_type": "octopus_merge",
                                "error_message": f"Commit has {n_parents} parents; excluded from the analysis.",
                            })
                            continue

                        if _PR_INTEGRATION_MERGE_RE.search(parsed["message"]):
                            extraction_errors.append({
                                "repo_full_name": repo_full_name,
                                "pr_id": pr_id,
                                "merge_sha": sha,
                                "error_type": "pr_integration_merge",
                                "error_message": "Excluded GitHub 'Merge pull request #N' integration merge.",
                            })
                            continue

                        im, cc, rc, ca, errs = _process_one_merge(
                            repo_path, repo_full_name, pr_id, sha, parsed
                        )
                        internal_merges.append(im)
                        conflict_chunks.extend(cc)
                        resolved_chunks.extend(rc)
                        classified_chunks.extend(ca)
                        extraction_errors.extend(errs)

                    except Exception as e:
                        extraction_errors.append({
                            "repo_full_name": repo_full_name,
                            "pr_id": pr_id,
                            "merge_sha": sha,
                            "error_type": "sha_processing_exception",
                            "error_message": str(e),
                        })
            except Exception as e:
                extraction_errors.append({
                    "repo_full_name": repo_full_name,
                    "pr_id": pr_id,
                    "merge_sha": None,
                    "error_type": "pr_processing_exception",
                    "error_message": str(e),
                })
    finally:
        def _remove_readonly(func, path, excinfo):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)

        try:
            shutil.rmtree(repo_path, onerror=_remove_readonly)
        except Exception as e:
            logger.warning(f"Falha ao remover {repo_path}: {e}")

    return (
        repo_full_name,
        internal_merges,
        conflict_chunks,
        resolved_chunks,
        classified_chunks,
        extraction_errors,
    )


def append_jsonl(filename: str, records: list):
    if not records:
        return
    with open(DATA_DIR / filename, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def check_processed() -> set:
    tracker = DATA_DIR / "processed_repos.txt"
    if not tracker.exists():
        return set()
    with open(tracker, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def mark_processed(repo_name: str):
    with open(DATA_DIR / "processed_repos.txt", "a", encoding="utf-8") as f:
        f.write(f"{repo_name}\n")


_DEDUP_KEYS = {
    "internal_merges.jsonl": ["repo_full_name", "pr_id", "merge_sha"],
    "conflict_chunks.jsonl": ["repo_full_name", "merge_sha", "file_path", "chunk_index"],
    "resolved_chunks.jsonl": ["repo_full_name", "merge_sha", "file_path", "chunk_index"],
    "classified_chunks.jsonl": ["repo_full_name", "merge_sha", "file_path", "chunk_index"],
    "extraction_errors.jsonl": None,
}


def aggregate_jsonl_to_parquet():
    """Convert the incremental jsonl files into the final parquet format.

    De-duplicates each table on its natural key before persisting (PLAN.md §6.3).
    Errors are kept verbatim.
    """
    logger.info("Aggregating incremental JSONL files to Parquet...")
    files_map = {
        "internal_merges.jsonl": "internal_merges.parquet",
        "conflict_chunks.jsonl": "conflict_chunks.parquet",
        "resolved_chunks.jsonl": "resolved_chunks.parquet",
        "classified_chunks.jsonl": "classified_chunks.parquet",
        "extraction_errors.jsonl": "extraction_errors.parquet",
    }

    for jsonl_name, parquet_name in files_map.items():
        jsonl_path = DATA_DIR / jsonl_name
        if not jsonl_path.exists():
            continue
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Pulando linha malformada em {jsonl_name}")
        df = pd.DataFrame(records)
        keys = _DEDUP_KEYS.get(jsonl_name)
        if keys and not df.empty and all(k in df.columns for k in keys):
            before = len(df)
            df = df.drop_duplicates(subset=keys, keep="first")
            after = len(df)
            if before != after:
                logger.info(f"  Dedup {jsonl_name}: {before:,} -> {after:,} rows")
        df.to_parquet(DATA_DIR / parquet_name)
        logger.info(f"Converted {jsonl_name} -> {parquet_name} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 extraction over AIDev. The per-PR commit inventory is "
            "produced by extract_pr_chronology.py (see PLAN.md §5.1). Default "
            "scope is the full AIDev tier; pass --pop-only to restrict to "
            "AIDev-pop (stars > 100)."
        )
    )
    parser.add_argument("--pilot", type=int, default=0,
                        help="Pilot mode: process only the first N repositories.")
    parser.add_argument("--pop-only", action="store_true",
                        help="Restrict the universe to AIDev-pop "
                             "(pull_request.parquet + repository.parquet) "
                             "instead of the full AIDev tier.")
    parser.add_argument("--pr-commits", type=str, default=str(DEFAULT_PR_COMMITS),
                        help=f"Path to the chronology parquet produced by "
                             f"extract_pr_chronology.py "
                             f"(default: {DEFAULT_PR_COMMITS}).")
    args = parser.parse_args()

    is_pilot = args.pilot > 0
    pr_commits_path = Path(args.pr_commits)
    scope_label = "AIDev-pop" if args.pop_only else "full AIDev"
    logger.info(f"Scope: {scope_label}; chronology source: {pr_commits_path}")
    universe_df = build_universe(
        is_pilot, args.pilot, args.pop_only, pr_commits_path,
    )

    processed_repos = check_processed()
    logger.info(f"Found {len(processed_repos)} already processed repositories.")

    repo_groups = []
    for name, group in universe_df.groupby('full_name'):
        if name not in processed_repos:
            repo_groups.append((name, group))

    pool_size = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Processing {len(repo_groups)} repositories with {pool_size} workers...")

    total_repo = len(repo_groups)
    if pool_size == 1:
        for i, rg in enumerate(repo_groups):
            name, im, cc, rc, ca, errs = process_repository(rg)
            logger.info(f"[Progresso: {i+1}/{total_repo} ({(i+1)/total_repo:.1%})] Processado {name}")
            append_jsonl("internal_merges.jsonl", im)
            append_jsonl("conflict_chunks.jsonl", cc)
            append_jsonl("resolved_chunks.jsonl", rc)
            append_jsonl("classified_chunks.jsonl", ca)
            append_jsonl("extraction_errors.jsonl", errs)
            mark_processed(name)
    else:
        with multiprocessing.Pool(pool_size) as pool:
            for i, (name, im, cc, rc, ca, errs) in enumerate(
                pool.imap_unordered(process_repository, repo_groups)
            ):
                logger.info(f"[Progresso: {i+1}/{total_repo} ({(i+1)/total_repo:.1%})] Processado {name}")
                append_jsonl("internal_merges.jsonl", im)
                append_jsonl("conflict_chunks.jsonl", cc)
                append_jsonl("resolved_chunks.jsonl", rc)
                append_jsonl("classified_chunks.jsonl", ca)
                append_jsonl("extraction_errors.jsonl", errs)
                mark_processed(name)

    aggregate_jsonl_to_parquet()

    im_path = DATA_DIR / "internal_merges.parquet"
    if im_path.exists():
        df_im = pd.read_parquet(im_path)
        if not df_im.empty and "resolver_type" in df_im.columns:
            df_im[['pr_id', 'merge_sha', 'parent1_sha', 'parent2_sha',
                   'author', 'committer', 'repo_full_name', 'resolver_type']
                  ].to_parquet(DATA_DIR / "resolver_labels.parquet")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
