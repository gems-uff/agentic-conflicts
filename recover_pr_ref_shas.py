#!/usr/bin/env python3
"""
recover_pr_ref_shas.py — Complementary Pass B for sha_not_in_repo SHAs.

During the main extraction (extract_aidev_nature.py / Pass B), bare-cloned
repositories only receive ``refs/heads/*``.  Any SHA that lives exclusively on
a PR ref (``refs/pull/<N>/head``) and was never merged into a permanent branch
is therefore unreachable by ``git cat-file``, which produces a
``sha_not_in_repo`` error record.

This script performs the following steps:

  1. Reads all ``sha_not_in_repo`` records from extraction_errors.parquet.
  2. Joins with the chronology (pr_commits.parquet) to recover the PR number
     for each (repo, SHA) pair — needed to fetch the right
     ``refs/pull/<N>/head``.
  3. Excludes (repo, SHA) pairs that are already present in
     internal_merges.parquet (recovered by a prior run of this script).
  4. For each affected repository:
       a. Clones / updates the bare repo (same logic as Pass B).
       b. Fetches only the PR refs required: one ``git fetch`` call per
          repository covering all affected PR numbers in that repo.
       c. Re-runs the full Pass-B processing logic on the recovered SHAs,
          reusing the same SHA cache strategy to avoid redundant work when
          the same SHA appears under multiple pr_ids.
  5. Appends new results to the existing JSONL sinks:
       internal_merges.jsonl, conflict_chunks.jsonl, resolved_chunks.jsonl,
       classified_chunks.jsonl, extraction_errors.jsonl.
  6. Calls the standard ``aggregate_jsonl_to_parquet()`` so all five Parquet
     tables are rebuilt from scratch (existing dedup keys guarantee no
     double-counting).
  7. Rebuilds extraction_errors.parquet a second time, removing stale
     ``sha_not_in_repo`` rows for SHAs that were successfully recovered, and
     replacing them with whatever new outcome (success or new error type) was
     produced in step 4.
  8. Rebuilds resolver_labels.parquet from the updated internal_merges.

Safety guarantees
-----------------
* Existing Parquet data is *never overwritten* mid-run; only JSONL appends are
  used during processing, and Parquets are rebuilt once at the very end.
* A per-run tracker (``recovered_pr_refs_repos.txt``) prevents re-fetching and
  re-processing the same repository if the script is interrupted and restarted.
* SHAs already present in internal_merges.parquet (from a previous recovery
  run) are skipped before any network call is made.
* The script never modifies the original ``processed_repos.txt`` tracker used
  by extract_aidev_nature.py, so the two scripts remain independent.

Usage
-----
    python recover_pr_ref_shas.py [--pilot N] [--pr-commits PATH]

    --pilot N         Process only the first N repositories (useful for testing).
    --pr-commits PATH Path to pr_commits.parquet produced by
                      extract_pr_chronology.py (default:
                      data/pr_chronology/pr_commits.parquet).
"""

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Reuse all processing logic from the main extraction script.  Importing
# extract_aidev_nature triggers its top-level directory creation and logger
# setup, both of which are harmless.
# ---------------------------------------------------------------------------
from collect import clone_repo_bare, run_git_command
from extract_aidev_nature import (
    DATA_DIR,
    SCRATCH_DIR,
    LOGS_DIR,
    LOCALIZE_TIMEOUT_S,
    _normalize_repo_url,
    _parse_commit_object,
    _PR_INTEGRATION_MERGE_RE,
    _process_one_merge,
    append_jsonl,
    aggregate_jsonl_to_parquet,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_PR_COMMITS = Path("data/pr_chronology/pr_commits.parquet")
AIDEV_DIR = Path("AIDev")

# Tracker: repos whose PR refs have already been fetched and processed by this
# script.  Independent of the processed_repos.txt used by Pass B.
RECOVERY_TRACKER = DATA_DIR / "recovered_pr_refs_repos.txt"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("recover_pr_refs")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = LOGS_DIR / f"recovery_{_timestamp}.log"
_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
_fh = logging.FileHandler(_log_path, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)

logger.addHandler(_ch)
logger.addHandler(_fh)
logger.propagate = False


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

def _check_recovered() -> set:
    if not RECOVERY_TRACKER.exists():
        return set()
    with open(RECOVERY_TRACKER, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _mark_recovered(repo_name: str) -> None:
    with open(RECOVERY_TRACKER, "a", encoding="utf-8") as f:
        f.write(f"{repo_name}\n")


# ---------------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------------

def load_recovery_targets(pr_commits_path: Path) -> pd.DataFrame:
    """Return a DataFrame of (repo_full_name, merge_sha, pr_id, pr_number, repo_url)
    for every sha_not_in_repo record that has not yet been recovered.

    Steps
    -----
    1. Load sha_not_in_repo errors from extraction_errors.parquet.
    2. Join with pr_commits.parquet on (repo_full_name, pr_id, sha) to get the
       PR number — needed to construct the fetch refspec.
    3. Add repo_url from universe.parquet (or directly from AIDev parquets as
       fallback).
    4. Exclude (repo, SHA) pairs already present in internal_merges.parquet.
    """

    # -- 1. sha_not_in_repo errors ------------------------------------------
    err_path = DATA_DIR / "extraction_errors.parquet"
    if not err_path.exists():
        logger.info("extraction_errors.parquet not found — nothing to recover.")
        return pd.DataFrame()

    errors_df = pd.read_parquet(err_path)
    sha_errors = errors_df[errors_df["error_type"] == "sha_not_in_repo"].copy()
    if sha_errors.empty:
        logger.info("No sha_not_in_repo errors found — nothing to recover.")
        return pd.DataFrame()

    logger.info(
        f"Found {len(sha_errors):,} sha_not_in_repo records across "
        f"{sha_errors['repo_full_name'].nunique():,} repositories."
    )

    # -- 2. Join with pr_commits to get pr_number ---------------------------
    if not pr_commits_path.exists():
        raise FileNotFoundError(
            f"PR commit inventory not found at {pr_commits_path}. "
            "Run extract_pr_chronology.py first."
        )

    pr_commits = pd.read_parquet(pr_commits_path)

    if "pr_number" not in pr_commits.columns:
        logger.error(
            "pr_commits.parquet has no 'pr_number' column.  "
            "Re-run extract_pr_chronology.py to produce an up-to-date inventory."
        )
        return pd.DataFrame()

    # Keep only columns relevant for the join
    sha_map = (
        pr_commits[["pr_id", "repo_full_name", "sha", "pr_number"]]
        .drop_duplicates()
        .rename(columns={"sha": "merge_sha"})
    )

    joined = sha_errors.merge(
        sha_map,
        on=["repo_full_name", "merge_sha", "pr_id"],
        how="left",
    )

    no_number = joined["pr_number"].isna().sum()
    if no_number > 0:
        logger.warning(
            f"{no_number:,} sha_not_in_repo records have no matching pr_number "
            "in pr_commits.parquet (SHA may predate the chronology run or belong "
            "to a PR that was not crawled).  These cannot be recovered and will "
            "be skipped."
        )
        joined = joined.dropna(subset=["pr_number"])

    joined["pr_number"] = joined["pr_number"].astype(int)

    if joined.empty:
        logger.info("No recoverable targets after pr_number join.")
        return pd.DataFrame()

    # -- 3. Add repo_url ----------------------------------------------------
    universe_path = DATA_DIR / "universe.parquet"
    if universe_path.exists():
        url_df = (
            pd.read_parquet(universe_path)[["pr_id", "repo_url"]]
            .drop_duplicates("pr_id")
        )
        joined = joined.merge(url_df, on="pr_id", how="left")
    else:
        logger.warning(
            "universe.parquet not found — loading repo_url from AIDev parquets."
        )
        try:
            pr_df = pd.read_parquet(AIDEV_DIR / "all_pull_request.parquet")[
                ["id", "repo_url"]
            ].rename(columns={"id": "pr_id"})
            joined = joined.merge(pr_df, on="pr_id", how="left")
        except FileNotFoundError:
            pr_df = pd.read_parquet(AIDEV_DIR / "pull_request.parquet")[
                ["id", "repo_url"]
            ].rename(columns={"id": "pr_id"})
            joined = joined.merge(pr_df, on="pr_id", how="left")

    joined = joined.dropna(subset=["repo_url"])
    if joined.empty:
        logger.info("No recoverable targets after repo_url join.")
        return pd.DataFrame()

    # -- 4. Exclude already-recovered SHAs ----------------------------------
    im_path = DATA_DIR / "internal_merges.parquet"
    if im_path.exists():
        already_done = set(
            map(
                tuple,
                pd.read_parquet(im_path)[["repo_full_name", "merge_sha"]]
                .drop_duplicates()
                .values,
            )
        )
        before = len(joined)
        joined = joined[
            ~joined.apply(
                lambda r: (r["repo_full_name"], r["merge_sha"]) in already_done,
                axis=1,
            )
        ]
        skipped = before - len(joined)
        if skipped:
            logger.info(
                f"Skipping {skipped:,} SHAs already present in "
                "internal_merges.parquet (recovered by a prior run)."
            )

    if joined.empty:
        logger.info("All sha_not_in_repo SHAs have already been recovered.")
        return pd.DataFrame()

    logger.info(
        f"Recovery targets: {len(joined):,} (repo, SHA) pairs across "
        f"{joined['repo_full_name'].nunique():,} repositories."
    )
    return joined


# ---------------------------------------------------------------------------
# Per-repo PR ref fetch
# ---------------------------------------------------------------------------

def _fetch_pr_refs(repo_path: Path, pr_numbers: list) -> None:
    """Fetch ``refs/pull/<N>/head`` for each PR number in one git call.

    A single ``git fetch`` with multiple refspecs is used to minimise
    round-trips.  Individual refspec failures (e.g., the PR ref has been
    deleted on the remote) are non-fatal: the SHA will simply remain
    inaccessible and will be recorded as ``sha_still_not_in_repo``.
    """
    refspecs = [
        f"+refs/pull/{n}/head:refs/pull/{n}/head" for n in pr_numbers
    ]
    my_env = os.environ.copy()
    my_env["GIT_TERMINAL_PROMPT"] = "0"
    my_env["GIT_CEILING_DIRECTORIES"] = str(Path(repo_path).resolve().parent)

    try:
        subprocess.run(
            ["git", "-C", str(repo_path), "fetch", "origin"] + refspecs,
            check=False,         # non-fatal: some refs may have been deleted
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800,
            env=my_env,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout fetching PR refs for {repo_path}.")
    except Exception as exc:
        logger.warning(f"fetch PR refs failed for {repo_path}: {exc}")


# ---------------------------------------------------------------------------
# Per-repo recovery worker
# ---------------------------------------------------------------------------

def _process_repo_recovery(args: tuple):
    """Clone, fetch PR refs, and reprocess all targeted SHAs for one repo.

    Mirrors the structure of ``process_repository`` in extract_aidev_nature.py,
    including the intra-repo SHA cache, but operates only on the subset of
    SHAs recorded as sha_not_in_repo.
    """
    repo_full_name, group, repo_url = args

    repo_path = clone_repo_bare(_normalize_repo_url(repo_url), SCRATCH_DIR)
    if not repo_path:
        logger.error(f"Could not clone/update {repo_full_name}.")
        return repo_full_name, [], [], [], [], [
            {
                "repo_full_name": repo_full_name,
                "pr_id": None,
                "merge_sha": None,
                "error_type": "clone_failed",
                "error_message": "clone_repo_bare returned None during recovery.",
            }
        ]

    # Fetch the specific PR refs required to access the missing SHAs.
    pr_numbers = sorted(group["pr_number"].dropna().astype(int).unique().tolist())
    logger.info(
        f"{repo_full_name}: fetching {len(pr_numbers)} PR refs "
        f"(PRs: {pr_numbers[:5]}{'...' if len(pr_numbers) > 5 else ''})."
    )
    _fetch_pr_refs(repo_path, pr_numbers)

    internal_merges = []
    conflict_chunks = []
    resolved_chunks = []
    classified_chunks = []
    extraction_errors = []

    # SHA cache mirrors the one in extract_aidev_nature.process_repository:
    #   sha -> ("merge", template_without_pr_id)   — successful merge
    #   sha -> ("error", template_without_pr_id)   — failed (any reason)
    #   sha -> ("skip",  None)                     — non-merge commit
    sha_cache: dict = {}

    try:
        for pr_id, pr_group in group.groupby("pr_id"):
            shas = (
                pr_group["merge_sha"].dropna().drop_duplicates().tolist()
            )
            for sha in shas:
                try:
                    # Cache hit: reuse the verdict from the first visit.
                    if sha in sha_cache:
                        kind, payload = sha_cache[sha]
                        if kind == "merge":
                            internal_merges.append({**payload, "pr_id": pr_id})
                        elif kind == "error":
                            extraction_errors.append({**payload, "pr_id": pr_id})
                        # kind == "skip" → silent
                        continue

                    blob = run_git_command(
                        repo_path, "cat-file", "-p", sha, check=False
                    )

                    if not blob:
                        # Still unreachable after fetching PR refs (deleted,
                        # force-pushed further, or the ref pointed to a
                        # different commit).
                        err_template = {
                            "repo_full_name": repo_full_name,
                            "merge_sha": sha,
                            "error_type": "sha_still_not_in_repo",
                            "error_message": (
                                "SHA still not accessible after fetching "
                                "refs/pull/<N>/head for all affected PRs in "
                                "this repository.  The PR ref may have been "
                                "deleted or force-pushed beyond this SHA."
                            ),
                        }
                        extraction_errors.append({**err_template, "pr_id": pr_id})
                        sha_cache[sha] = ("error", err_template)
                        continue

                    parsed = _parse_commit_object(blob)
                    n_parents = len(parsed["parents"])

                    if n_parents < 2:
                        sha_cache[sha] = ("skip", None)
                        continue

                    if n_parents >= 3:
                        err_template = {
                            "repo_full_name": repo_full_name,
                            "merge_sha": sha,
                            "error_type": "octopus_merge",
                            "error_message": (
                                f"Commit has {n_parents} parents; "
                                "excluded from the analysis."
                            ),
                        }
                        extraction_errors.append({**err_template, "pr_id": pr_id})
                        sha_cache[sha] = ("error", err_template)
                        continue

                    if _PR_INTEGRATION_MERGE_RE.search(parsed["message"]):
                        err_template = {
                            "repo_full_name": repo_full_name,
                            "merge_sha": sha,
                            "error_type": "pr_integration_merge",
                            "error_message": (
                                "Excluded GitHub 'Merge pull request #N' "
                                "integration merge."
                            ),
                        }
                        extraction_errors.append({**err_template, "pr_id": pr_id})
                        sha_cache[sha] = ("error", err_template)
                        continue

                    # Full merge processing — identical to Pass B.
                    im, cc, rc, ca, errs = _process_one_merge(
                        repo_path, repo_full_name, pr_id, sha, parsed
                    )
                    internal_merges.append(im)
                    conflict_chunks.extend(cc)
                    resolved_chunks.extend(rc)
                    classified_chunks.extend(ca)
                    extraction_errors.extend(errs)

                    im_template = {k: v for k, v in im.items() if k != "pr_id"}
                    sha_cache[sha] = ("merge", im_template)

                except Exception as exc:
                    extraction_errors.append(
                        {
                            "repo_full_name": repo_full_name,
                            "pr_id": pr_id,
                            "merge_sha": sha,
                            "error_type": "sha_processing_exception",
                            "error_message": str(exc),
                        }
                    )

    finally:
        def _remove_readonly(func, path, excinfo):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)

        try:
            shutil.rmtree(repo_path, onerror=_remove_readonly)
        except Exception as exc:
            logger.warning(f"Failed to remove scratch clone {repo_path}: {exc}")

    return (
        repo_full_name,
        internal_merges,
        conflict_chunks,
        resolved_chunks,
        classified_chunks,
        extraction_errors,
    )


# ---------------------------------------------------------------------------
# Post-processing: clean up stale sha_not_in_repo in extraction_errors
# ---------------------------------------------------------------------------

def _rebuild_errors_parquet(recovered_sha_keys: set) -> None:
    """Remove stale sha_not_in_repo rows from extraction_errors.parquet.

    ``aggregate_jsonl_to_parquet()`` (called just before this function)
    already rebuilt extraction_errors.parquet from the full JSONL, which still
    contains the original sha_not_in_repo entries from Pass B.  This function
    performs a targeted in-place update on the Parquet to remove those stale
    rows for SHAs that were successfully recovered (i.e., now appear in
    internal_merges.parquet), leaving all other rows untouched.

    The JSONL is intentionally not modified: it is an append-only audit log.
    The Parquet is the authoritative view consumed by analysis notebooks.
    """
    parquet_path = DATA_DIR / "extraction_errors.parquet"
    if not parquet_path.exists():
        return

    df = pd.read_parquet(parquet_path)
    if df.empty or not recovered_sha_keys:
        return

    stale_mask = (df["error_type"] == "sha_not_in_repo") & (
        df.apply(
            lambda r: (r.get("repo_full_name"), r.get("merge_sha"))
            in recovered_sha_keys,
            axis=1,
        )
    )
    n_stale = stale_mask.sum()
    if n_stale == 0:
        return

    df = df[~stale_mask]
    df.to_parquet(parquet_path)
    logger.info(
        f"Removed {n_stale:,} stale sha_not_in_repo rows from "
        "extraction_errors.parquet."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Complementary Pass B: recover SHAs recorded as sha_not_in_repo "
            "by fetching the original refs/pull/<N>/head for each affected PR."
        )
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=0,
        metavar="N",
        help="Process only the first N repositories (for testing).",
    )
    parser.add_argument(
        "--pr-commits",
        type=str,
        default=str(DEFAULT_PR_COMMITS),
        metavar="PATH",
        help=(
            f"Path to pr_commits.parquet produced by extract_pr_chronology.py "
            f"(default: {DEFAULT_PR_COMMITS})."
        ),
    )
    args = parser.parse_args()
    pr_commits_path = Path(args.pr_commits)

    logger.info("=== recover_pr_ref_shas.py started ===")
    logger.info(f"Chronology source: {pr_commits_path}")

    # -- Load targets --------------------------------------------------------
    targets = load_recovery_targets(pr_commits_path)
    if targets.empty:
        logger.info("No recovery targets — exiting.")
        return

    already_recovered = _check_recovered()

    repo_groups = []
    for repo_full_name, group in targets.groupby("repo_full_name"):
        if repo_full_name in already_recovered:
            logger.debug(f"Skipping {repo_full_name} (already in recovery tracker).")
            continue
        repo_url = group["repo_url"].iloc[0]
        repo_groups.append((repo_full_name, group, repo_url))

    if not repo_groups:
        logger.info("All repositories already processed by this script — exiting.")
        return

    if args.pilot > 0:
        repo_groups = repo_groups[: args.pilot]
        logger.info(f"Pilot mode: restricted to {len(repo_groups)} repositories.")

    total = len(repo_groups)
    pool_size = max(1, multiprocessing.cpu_count() - 1)
    logger.info(
        f"Recovering {total} repositories with {pool_size} worker(s)..."
    )

    # Tracks (repo_full_name, merge_sha) pairs that were successfully processed
    # so we can remove the corresponding stale sha_not_in_repo rows later.
    recovered_sha_keys: set = set()

    def _handle_result(i, name, im, cc, rc, ca, errs):
        pct = (i + 1) / total
        logger.info(
            f"[{i+1}/{total} ({pct:.1%})] {name}: "
            f"{len(im)} merges, {len(cc)} chunks, {len(errs)} errors"
        )
        append_jsonl("internal_merges.jsonl", im)
        append_jsonl("conflict_chunks.jsonl", cc)
        append_jsonl("resolved_chunks.jsonl", rc)
        append_jsonl("classified_chunks.jsonl", ca)
        append_jsonl("extraction_errors.jsonl", errs)

        for merge_record in im:
            recovered_sha_keys.add(
                (merge_record["repo_full_name"], merge_record["merge_sha"])
            )

        _mark_recovered(name)

    if pool_size == 1:
        for i, rg in enumerate(repo_groups):
            result = _process_repo_recovery(rg)
            _handle_result(i, *result)
    else:
        with multiprocessing.Pool(pool_size) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_process_repo_recovery, repo_groups)
            ):
                _handle_result(i, *result)

    # -- Rebuild all Parquets from the updated JSONLs -----------------------
    logger.info("Aggregating JSONL files to Parquet...")
    aggregate_jsonl_to_parquet()

    # -- Clean up stale sha_not_in_repo rows from extraction_errors ---------
    if recovered_sha_keys:
        logger.info(
            f"Cleaning up {len(recovered_sha_keys):,} stale sha_not_in_repo "
            "rows from extraction_errors.parquet..."
        )
        _rebuild_errors_parquet(recovered_sha_keys)

    # -- Rebuild resolver_labels.parquet ------------------------------------
    im_path = DATA_DIR / "internal_merges.parquet"
    if im_path.exists():
        df_im = pd.read_parquet(im_path)
        if not df_im.empty and "resolver_type" in df_im.columns:
            df_im[
                [
                    "pr_id",
                    "merge_sha",
                    "parent1_sha",
                    "parent2_sha",
                    "author",
                    "committer",
                    "repo_full_name",
                    "resolver_type",
                ]
            ].to_parquet(DATA_DIR / "resolver_labels.parquet")
            logger.info("Rebuilt resolver_labels.parquet.")

    logger.info("=== recovery complete ===")
    logger.info(
        f"Successfully recovered {len(recovered_sha_keys):,} merge SHAs."
    )


if __name__ == "__main__":
    main()
