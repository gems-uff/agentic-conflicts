#!/usr/bin/env python3
"""
Serial retry pass for repositories that failed to clone/fetch during the main
parallel runs of ``extract_pr_chronology.py`` (Pass A) or
``extract_aidev_nature.py`` (Pass B).

Why this script exists
----------------------

Both main passes use a ``multiprocessing.Pool`` so that N workers clone
different repositories in parallel. Under that regime, a transient GitHub hiccup
(DNS blip, 5xx, rate-limit spike, connection reset) looks like a permanent
``clone_failed`` entry in the audit, and the repository is marked as processed
so the main script never revisits it. That over-attributes the failure to the
repository itself.

The right way to distinguish transient from permanent failures is to walk the
failed repositories again, but **serially** (no pool contention), with a delay
between attempts and a small retry budget per repository. Repositories that
still fail after this pass can be safely attributed to something out of our
control -- typically repositories that were deleted, renamed, or turned private
between AIDev's snapshot (August 2025) and our local run.

What it does
------------

For each selected pass:

1. Load the failed repositories from the audit file (``data/pr_chronology/
   errors.jsonl`` for Pass A, ``data/nature_of_agent_conflicts/
   extraction_errors.jsonl`` for Pass B), filtering for
   ``clone_failed`` / ``fetch_failed`` entries only.
2. Exclude repositories that this retry pass has already rescued on a previous
   invocation (tracked in ``retry_success.txt`` next to each pass's output).
3. Build the corresponding universe (full AIDev by default, AIDev-pop under
   ``--pop-only``) and filter it down to the failed repositories.
4. Iterate serially. For each repository, invoke the main pass's
   ``process_repository`` up to ``--max-retries`` times, waiting
   ``--retry-delay`` seconds between attempts.
5. On success (i.e. no ``clone_failed`` / ``fetch_failed`` in the returned
   error list), append the new rows to the same JSONL files the main pass
   writes and record the repository name in ``retry_success.txt``.
6. Re-aggregate JSONL to Parquet at the end of each pass so downstream
   consumers see the rescued data.

Usage
-----

    # Retry Pass A (chronology) failures only
    python retry_clone_failures.py --pass chronology

    # Retry Pass B (merge extraction) failures only
    python retry_clone_failures.py --pass extraction

    # Retry both (chronology first, then extraction)
    python retry_clone_failures.py --pass both

    # More patient: try each repository up to 5 times with 2 minutes between
    python retry_clone_failures.py --max-retries 5 --retry-delay 120
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import extract_aidev_nature as nature
import extract_pr_chronology as chrono

# --- Output directories for this script's own state and logs -----------------

RETRY_DIR = Path("data/retry_clone_failures")
RETRY_LOGS_DIR = RETRY_DIR / "logs"
RETRY_DIR.mkdir(parents=True, exist_ok=True)
RETRY_LOGS_DIR.mkdir(parents=True, exist_ok=True)

CHRONO_SUCCESS_TRACKER = chrono.DATA_DIR / "retry_success.txt"
NATURE_SUCCESS_TRACKER = nature.DATA_DIR / "retry_success.txt"

_CLONE_LEVEL_ERRORS = {"clone_failed", "fetch_failed"}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("retry_clone")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ch = logging.StreamHandler()
    fh = logging.FileHandler(
        RETRY_LOGS_DIR / f"retry_{timestamp}.log",
        mode="w",
        encoding="utf-8",
    )
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# --- State trackers ----------------------------------------------------------


def _load_tracker(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _append_tracker(path: Path, repo_full_name: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{repo_full_name}\n")


# --- Failure discovery -------------------------------------------------------


def load_chronology_failures() -> set:
    """Repositories with clone/fetch failure in Pass A that have no evidence
    of a later successful run.

    "Evidence of success" = at least one row in ``pr_commits.jsonl`` for the
    repository, or presence in this script's ``retry_success.txt``.
    """
    errors_path = chrono.DATA_DIR / "errors.jsonl"
    if not errors_path.exists():
        return set()

    failed = set()
    with open(errors_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") in _CLONE_LEVEL_ERRORS:
                repo = rec.get("repo_full_name")
                if repo:
                    failed.add(repo)

    # Consider successful any repo that ended up with chronology rows.
    commits_path = chrono.DATA_DIR / "pr_commits.jsonl"
    succeeded_from_data = set()
    if commits_path.exists():
        with open(commits_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                repo = rec.get("repo_full_name")
                if repo:
                    succeeded_from_data.add(repo)

    succeeded = succeeded_from_data | _load_tracker(CHRONO_SUCCESS_TRACKER)
    return failed - succeeded


def load_extraction_failures() -> set:
    """Same idea for Pass B. "Evidence of success" = at least one row in any
    of the Pass B chunk tables for the repository, OR presence in the
    ``retry_success.txt`` tracker for Pass B.
    """
    errors_path = nature.DATA_DIR / "extraction_errors.jsonl"
    if not errors_path.exists():
        return set()

    failed = set()
    with open(errors_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error_type") == "clone_failed":
                repo = rec.get("repo_full_name")
                if repo:
                    failed.add(repo)

    succeeded_from_data = set()
    # Any JSONL produced by Pass B is evidence the clone worked at some point.
    for jsonl_name in (
        "internal_merges.jsonl",
        "conflict_chunks.jsonl",
        "resolved_chunks.jsonl",
        "classified_chunks.jsonl",
    ):
        jsonl_path = nature.DATA_DIR / jsonl_name
        if not jsonl_path.exists():
            continue
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                repo = rec.get("repo_full_name")
                if repo:
                    succeeded_from_data.add(repo)

    succeeded = succeeded_from_data | _load_tracker(NATURE_SUCCESS_TRACKER)
    return failed - succeeded


# --- Universe builders for the retry (subset to failed repos) --------------


def _build_chronology_retry_universe(
    failed_repos: set, use_pop_only: bool, logger: logging.Logger
) -> pd.DataFrame:
    if use_pop_only:
        logger.info("Retry universe: AIDev-pop (pull_request + repository).")
        pr_df = pd.read_parquet(chrono.AIDEV_DIR / "pull_request.parquet")
        repo_df = pd.read_parquet(chrono.AIDEV_DIR / "repository.parquet")
    else:
        logger.info("Retry universe: full AIDev (all_pull_request + all_repository).")
        pr_df = pd.read_parquet(chrono.AIDEV_DIR / "all_pull_request.parquet")
        repo_df = pd.read_parquet(chrono.AIDEV_DIR / "all_repository.parquet")

    df = pr_df[["id", "number", "repo_id"]].merge(
        repo_df[["id", "full_name", "url"]],
        left_on="repo_id",
        right_on="id",
    )
    df = df.drop(columns=["id_y"] if "id_y" in df.columns else ["id"])
    df = df.rename(columns={"id_x": "pr_id", "url": "repo_url", "id": "pr_id"})

    df = df[df["full_name"].isin(failed_repos)]
    return df


def _build_extraction_retry_universe(
    failed_repos: set,
    use_pop_only: bool,
    pr_commits_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    # Reuse the main script's build_universe for fidelity: it sets the same
    # columns and does the same joins with pr_task_type, etc.
    universe_df = nature.build_universe(
        is_pilot=False,
        pilot_count=0,
        use_pop_only=use_pop_only,
        pr_commits_path=pr_commits_path,
    )
    return universe_df[universe_df["full_name"].isin(failed_repos)].copy()


# --- Retry loops -------------------------------------------------------------


def _retry_one_repo_chronology(
    name: str,
    group: pd.DataFrame,
    max_retries: int,
    retry_delay: int,
    logger: logging.Logger,
) -> bool:
    """Return True if clone worked on some attempt."""
    for attempt in range(max_retries):
        if attempt > 0:
            logger.info(
                f"  {name}: waiting {retry_delay}s before attempt "
                f"{attempt + 1}/{max_retries}"
            )
            time.sleep(retry_delay)
        try:
            _, chronology, errs = chrono.process_repository((name, group))
        except Exception as e:
            logger.error(f"  {name}: exception on attempt {attempt + 1}: {e}")
            continue

        clone_failed = any(
            e.get("error") in _CLONE_LEVEL_ERRORS for e in errs
        )
        if clone_failed:
            logger.warning(f"  {name}: clone/fetch failed again on attempt {attempt + 1}")
            continue

        # Success (even if some per-PR errors were produced)
        chrono.append_jsonl("pr_commits.jsonl", chronology)
        chrono.append_jsonl("errors.jsonl", errs)
        _append_tracker(CHRONO_SUCCESS_TRACKER, name)
        logger.info(
            f"  {name}: OK on attempt {attempt + 1} "
            f"({len(chronology)} commits, {len(errs)} per-PR notes)"
        )
        return True
    return False


def _retry_one_repo_extraction(
    name: str,
    group: pd.DataFrame,
    max_retries: int,
    retry_delay: int,
    logger: logging.Logger,
) -> bool:
    for attempt in range(max_retries):
        if attempt > 0:
            logger.info(
                f"  {name}: waiting {retry_delay}s before attempt "
                f"{attempt + 1}/{max_retries}"
            )
            time.sleep(retry_delay)
        try:
            _, im, cc, rc, ca, errs = nature.process_repository((name, group))
        except Exception as e:
            logger.error(f"  {name}: exception on attempt {attempt + 1}: {e}")
            continue

        clone_failed = any(
            e.get("error_type") == "clone_failed" for e in errs
        )
        if clone_failed:
            logger.warning(f"  {name}: clone failed again on attempt {attempt + 1}")
            continue

        nature.append_jsonl("internal_merges.jsonl", im)
        nature.append_jsonl("conflict_chunks.jsonl", cc)
        nature.append_jsonl("resolved_chunks.jsonl", rc)
        nature.append_jsonl("classified_chunks.jsonl", ca)
        nature.append_jsonl("extraction_errors.jsonl", errs)
        _append_tracker(NATURE_SUCCESS_TRACKER, name)
        logger.info(
            f"  {name}: OK on attempt {attempt + 1} "
            f"({len(im)} merges, {len(ca)} classified chunks, {len(errs)} notes)"
        )
        return True
    return False


def retry_chronology(
    logger: logging.Logger,
    failed_repos: set,
    use_pop_only: bool,
    max_retries: int,
    retry_delay: int,
) -> None:
    if not failed_repos:
        logger.info("Pass A retry: no clone/fetch failures pending.")
        return

    logger.info(f"Pass A retry: {len(failed_repos)} repositories to revisit.")
    universe = _build_chronology_retry_universe(failed_repos, use_pop_only, logger)
    logger.info(
        f"Pass A retry: universe has {len(universe)} PRs across "
        f"{universe['full_name'].nunique()} repositories."
    )

    groups = list(universe.groupby("full_name"))
    total = len(groups)
    rescued = 0
    for i, (name, group) in enumerate(groups, start=1):
        logger.info(f"[Pass A {i}/{total}] Retrying {name}...")
        if _retry_one_repo_chronology(name, group, max_retries, retry_delay, logger):
            rescued += 1

    logger.info(
        f"Pass A retry complete: rescued {rescued}/{total} repositories. "
        f"Aggregating JSONL to Parquet..."
    )
    chrono.aggregate_to_parquet()


def retry_extraction(
    logger: logging.Logger,
    failed_repos: set,
    use_pop_only: bool,
    pr_commits_path: Path,
    max_retries: int,
    retry_delay: int,
) -> None:
    if not failed_repos:
        logger.info("Pass B retry: no clone failures pending.")
        return

    logger.info(f"Pass B retry: {len(failed_repos)} repositories to revisit.")
    universe = _build_extraction_retry_universe(
        failed_repos, use_pop_only, pr_commits_path, logger
    )
    if universe.empty:
        logger.warning(
            "Pass B retry: failed repositories are not present in the chronology; "
            "they must be recovered by Pass A retry first before Pass B can process them."
        )
        return
    logger.info(
        f"Pass B retry: universe has {len(universe)} (pr_id, sha) rows across "
        f"{universe['full_name'].nunique()} repositories."
    )

    groups = list(universe.groupby("full_name"))
    total = len(groups)
    rescued = 0
    for i, (name, group) in enumerate(groups, start=1):
        logger.info(f"[Pass B {i}/{total}] Retrying {name}...")
        if _retry_one_repo_extraction(name, group, max_retries, retry_delay, logger):
            rescued += 1

    logger.info(
        f"Pass B retry complete: rescued {rescued}/{total} repositories. "
        f"Aggregating JSONL to Parquet..."
    )
    nature.aggregate_jsonl_to_parquet()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serial retry over repositories that failed to clone or fetch "
            "during the main parallel pipelines. Succeeds only if the clone "
            "now works; rescued data is merged into the main outputs."
        )
    )
    parser.add_argument(
        "--pass",
        dest="which_pass",
        choices=["chronology", "extraction", "both"],
        default="both",
        help="Which pipeline to retry (default: both).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum clone attempts per repository (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=30,
        help="Seconds to wait between attempts on the same repository (default: 30).",
    )
    parser.add_argument(
        "--pop-only",
        action="store_true",
        help="Restrict to AIDev-pop (stars > 100). Affects which AIDev tables are "
        "loaded when building the retry universe.",
    )
    parser.add_argument(
        "--pr-commits",
        type=str,
        default=str(nature.DEFAULT_PR_COMMITS),
        help=f"Path to the chronology parquet consumed by Pass B "
        f"(default: {nature.DEFAULT_PR_COMMITS}).",
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info(
        f"Retry configuration: pass={args.which_pass}, max_retries={args.max_retries}, "
        f"retry_delay={args.retry_delay}s, pop_only={args.pop_only}."
    )

    if args.which_pass in ("chronology", "both"):
        failed_a = load_chronology_failures()
        logger.info(f"Pass A: {len(failed_a)} candidate clone/fetch failures.")
        retry_chronology(
            logger,
            failed_a,
            use_pop_only=args.pop_only,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

    if args.which_pass in ("extraction", "both"):
        failed_b = load_extraction_failures()
        logger.info(f"Pass B: {len(failed_b)} candidate clone failures.")
        retry_extraction(
            logger,
            failed_b,
            use_pop_only=args.pop_only,
            pr_commits_path=Path(args.pr_commits),
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

    logger.info("Retry pass complete.")


if __name__ == "__main__":
    main()
