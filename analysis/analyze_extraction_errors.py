"""analyze_extraction_errors.py
-------------------------------
Quantify and classify extraction errors from both pipeline passes for
inclusion in the paper's Dataset Overview / Threats section.

Pass A  →  data/pr_chronology/errors.parquet
            one row per (repo, pr) combination that failed during the
            commit-chronology extraction step.

Pass B  →  data/nature_of_agent_conflicts/extraction_errors.parquet
            one row per (repo, pr, sha) combination that failed or was
            intentionally excluded during the conflict-extraction step.

Run on the analysis machine:

    cd /path/to/agentic-conflicts
    python -m analysis.analyze_extraction_errors

or, if the project root is on sys.path:

    python analysis/analyze_extraction_errors.py

Output is printed to stdout and also written to
``analysis/results/extraction_errors_report.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NATURE_DIR   = PROJECT_ROOT / "data" / "nature_of_agent_conflicts"
CHRONO_DIR   = PROJECT_ROOT / "data" / "pr_chronology"
RESULTS_DIR  = PROJECT_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARNING] File not found: {path}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"  Loaded {path.name}: {len(df):,} rows, columns={list(df.columns)}")
    return df


def _pct(n: int, total: int, decimals: int = 1) -> str:
    if total == 0:
        return "N/A"
    return f"{100 * n / total:.{decimals}f}%"


def _section(title: str, lines: list[str]) -> str:
    bar = "=" * 60
    return "\n".join(["", bar, title, bar] + lines) + "\n"


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

# Pass B error types that represent *deliberate exclusions* (not failures).
# These are commits that exist and were processed, but intentionally removed
# from the analysis perimeter before or after conflict extraction.
PASS_B_EXCLUDED_BY_DESIGN = {
    "pr_integration_merge",  # GitHub "Merge pull request #N" commits
    "octopus_merge",         # ≥3-parent merges; not supported by diff3
}

# Pass B error types that represent *pipeline failures* (unexpected).
PASS_B_FAILURES = {
    "clone_failed",             # could not clone or update the bare repo
    "no_sha_for_pr",            # PR had no chronology output (Pass A not run yet, or 0 own commits)
    "sha_not_in_repo",          # SHA vanished (force-push, history rewrite)
    "localize_timeout",         # LOCALIZERESREGION exceeded time budget
    "file_processing_exception",    # unexpected exception during file diff
    "sha_processing_exception",     # unexpected exception during SHA processing
    "pr_processing_exception",      # unexpected exception during PR loop
}


# ---------------------------------------------------------------------------
# Pass A analysis  (pr_chronology/errors.parquet)
# ---------------------------------------------------------------------------

def analyze_pass_a(df: pd.DataFrame, universe_prs: int, universe_repos: int) -> list[str]:
    """Characterise chronology-extraction errors (Pass A)."""
    lines: list[str] = []

    if df.empty:
        lines.append("  [no data]")
        return lines

    # Normalise: the 'error' column may contain either a short tag or a
    # free-text exception string.  We keep the short tags and bucket the
    # rest as 'other_exception'.
    KNOWN_TAGS = {"clone_failed", "fetch_failed", "pr_ref_not_found", "no_merge_base"}
    if "error" in df.columns:
        df = df.copy()
        df["error_tag"] = df["error"].where(df["error"].isin(KNOWN_TAGS), other="other_exception")
    else:
        lines.append("  [WARNING] 'error' column not found")
        return lines

    total_rows   = len(df)
    repos_hit    = df["repo_full_name"].nunique() if "repo_full_name" in df.columns else "?"
    prs_hit      = df["pr_id"].dropna().nunique() if "pr_id" in df.columns else "?"

    lines.append(f"  Total error rows            : {total_rows:,}")
    lines.append(f"  Distinct repos affected     : {repos_hit:,}  ({_pct(repos_hit, universe_repos)} of universe repos)")
    lines.append(f"  Distinct PRs affected       : {prs_hit:,}  ({_pct(prs_hit, universe_prs)} of universe PRs)")
    lines.append("")
    lines.append("  Breakdown by error tag:")

    vc = df["error_tag"].value_counts()
    for tag, count in vc.items():
        pr_count = df.loc[df["error_tag"] == tag, "pr_id"].dropna().nunique() if "pr_id" in df.columns else "?"
        lines.append(f"    {tag:<35} {count:>8,} rows  ({pr_count:>6,} PRs)")

    return lines


# ---------------------------------------------------------------------------
# Pass B analysis  (nature_of_agent_conflicts/extraction_errors.parquet)
# ---------------------------------------------------------------------------

def analyze_pass_b(
    df: pd.DataFrame,
    universe_prs: int,
    universe_repos: int,
    internal_merges: int,
) -> list[str]:
    """Characterise conflict-extraction errors and exclusions (Pass B)."""
    lines: list[str] = []

    if df.empty:
        lines.append("  [no data]")
        return lines

    if "error_type" not in df.columns:
        lines.append("  [WARNING] 'error_type' column not found")
        return lines

    df = df.copy()
    df["category"] = df["error_type"].apply(
        lambda t: "excluded_by_design" if t in PASS_B_EXCLUDED_BY_DESIGN
        else ("failure" if t in PASS_B_FAILURES else "unknown")
    )

    total_rows = len(df)

    # Unique physical merges per category (where merge_sha is available)
    excl_mask    = df["category"] == "excluded_by_design"
    failure_mask = df["category"] == "failure"

    excl_rows    = excl_mask.sum()
    failure_rows = failure_mask.sum()

    excl_shas    = df.loc[excl_mask,    "merge_sha"].dropna().nunique() if "merge_sha" in df.columns else "?"
    failure_shas = df.loc[failure_mask, "merge_sha"].dropna().nunique() if "merge_sha" in df.columns else "?"
    excl_repos   = df.loc[excl_mask,    "repo_full_name"].nunique() if "repo_full_name" in df.columns else "?"
    failure_repos= df.loc[failure_mask, "repo_full_name"].nunique() if "repo_full_name" in df.columns else "?"
    excl_prs     = df.loc[excl_mask,    "pr_id"].dropna().nunique() if "pr_id" in df.columns else "?"
    failure_prs  = df.loc[failure_mask, "pr_id"].dropna().nunique() if "pr_id" in df.columns else "?"

    lines.append(f"  Total error/exclusion rows  : {total_rows:,}")
    lines.append(f"    Excluded by design        : {excl_rows:,} rows  ({excl_shas:,} unique SHAs,  {excl_prs:,} PRs,  {excl_repos:,} repos)")
    lines.append(f"    Pipeline failures         : {failure_rows:,} rows  ({failure_shas:,} unique SHAs,  {failure_prs:,} PRs,  {failure_repos:,} repos)")
    lines.append("")

    # ---- Excluded-by-design breakdown ----
    lines.append("  Excluded-by-design breakdown:")
    for tag in sorted(PASS_B_EXCLUDED_BY_DESIGN):
        mask = df["error_type"] == tag
        n    = mask.sum()
        shas = df.loc[mask, "merge_sha"].dropna().nunique() if "merge_sha" in df.columns else "?"
        prs  = df.loc[mask, "pr_id"].dropna().nunique() if "pr_id" in df.columns else "?"
        lines.append(f"    {tag:<35} {n:>8,} rows  ({shas:>6,} unique SHAs,  {prs:>6,} PRs)")

    lines.append("")

    # ---- Failure breakdown ----
    lines.append("  Failure breakdown:")
    for tag in sorted(PASS_B_FAILURES):
        mask = df["error_type"] == tag
        n    = mask.sum()
        shas = df.loc[mask, "merge_sha"].dropna().nunique() if "merge_sha" in df.columns else "?"
        prs  = df.loc[mask, "pr_id"].dropna().nunique() if "pr_id" in df.columns else "?"
        repos= df.loc[mask, "repo_full_name"].nunique() if "repo_full_name" in df.columns else "?"
        lines.append(f"    {tag:<35} {n:>8,} rows  ({shas:>6,} unique SHAs,  {prs:>6,} PRs,  {repos:>5,} repos)")

    # Any unrecognised error types
    unknown = df[df["category"] == "unknown"]["error_type"].value_counts()
    if not unknown.empty:
        lines.append("")
        lines.append("  Unrecognised error types (may need classification):")
        for tag, count in unknown.items():
            lines.append(f"    {tag:<35} {count:>8,} rows")

    lines.append("")

    # ---- Impact relative to universe ----
    all_fail_prs  = df.loc[failure_mask, "pr_id"].dropna().nunique() if "pr_id" in df.columns else 0
    all_fail_repos = df.loc[failure_mask, "repo_full_name"].nunique() if "repo_full_name" in df.columns else 0
    lines.append("  Impact of pipeline failures relative to universe:")
    lines.append(f"    PRs affected              : {all_fail_prs:,}  ({_pct(all_fail_prs,  universe_prs)}  of universe PRs)")
    lines.append(f"    Repos affected            : {all_fail_repos:,}  ({_pct(all_fail_repos, universe_repos)} of universe repos)")

    # clone_failed is the most impactful failure (entire repo lost)
    clone_mask  = df["error_type"] == "clone_failed"
    clone_repos = df.loc[clone_mask, "repo_full_name"].nunique() if "repo_full_name" in df.columns else 0
    clone_prs   = df.loc[clone_mask, "pr_id"].dropna().nunique() if "pr_id" in df.columns else 0
    lines.append(f"    Repos lost to clone_failed: {clone_repos:,}  ({_pct(clone_repos, universe_repos)} of universe repos)")
    lines.append(f"    PRs  lost to clone_failed : {clone_prs:,}  ({_pct(clone_prs, universe_prs)} of universe PRs)")

    return lines


# ---------------------------------------------------------------------------
# Cross-table summary (numbers for the paper)
# ---------------------------------------------------------------------------

def paper_summary(
    pass_a: pd.DataFrame,
    pass_b: pd.DataFrame,
    universe_prs: int,
    universe_repos: int,
    internal_merges: int,
) -> list[str]:
    """One-stop section with the headline numbers cited in the paper."""
    lines: list[str] = []

    # Pass A totals
    a_total_rows = len(pass_a)
    a_prs_lost   = pass_a["pr_id"].dropna().nunique() if not pass_a.empty and "pr_id" in pass_a.columns else 0

    # Pass B totals (split by category)
    if not pass_b.empty and "error_type" in pass_b.columns:
        excl_mask    = pass_b["error_type"].isin(PASS_B_EXCLUDED_BY_DESIGN)
        failure_mask = pass_b["error_type"].isin(PASS_B_FAILURES)

        b_excl_shas    = pass_b.loc[excl_mask,    "merge_sha"].dropna().nunique() if "merge_sha" in pass_b.columns else 0
        b_failure_prs  = pass_b.loc[failure_mask, "pr_id"].dropna().nunique()     if "pr_id"     in pass_b.columns else 0
        b_failure_repos= pass_b.loc[failure_mask, "repo_full_name"].nunique()      if "repo_full_name" in pass_b.columns else 0

        # Octopus: interesting for paper
        octopus_shas = pass_b.loc[pass_b["error_type"] == "octopus_merge", "merge_sha"].dropna().nunique() \
                       if "merge_sha" in pass_b.columns else 0
        integ_shas   = pass_b.loc[pass_b["error_type"] == "pr_integration_merge", "merge_sha"].dropna().nunique() \
                       if "merge_sha" in pass_b.columns else 0

        # Timeout: fine-grained (affects only chunks with conflicting regions)
        timeout_rows  = (pass_b["error_type"] == "localize_timeout").sum()
        timeout_shas  = pass_b.loc[pass_b["error_type"] == "localize_timeout", "merge_sha"].dropna().nunique() \
                        if "merge_sha" in pass_b.columns else 0
    else:
        b_excl_shas = b_failure_prs = b_failure_repos = 0
        octopus_shas = integ_shas = timeout_rows = timeout_shas = 0

    lines.append("  --- Pass A (chronology extraction) ---")
    lines.append(f"  Error rows total            : {a_total_rows:,}")
    lines.append(f"  PRs lost (at least 1 error) : {a_prs_lost:,}  ({_pct(a_prs_lost, universe_prs)} of universe)")
    lines.append("")
    lines.append("  --- Pass B (conflict extraction) ---")
    lines.append(f"  Excluded by design")
    lines.append(f"    pr_integration_merge SHAs : {integ_shas:,}  (GitHub bot-generated merge commits, never content conflicts)")
    lines.append(f"    octopus_merge SHAs        : {octopus_shas:,}  (≥3-parent merges; diff3 undefined)")
    lines.append(f"    Total excluded SHAs       : {b_excl_shas:,}")
    lines.append(f"  Pipeline failures")
    lines.append(f"    PRs affected              : {b_failure_prs:,}  ({_pct(b_failure_prs,  universe_prs)} of universe)")
    lines.append(f"    Repos affected            : {b_failure_repos:,}  ({_pct(b_failure_repos, universe_repos)} of universe)")
    lines.append(f"    localize_timeout rows     : {timeout_rows:,}  ({timeout_shas:,} unique merge SHAs)")
    lines.append("")
    lines.append("  --- Retained corpus ---")
    lines.append(f"  Internal merge commits kept : {internal_merges:,}")
    lines.append(f"  Universe PRs                : {universe_prs:,}")
    lines.append(f"  Universe repos              : {universe_repos:,}")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading tables…")

    # Load error tables
    pass_a_df = _read(CHRONO_DIR  / "errors.parquet")
    pass_b_df = _read(NATURE_DIR  / "extraction_errors.parquet")

    # Load universe for denominators
    universe_df      = _read(NATURE_DIR / "universe.parquet")
    internal_merges_df = _read(NATURE_DIR / "internal_merges.parquet")

    # Denominators
    universe_prs    = universe_df["pr_id"].nunique()  if not universe_df.empty    and "pr_id"          in universe_df.columns    else 0
    universe_repos  = universe_df["full_name"].nunique() if not universe_df.empty and "full_name"       in universe_df.columns    else 0
    # Fallback: try 'repo_full_name' if 'full_name' not present
    if universe_repos == 0 and not universe_df.empty and "repo_full_name" in universe_df.columns:
        universe_repos = universe_df["repo_full_name"].nunique()

    # For internal merges, use dedup count (natural key = repo_full_name + merge_sha)
    if not internal_merges_df.empty and "repo_full_name" in internal_merges_df.columns and "merge_sha" in internal_merges_df.columns:
        internal_merges = internal_merges_df.drop_duplicates(
            subset=["repo_full_name", "merge_sha"]
        ).shape[0]
    else:
        internal_merges = len(internal_merges_df)

    print(f"\n  Universe: {universe_prs:,} PRs, {universe_repos:,} repos, {internal_merges:,} internal merges (dedup)\n")

    report_sections: list[str] = []

    report_sections.append(_section(
        "PASS A — PR CHRONOLOGY ERRORS  (data/pr_chronology/errors.parquet)",
        analyze_pass_a(pass_a_df, universe_prs, universe_repos),
    ))

    report_sections.append(_section(
        "PASS B — CONFLICT EXTRACTION ERRORS  (data/nature_of_agent_conflicts/extraction_errors.parquet)",
        analyze_pass_b(pass_b_df, universe_prs, universe_repos, internal_merges),
    ))

    report_sections.append(_section(
        "PAPER HEADLINE NUMBERS",
        paper_summary(pass_a_df, pass_b_df, universe_prs, universe_repos, internal_merges),
    ))

    report = "\n".join(report_sections)
    print(report)

    out_path = RESULTS_DIR / "extraction_errors_report.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
