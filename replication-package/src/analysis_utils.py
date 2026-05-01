"""Shared analysis utilities for merge conflict resolution study.

Includes mining orchestration functions and result post-processing.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from .mining_utils import (
    clone_repo_bare,
    run_git_command,
    parse_diff3_chunks,
    run_merge_file,
    find_resolution,
    _parse_commit_object,
    MERGE_TREE_CONFLICT_REGEX,
    time_limit,
    FunctionTimeoutError,
    LOCALIZE_TIMEOUT_S,
    _PR_INTEGRATION_MERGE_RE,
)
from .strategies_utils import identify_resolution, remove_empty_lines


def cleanup_repo_scratch(repo_path: Path) -> bool:
    """Delete a bare repository from scratch directory with robust error handling.

    Args:
        repo_path: Path to the bare repository directory

    Returns:
        True if deletion succeeded, False otherwise
    """
    if not repo_path.exists():
        return True  # Already gone

    def _remove_readonly(func, path, excinfo):
        """Error handler for rmtree to handle read-only files on Windows."""
        import os
        import stat
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception as e:
            logging.warning(f"Failed to change permissions on {path}: {e}")

    try:
        shutil.rmtree(repo_path, onerror=_remove_readonly)
        repo_name = repo_path.name
        repo_size_gb = sum(f.stat().st_size for f in repo_path.rglob('*')) / (1024**3) if repo_path.exists() else 0
        logging.info(f"Cleaned up scratch: {repo_name} ({repo_size_gb:.2f} GB)")
        return True
    except Exception as e:
        logging.warning(f"Failed to remove {repo_path}: {e}")
        return False


def is_bot_signature(text: str) -> bool:
    """Classify if a string matches typical agent/bot signatures."""
    t = text.lower()
    bot_markers = ["[bot]", "-bot", "copilot", "devin", "claude", "cursor", "openai"]
    if t.endswith("bot"):
        return True
    return any(marker in t for marker in bot_markers)


def classify_resolver(author: str, message: str) -> str:
    """Classify resolver into agent, agent-assisted, or human."""
    if is_bot_signature(author):
        return "agent"

    lines = message.splitlines()
    for line in lines:
        if line.lower().startswith("co-authored-by:"):
            if is_bot_signature(line):
                return "agent-assisted"

    return "human"


def _normalize_repo_url(repo_url: str) -> str:
    """Normalize repo URL to https://github.com/<owner>/<repo>.git format."""
    if not isinstance(repo_url, str):
        return repo_url
    url = repo_url.strip()
    if url.startswith("https://api.github.com/repos/"):
        url = url.replace("https://api.github.com/repos/", "https://github.com/")
    if not url.endswith(".git"):
        url += ".git"
    return url


def _process_one_merge(repo_path: Path, repo_full_name: str, pr_id: str, sha: str, parsed: Dict) -> Tuple:
    """Process a single internal merge commit."""
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


def process_single_repository(repo_info: Tuple, scratch_dir: Path = None) -> Tuple:
    """Process all merges in a single repository.

    Args:
        repo_info: Tuple of (repo_full_name, repo_df)
        scratch_dir: Directory for bare clones (required)
    """
    repo_full_name, repo_df = repo_info
    repo_url = _normalize_repo_url(repo_df.iloc[0]['repo_url'])

    if scratch_dir is None:
        raise ValueError("scratch_dir must be provided")

    repo_path = clone_repo_bare(repo_url, scratch_dir)
    if not repo_path:
        logging.error(f"Could not clone/update repository {repo_full_name}")
        return repo_full_name, [], [], [], [], [{
            "repo_full_name": repo_full_name,
            "pr_id": None,
            "merge_sha": None,
            "error_type": "clone_failed",
            "error_message": "Failed to clone or update repository",
        }]

    internal_merges = []
    conflict_chunks = []
    resolved_chunks = []
    classified_chunks = []
    extraction_errors = []

    sha_cache = {}

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
                        "error_message": "PR has no entries in chronology (not yet processed)",
                    })
                    continue

                for sha in shas:
                    if sha in sha_cache:
                        kind, payload = sha_cache[sha]
                        if kind == "merge":
                            internal_merges.append({**payload, "pr_id": pr_id})
                        elif kind == "error":
                            extraction_errors.append({**payload, "pr_id": pr_id})
                        continue

                    blob = run_git_command(repo_path, "cat-file", "-p", sha, check=False)
                    if not blob:
                        err_template = {
                            "repo_full_name": repo_full_name,
                            "merge_sha": sha,
                            "error_type": "sha_not_in_repo",
                            "error_message": "git cat-file returned empty",
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
                            "error_message": f"Commit has {n_parents} parents (excluded)",
                        }
                        extraction_errors.append({**err_template, "pr_id": pr_id})
                        sha_cache[sha] = ("error", err_template)
                        continue

                    if _PR_INTEGRATION_MERGE_RE.search(parsed["message"]):
                        err_template = {
                            "repo_full_name": repo_full_name,
                            "merge_sha": sha,
                            "error_type": "pr_integration_merge",
                            "error_message": "GitHub 'Merge pull request' integration merge (excluded)",
                        }
                        extraction_errors.append({**err_template, "pr_id": pr_id})
                        sha_cache[sha] = ("error", err_template)
                        continue

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

            except Exception as e:
                extraction_errors.append({
                    "repo_full_name": repo_full_name,
                    "pr_id": pr_id,
                    "merge_sha": None,
                    "error_type": "pr_processing_exception",
                    "error_message": str(e),
                })
    finally:
        cleanup_repo_scratch(repo_path)

    return (
        repo_full_name,
        internal_merges,
        conflict_chunks,
        resolved_chunks,
        classified_chunks,
        extraction_errors,
    )


def append_jsonl(filename: str, records: List, data_dir: Path):
    """Append records to a JSONL file."""
    if not records:
        return
    with open(data_dir / filename, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


_DEDUP_KEYS = {
    "internal_merges.jsonl": ["repo_full_name", "pr_id", "merge_sha"],
    "conflict_chunks.jsonl": ["repo_full_name", "merge_sha", "file_path", "chunk_index"],
    "resolved_chunks.jsonl": ["repo_full_name", "merge_sha", "file_path", "chunk_index"],
    "classified_chunks.jsonl": ["repo_full_name", "merge_sha", "file_path", "chunk_index"],
    "extraction_errors.jsonl": None,
}


def aggregate_jsonl_to_parquet(data_dir: Path):
    """Convert incremental JSONL files to final parquet format with deduplication."""
    logging.info("Aggregating incremental JSONL files to Parquet...")
    files_map = {
        "internal_merges.jsonl": "internal_merges.parquet",
        "conflict_chunks.jsonl": "conflict_chunks.parquet",
        "resolved_chunks.jsonl": "resolved_chunks.parquet",
        "classified_chunks.jsonl": "classified_chunks.parquet",
        "extraction_errors.jsonl": "extraction_errors.parquet",
    }

    for jsonl_name, parquet_name in files_map.items():
        jsonl_path = data_dir / jsonl_name
        if not jsonl_path.exists():
            continue
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed line in {jsonl_name}")
        df = pd.DataFrame(records)
        keys = _DEDUP_KEYS.get(jsonl_name)
        if keys and not df.empty and all(k in df.columns for k in keys):
            before = len(df)
            df = df.drop_duplicates(subset=keys, keep="first")
            after = len(df)
            if before != after:
                logging.info(f"  Dedup {jsonl_name}: {before:,} -> {after:,} rows")
        df.to_parquet(data_dir / parquet_name)
        logging.info(f"Converted {jsonl_name} -> {parquet_name} ({len(df)} rows)")


def load_results(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all parquet result files from data directory."""
    results = {}
    files_to_load = [
        'universe.parquet',
        'internal_merges.parquet',
        'conflict_chunks.parquet',
        'classified_chunks.parquet',
        'resolver_labels.parquet',
    ]
    for fname in files_to_load:
        path = f'{data_dir}/{fname}'
        try:
            results[fname.replace('.parquet', '')] = pd.read_parquet(path)
        except FileNotFoundError:
            pass
    return results


def compute_self_resolution_rate(
    resolver_labels: pd.DataFrame,
    by: Optional[str] = None
) -> pd.DataFrame:
    """Compute self-resolution rate (conflicts resolved by agent)."""
    if by is None:
        total = len(resolver_labels)
        agent_resolved = (resolver_labels['resolver'] == 'agent').sum()
        return pd.DataFrame({
            'self_resolution_rate': [agent_resolved / total if total > 0 else 0]
        })
    return resolver_labels.groupby(by).apply(
        lambda g: (g['resolver'] == 'agent').sum() / len(g)
    ).reset_index(name='self_resolution_rate')


def compute_strategy_distribution(
    classified_chunks: pd.DataFrame,
    by: Optional[str] = None
) -> pd.DataFrame:
    """Compute distribution of resolution strategies."""
    if by is None:
        return classified_chunks['strategy'].value_counts().to_frame('count')
    return classified_chunks.groupby([by, 'strategy']).size().unstack(fill_value=0)


__all__ = [
    "classify_resolver",
    "_normalize_repo_url",
    "_process_one_merge",
    "process_single_repository",
    "append_jsonl",
    "aggregate_jsonl_to_parquet",
    "load_results",
    "compute_self_resolution_rate",
    "compute_strategy_distribution",
]
