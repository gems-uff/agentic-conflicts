#!/usr/bin/env python3
"""
Pipeline for Phase 1: Nature of Agent Conflicts
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from collect import clone_repo_bare, run_git_command, run_merge_file, parse_diff3_chunks, find_resolution, MERGE_TREE_CONFLICT_REGEX
from extract_resolution_strategies import identify_resolution, remove_empty_lines

# --- Configuration ---
AIDEV_DIR = Path("AIDev")
DATA_DIR = Path("data/nature_of_agent_conflicts")
SCRATCH_DIR = DATA_DIR / "scratch"
LOGS_DIR = DATA_DIR / "logs"

for d in [DATA_DIR, SCRATCH_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configure custom logger to prevent hijack
logger = logging.getLogger("extract_aidev")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler()
from datetime import datetime

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
    if t.endswith("bot"): return True
    return any(marker in t for marker in bot_markers)

def classify_resolver(author: str, message: str) -> str:
    """Classify resolver into agent, agent-assisted, human based on author and commit message."""
    if is_bot_signature(author):
        return "agent"
    
    # Analyze commit message for Co-authored-by
    lines = message.splitlines()
    for line in lines:
        if line.lower().startswith("co-authored-by:"):
            if is_bot_signature(line):
                return "agent-assisted"
    
    return "human"

def build_universe(is_pilot: bool, pilot_count: int, use_full_aidev: bool = False) -> pd.DataFrame:
    logger.info("Stage 0: Building universe...")

    if use_full_aidev:
        logger.info("Full-AIDev mode: using all_pull_request.parquet + all_repository.parquet")
        pr_df   = pd.read_parquet(AIDEV_DIR / "all_pull_request.parquet")
        repo_df = pd.read_parquet(AIDEV_DIR / "all_repository.parquet")
    else:
        pr_df   = pd.read_parquet(AIDEV_DIR / "pull_request.parquet")
        repo_df = pd.read_parquet(AIDEV_DIR / "repository.parquet")

    try:
        task_df = pd.read_parquet(AIDEV_DIR / "pr_task_type.parquet")
    except Exception:
        task_df = pd.DataFrame(columns=['id', 'type'])

    pr_repo_df = pr_df.merge(repo_df[['id', 'full_name', 'language']], left_on='repo_id', right_on='id', suffixes=('', '_repo'))

    if not task_df.empty:
        pr_repo_task_df = pr_repo_df.merge(task_df[['id', 'type']], on='id', how='left')
        pr_repo_task_df.rename(columns={'type': 'pr_task_type'}, inplace=True)
    else:
        pr_repo_task_df = pr_repo_df.copy()
        pr_repo_task_df['pr_task_type'] = None

    # 'number' (PR number within repo) is needed for git-log matching in full-AIDev mode
    cols_to_keep = ['id', 'number', 'repo_url', 'full_name', 'language', 'agent', 'pr_task_type', 'state']
    if 'merged_at' in pr_repo_task_df.columns:
        cols_to_keep.append('merged_at')
    cols_to_keep = [c for c in cols_to_keep if c in pr_repo_task_df.columns]

    if use_full_aidev:
        # PR-level universe: left-join with pr_commits so AIDev-pop repos keep their
        # pre-collected SHAs while new repos get sha=NaN (handled in process_repository).
        pr_level = pr_repo_task_df[cols_to_keep].copy()
        pr_level = pr_level.rename(columns={'id': 'pr_id'})

        # Only merged PRs are relevant
        if 'merged_at' in pr_level.columns:
            pr_level = pr_level[pr_level['merged_at'].notna()]

        try:
            commits_df = pd.read_parquet(AIDEV_DIR / "pr_commits.parquet")
            universe_df = pr_level.merge(commits_df[['pr_id', 'sha']], on='pr_id', how='left')
        except Exception:
            universe_df = pr_level.copy()
            universe_df['sha'] = None
    else:
        commits_df = pd.read_parquet(AIDEV_DIR / "pr_commits.parquet")
        universe_df = commits_df.merge(
            pr_repo_task_df[cols_to_keep],
            left_on='pr_id', right_on='id'
        )

    if is_pilot:
        unique_repos = universe_df['full_name'].dropna().unique()[:pilot_count]
        universe_df = universe_df[universe_df['full_name'].isin(unique_repos)]
        logger.info(f"Pilot mode: restricted to {len(unique_repos)} repos.")

    out_path = DATA_DIR / "aidev_pop_universe.parquet"
    universe_df.to_parquet(out_path)
    logger.info(f"Stage 0 complete. Universe has {len(universe_df)} records.")
    return universe_df

# Regex para identificar o merge final de um PR no GitHub ("Merge pull request #N …").
# Esses commits são EXTERNOS (PR branch → base) e devem ser ignorados na busca
# por merges internos (base/outra branch → feature branch).
_PR_MERGE_MSG_RE = re.compile(r'merge pull request\s+#\d+', re.IGNORECASE)

def _get_sha_list_for_pr(repo_path, pr_id, pr_number, pr_commits_series):
    """Retorna lista de SHAs a inspecionar para um PR.

    - Modo pr_commits (AIDev-pop): usa os SHAs pré-coletados.
    - Modo git-log (AIDev completo): varre o histórico git buscando merges
      cujo commit pai é identificável como pertencente ao PR pelo número.
    """
    known_shas = pr_commits_series.dropna().tolist() if pr_commits_series is not None else []
    if known_shas:
        return known_shas

    if not pr_number:
        return []

    # Busca no git log por merges associados a este PR pelo número.
    # "--ancestry-path" limita a ancestrais diretos; "--format=%H %P %s" dá sha, pais e assunto.
    log_bytes = run_git_command(
        repo_path, "log", "--all", "--merges",
        f"--grep=pull request #{int(pr_number)}",
        "--format=%H %s",
        check=False,
    )
    if not log_bytes:
        return []

    candidate_shas = []
    for line in log_bytes.decode("utf-8", "ignore").splitlines():
        parts = line.split(" ", 1)
        if not parts:
            continue
        sha = parts[0].strip()
        subject = parts[1] if len(parts) > 1 else ""
        # O merge final do PR (externo) tem mensagem "Merge pull request #N from …"
        # Queremos *excluir* esse e incluir merges *internos* dentro do PR.
        if _PR_MERGE_MSG_RE.search(subject):
            continue  # é o merge de integração, não interno
        candidate_shas.append(sha)

    return candidate_shas


def process_repository(repo_info: tuple):
    repo_full_name, repo_df = repo_info

    repo_url = repo_df.iloc[0]['repo_url']
    if repo_url.startswith("https://api.github.com/repos/"):
        repo_url = repo_url.replace("https://api.github.com/repos/", "https://github.com/")
        if not repo_url.endswith(".git"):
            repo_url += ".git"

    repo_path = clone_repo_bare(repo_url, SCRATCH_DIR)
    if not repo_path:
        logger.error(f"Não foi possível clonar ou atualizar o repositório {repo_full_name} (Possível timeout na rede).")
        return repo_full_name, [], [], [], []

    internal_merges = []
    conflict_chunks = []
    resolved_chunks = []
    classified_chunks = []

    has_sha_col = 'sha' in repo_df.columns

    prs = repo_df.groupby('pr_id')
    for pr_id, pr_data in prs:
        pr_number = pr_data.iloc[0].get('number') if 'number' in pr_data.columns else None
        sha_series = pr_data['sha'] if has_sha_col else None
        commits = _get_sha_list_for_pr(repo_path, pr_id, pr_number, sha_series)
        for sha in commits:
            try:
                log_output_bytes = run_git_command(repo_path, "cat-file", "-p", sha, check=False)
                if not log_output_bytes: continue
                log_output = log_output_bytes.decode("utf-8", "ignore")
                
                parts = log_output.split("\n\n", 1)
                metadata = parts[0]
                message = parts[1] if len(parts) > 1 else ""
                
                parents = [line.split(" ")[1] for line in metadata.split("\n") if line.startswith("parent ")]
                
                if len(parents) == 2:
                    p1, p2 = parents
                    author_line = next((line for line in metadata.split("\n") if line.startswith("author ")), "author Unknown <unk>")
                    committer_line = next((line for line in metadata.split("\n") if line.startswith("committer ")), "committer Unknown <unk>")
                    
                    author = author_line.split("author ")[1].split("<")[0].strip()
                    committer = committer_line.split("committer ")[1].split("<")[0].strip()
                    
                    # Stage 6: Inline resolver classification
                    resolver_type = classify_resolver(author, message)
                    
                    internal_merges.append({
                        "pr_id": pr_id,
                        "merge_sha": sha,
                        "parent1_sha": p1,
                        "parent2_sha": p2,
                        "author": author,
                        "committer": committer,
                        "repo_full_name": repo_full_name,
                        "resolver_type": resolver_type
                    })
                    
                    base_bytes = run_git_command(repo_path, "merge-base", p1, p2, check=False)
                    if not base_bytes: continue
                    base = base_bytes.decode("utf-8", "ignore").strip()
                    if not base: continue
                    
                    tree_output_bytes = run_git_command(repo_path, "merge-tree", base, p1, p2, check=False)
                    if not tree_output_bytes: continue
                    tree_output = tree_output_bytes.decode("utf-8", "ignore")
                    
                    conflicting_files = MERGE_TREE_CONFLICT_REGEX.finditer(tree_output)
                    
                    for match in conflicting_files:
                        base_blob, path, p1_blob, p2_blob = match.groups()
                        
                        base_content = run_git_command(repo_path, "show", base_blob, check=False)
                        p1_content = run_git_command(repo_path, "show", p1_blob, check=False)
                        p2_content = run_git_command(repo_path, "show", p2_blob, check=False)
                        resolved_content_bytes = run_git_command(repo_path, "show", f"{sha}:{path}", check=False)
                        resolved_content = resolved_content_bytes.decode('utf-8', 'ignore') if resolved_content_bytes else ""
                        
                        conflict_content, has_conflict = run_merge_file(p1_content, base_content, p2_content)
                        if not has_conflict: continue
                        
                        chunks = parse_diff3_chunks(conflict_content)
                        for chunk_idx, chunk in enumerate(chunks):
                            chunk_data = {
                                "pr_id": pr_id,
                                "merge_sha": sha,
                                "file_path": path,
                                "chunk_index": chunk_idx,
                                "v1": chunk['v1'],
                                "base": chunk['base'],
                                "v2": chunk['v2'],
                                "v1_loc": len(remove_empty_lines(chunk['v1'].splitlines())),
                                "v2_loc": len(remove_empty_lines(chunk['v2'].splitlines())),
                                "base_loc": len(remove_empty_lines(chunk['base'].splitlines()))
                            }
                            conflict_chunks.append(chunk_data.copy())
                            
                            resolution, status = find_resolution(chunk["pre_context"], chunk["post_context"], resolved_content)
                            chunk_data["resolution"] = resolution
                            chunk_data["localized_ok"] = (status == "found")
                            chunk_data["resolution_loc"] = len(remove_empty_lines(resolution.splitlines())) if resolution else 0
                            resolved_chunks.append(chunk_data.copy())
                            
                            strategy = identify_resolution(chunk['v1'], chunk['v2'], resolution) if resolution is not None else "Imprecise"
                            chunk_data["strategy"] = strategy
                            classified_chunks.append(chunk_data.copy())
                            
            except Exception as e:
                pass
                
    def remove_readonly(func, path, excinfo):
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)
    shutil.rmtree(repo_path, onerror=remove_readonly)
    return repo_full_name, internal_merges, conflict_chunks, resolved_chunks, classified_chunks


def append_jsonl(filename: str, records: list):
    if not records: return
    with open(DATA_DIR / filename, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def check_processed() -> set:
    tracker = DATA_DIR / "processed_repos.txt"
    if not tracker.exists(): return set()
    with open(tracker, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def mark_processed(repo_name: str):
    with open(DATA_DIR / "processed_repos.txt", "a", encoding="utf-8") as f:
        f.write(f"{repo_name}\n")

def aggregate_jsonl_to_parquet():
    """Converts the incremental jsonl files into the final parquet format."""
    logger.info("Aggregating incremental JSONL files to Parquet...")
    files_map = {
        "internal_merges.jsonl": "internal_merges.parquet",
        "conflict_chunks.jsonl": "conflict_chunks.parquet",
        "resolved_chunks.jsonl": "resolved_chunks.parquet",
        "classified_chunks.jsonl": "classified_chunks.parquet"
    }

    for jsonl_name, parquet_name in files_map.items():
        jsonl_path = DATA_DIR / jsonl_name
        if jsonl_path.exists():
            records = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
            df = pd.DataFrame(records)
            df.to_parquet(DATA_DIR / parquet_name)
            logger.info(f"Converted {jsonl_name} -> {parquet_name} ({len(df)} rows)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=0, help="Run in pilot mode for N repositories")
    parser.add_argument("--full-aidev", action="store_true",
                        help="Expand scope to the full AIDev dataset (all_pull_request + all_repository). "
                             "Repos already in processed_repos.txt are skipped automatically.")
    args = parser.parse_args()

    is_pilot = args.pilot > 0
    universe_df = build_universe(is_pilot, args.pilot, use_full_aidev=args.full_aidev)

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
            name, im, cc, rc, ca = process_repository(rg)
            logger.info(f"[Progresso: {i+1}/{total_repo} ({(i+1)/total_repo:.1%})] Processado {name}")
            append_jsonl("internal_merges.jsonl", im)
            append_jsonl("conflict_chunks.jsonl", cc)
            append_jsonl("resolved_chunks.jsonl", rc)
            append_jsonl("classified_chunks.jsonl", ca)
            mark_processed(name)
    else:
        with multiprocessing.Pool(pool_size) as pool:
            for i, (name, im, cc, rc, ca) in enumerate(pool.imap_unordered(process_repository, repo_groups)):
                logger.info(f"[Progresso: {i+1}/{total_repo} ({(i+1)/total_repo:.1%})] Processado {name}")
                append_jsonl("internal_merges.jsonl", im)
                append_jsonl("conflict_chunks.jsonl", cc)
                append_jsonl("resolved_chunks.jsonl", rc)
                append_jsonl("classified_chunks.jsonl", ca)
                mark_processed(name)

    aggregate_jsonl_to_parquet()

    # Generate standalone resolver_labels.parquet for backwards compatibility with tests
    im_path = DATA_DIR / "internal_merges.parquet"
    if im_path.exists():
        df_im = pd.read_parquet(im_path)
        if not df_im.empty and "resolver_type" in df_im.columns:
            df_im[['pr_id', 'merge_sha', 'parent1_sha', 'parent2_sha', 'author', 'committer', 'repo_full_name', 'resolver_type']].to_parquet(DATA_DIR / "resolver_labels.parquet")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
