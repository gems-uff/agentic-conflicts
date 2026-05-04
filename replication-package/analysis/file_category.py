"""File type categorization heuristics.

Categorizes files into: lock, generated, doc, config, migration, source
Following paper methodology (Section 3).
"""

from pathlib import Path
from typing import List


# Lock file basenames (exact matches)
LOCK_FILES = {
    'package-lock.json',
    'yarn.lock',
    'pnpm-lock.yaml',
    'pnpm-lock.yml',
    'Gemfile.lock',
    'Pipfile.lock',
    'poetry.lock',
    'requirements.lock',
    'composer.lock',
    'Cargo.lock',
    'go.sum',
    'go.mod',
    'mix.lock',
    '.lock',
}

# Generated file patterns (suffixes)
GENERATED_SUFFIXES = {
    '.min.js',
    '.min.css',
    '.bundle.js',
    '.bundle.css',
    '.map',
    '.snap',
    '.snapshot',
    '.coverage',
}

# Generated file directory patterns
GENERATED_DIRS = {
    'dist',
    'build',
    'out',
    'coverage',
    'node_modules',
    '.next',
    'venv',
    'env',
    '.venv',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    'site-packages',
}

# Documentation suffixes
DOC_SUFFIXES = {
    '.md',
    '.rst',
    '.txt',
    '.adoc',
    '.asciidoc',
    '.tex',
    '.org',
    '.markdown',
}

# Config file suffixes
CONFIG_SUFFIXES = {
    '.json',
    '.yaml',
    '.yml',
    '.toml',
    '.ini',
    '.cfg',
    '.conf',
    '.config',
    '.properties',
    '.env',
    '.env.example',
    '.env.local',
    '.xml',
}

# Config file basenames (exact matches)
CONFIG_FILES = {
    '.eslintrc',
    '.eslintrc.js',
    '.eslintrc.json',
    '.prettierrc',
    '.prettierrc.json',
    '.prettierrc.js',
    '.editorconfig',
    'babel.config.js',
    'next.config.js',
    'webpack.config.js',
    'jest.config.js',
    'tsconfig.json',
    'Dockerfile',
    'docker-compose.yml',
    'docker-compose.yaml',
}

# Migration file patterns
MIGRATION_SUFFIXES = {
    '.sql',
}

MIGRATION_DIRS = {
    'migrations',
    'db/migrations',
    'database/migrations',
}


def categorize_filepath(filepath: str) -> str:
    """Categorize file by type.

    Priority order:
    1. Lock files (exact basename match)
    2. Generated files (dir or suffix patterns)
    3. Documentation (suffix)
    4. Configuration (suffix or basename)
    5. Migrations (suffix or dir)
    6. Source code (default)

    Args:
        filepath: Path to file

    Returns:
        Category: 'lock', 'generated', 'doc', 'config', 'migration', or 'source'
    """
    if not isinstance(filepath, str):
        return 'source'

    path = Path(filepath)
    basename = path.name
    parts = [p.lower() for p in path.parts]
    filepath_lower = filepath.lower()

    # Priority 1: Lock files (exact basename match)
    if basename in LOCK_FILES:
        return 'lock'

    # Priority 2: Generated files (directory or suffix)
    if any(d in parts for d in GENERATED_DIRS):
        return 'generated'
    if any(filepath_lower.endswith(suffix) for suffix in GENERATED_SUFFIXES):
        return 'generated'

    # Priority 3: Documentation
    if any(filepath_lower.endswith(suffix) for suffix in DOC_SUFFIXES):
        return 'doc'

    # Priority 4: Configuration (basename or suffix)
    if basename in CONFIG_FILES:
        return 'config'
    if any(filepath_lower.endswith(suffix) for suffix in CONFIG_SUFFIXES):
        return 'config'

    # Priority 5: Migrations (directory or suffix)
    if any(d in parts for d in MIGRATION_DIRS):
        return 'migration'
    if any(filepath_lower.endswith(suffix) for suffix in MIGRATION_SUFFIXES):
        return 'migration'

    # Default: Source code
    return 'source'


def categorize_dataframe(
    df: 'pd.DataFrame',
    filepath_col: str = 'file_path'
) -> 'pd.DataFrame':
    """Add file_category column to DataFrame.

    Args:
        df: Input DataFrame with file_path column
        filepath_col: Name of column containing file paths

    Returns:
        DataFrame with added 'file_category' column
    """
    import pandas as pd

    df_out = df.copy()
    if filepath_col not in df_out.columns:
        df_out['file_category'] = 'source'
    else:
        df_out['file_category'] = df_out[filepath_col].apply(categorize_filepath)
    return df_out


__all__ = [
    'categorize_filepath',
    'categorize_dataframe',
    'LOCK_FILES',
    'GENERATED_SUFFIXES',
    'GENERATED_DIRS',
    'DOC_SUFFIXES',
    'CONFIG_SUFFIXES',
    'MIGRATION_SUFFIXES',
]
