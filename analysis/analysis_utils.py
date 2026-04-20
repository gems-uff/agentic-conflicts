import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nature_of_agent_conflicts")

def load_universe():
    df = pd.read_parquet(os.path.join(DATA_DIR, "aidev_pop_universe.parquet"))
    # We only need PR-level attributes for stratification
    pr_attrs = df[['pr_id', 'agent', 'language', 'pr_task_type']].drop_duplicates(subset=['pr_id'])
    
    # Clean up and normalize
    pr_attrs['agent'] = pr_attrs['agent'].fillna('Unknown').astype(str)
    pr_attrs['language'] = pr_attrs['language'].fillna('Unknown').astype(str)
    pr_attrs['pr_task_type'] = pr_attrs['pr_task_type'].fillna('Unknown').astype(str)
    
    return pr_attrs

def load_classified_chunks():
    df = pd.read_parquet(os.path.join(DATA_DIR, "classified_chunks.parquet"))
    pr_attrs = load_universe()
    resolver_df = pd.read_parquet(os.path.join(DATA_DIR, "resolver_labels.parquet"))[['pr_id', 'merge_sha', 'resolver_type']]
    df = df.merge(pr_attrs, on='pr_id', how='left')
    df = df.merge(resolver_df, on=['pr_id', 'merge_sha'], how='left')
    return df

def load_resolver_labels():
    df = pd.read_parquet(os.path.join(DATA_DIR, "resolver_labels.parquet"))
    pr_attrs = load_universe()
    return df.merge(pr_attrs, on='pr_id', how='left')

def load_internal_merges():
    df = pd.read_parquet(os.path.join(DATA_DIR, "internal_merges.parquet"))
    pr_attrs = load_universe()
    return df.merge(pr_attrs, on='pr_id', how='left')

def save_plot(fig, filename, directory="results"):
    os.makedirs(directory, exist_ok=True)
    fig.savefig(os.path.join(directory, filename), bbox_inches='tight', dpi=300)

def save_table(df, name, directory="results"):
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, f"{name}.csv"), index=False)
    # Also save as markdown for easy viewing
    with open(os.path.join(directory, f"{name}.md"), "w") as f:
        f.write(df.to_markdown(index=False))
