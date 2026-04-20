import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_utils import load_classified_chunks, save_plot, save_table

def compute_summary_stats(series):
    return {
        'count': series.count(),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        '25%': series.quantile(0.25),
        'median': series.median(),
        '75%': series.quantile(0.75),
        '90%': series.quantile(0.90),
        'max': series.max()
    }

def analyze_metric(df, metric_col, title, filename_prefix, directory="results/rq1"):
    # Output Global Stats
    global_stats = pd.DataFrame([compute_summary_stats(df[metric_col])])
    global_stats.insert(0, 'strata', 'global')
    global_stats.insert(1, 'category', 'all')
    
    # Plot Global (Boxplot)
    for fliers, suffix in [(True, 'with_outliers'), (False, 'no_outliers')]:
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.boxplot(x=df[metric_col], ax=ax, showfliers=fliers)
        ax.set_title(f"Global Distribution: {title}" + (" (No Outliers)" if not fliers else ""))
        save_plot(fig, f"{filename_prefix}_global_{suffix}.png", directory)
        plt.close(fig)

    all_stats = [global_stats]
    
    # Stratified stats
    for strata in ['agent', 'language', 'pr_task_type']:
        group_stats = df.groupby(strata)[metric_col].apply(compute_summary_stats).unstack()
        group_stats = group_stats.reset_index()
        group_stats.insert(0, 'strata', strata)
        group_stats.rename(columns={strata: 'category'}, inplace=True)
        all_stats.append(group_stats)
        
        # Plot Stratified
        top_cats = df[strata].value_counts().nlargest(10).index
        plot_df = df[df[strata].isin(top_cats)]
        
        for fliers, suffix in [(True, 'with_outliers'), (False, 'no_outliers')]:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=metric_col, y=strata, data=plot_df, order=top_cats, showfliers=fliers)
            plt.title(f"{title} by {strata.capitalize()}" + (" (No Outliers)" if not fliers else ""))
            plt.tight_layout()
            save_plot(plt.gcf(), f"{filename_prefix}_by_{strata}_{suffix}.png", directory)
            plt.close()

    final_stats_df = pd.concat(all_stats, ignore_index=True)
    save_table(final_stats_df, f"{filename_prefix}_stats", directory)

def main():
    print("Loading chunks for RQ1...")
    df = load_classified_chunks()

    if df.empty:
        print("No classified chunks found. Make sure extraction has run.")
        return

    print("Analyzing # chunks per merge...")
    # Group by merge_sha to get chunks per merge
    # But wait, we need agent, language etc. We can group by merge_sha and take first() for the strata.
    merge_df = df.groupby(['pr_id', 'merge_sha']).agg({
        'chunk_index': 'count',
        'agent': 'first',
        'language': 'first',
        'pr_task_type': 'first'
    }).rename(columns={'chunk_index': 'num_chunks'}).reset_index()

    analyze_metric(merge_df, 'num_chunks', 'Chunks per Merge', 'num_chunks')

    print("Analyzing v1_loc...")
    analyze_metric(df, 'v1_loc', 'V1 LOC per Chunk', 'v1_loc')
    
    print("Analyzing v2_loc...")
    analyze_metric(df, 'v2_loc', 'V2 LOC per Chunk', 'v2_loc')
    
    print("Analyzing resolution_loc...")
    analyze_metric(df, 'resolution_loc', 'Resolution LOC per Chunk', 'resolution_loc')

    print("RQ1 analysis complete.")

if __name__ == "__main__":
    main()
